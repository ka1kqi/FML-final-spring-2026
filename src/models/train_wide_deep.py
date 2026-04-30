"""Train the Wide & Deep draft win-probability model (v2 pairwise architecture).

Usage:
    python -m src.models.train_wide_deep
    python -m src.models.train_wide_deep --fast-dev-run        # 1 epoch, tiny subset
    python -m src.models.train_wide_deep --epochs 20 --batch-size 512

Inputs:
    data/raw/compositions_s16.csv   (player-row schema with match_id, champion_name,
                                     team_id, position, win, [patch], [gameCreation])

Outputs (under data/processed/draft_models/):
    wide_deep.pt              — best-val-AUC checkpoint
    wide_deep_vocab.json      — champion ↔ id (PAD=0, UNK=1) + role_order
    wide_deep_config.json     — hyperparams the adapter needs to re-instantiate the model
    wide_deep_metrics.json    — train/val/test AUC, accuracy, Brier, log-loss, history,
                                Recall@k / MRR over hidden-pick scoring
    wide_deep_calibrator.pkl  — sklearn IsotonicRegression fitted on val probs (optional)

Training improvements borrowed from feature/draft-pipeline-v2 (without pulling in
any of v2's data pipeline, LightGBM, hybrid stacker, AutoGluon, dashboards, etc.):

    1. v2_pairwise architecture (champion + role embeddings, intra/cross pairwise dots).
    2. Time-aware train/val/test split (gameCreation > patch > random fallback).
    3. PMI + SVD champion embedding pretraining on the train split only.
    4. Per-batch augmentation: partial slot dropout, unknown dropout, side flip.
    5. Listwise ranking auxiliary loss for hidden-slot prediction.
    6. Early stopping on val ROC-AUC (BCE fallback) with best-state-only save.
    7. Isotonic probability calibration on val with both classes.
    8. Eval includes Recall@1/3/5 and MRR with the W&D scorer alone.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.models.wide_deep import (
    LEGACY_ARCH,
    PAD_ID,
    PAD_TOKEN,
    ROLE_ORDER,
    UNK_ID,
    UNK_TOKEN,
    V2_ARCH,
    WideDeepDraftNet,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "draft_models"

DEFAULT_CONFIG = {
    "model_name": "wide_deep_draft_v2_pairwise",
    "architecture": V2_ARCH,
    "embedding_dim": 32,
    "hidden_dim": 128,
    "hidden_dims": [128, 64],   # only used by legacy_flat_mlp (kept for adapter back-compat)
    "dropout": 0.2,
    "combine": "sum",
    "target": "blue_win",
    "output": "blue_win_prob",
    "batch_size": 256,
    "epochs": 30,
    "patience": 5,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "partial_slot_dropout_p": 0.10,
    "unk_dropout_p": 0.05,
    "side_flip_p": 0.5,
    "ranking_weight": 0.1,
    "ranking_negatives": 31,
    "pretrain_embeddings": True,
    "enable_calibration": True,
    "val_size": 0.15,
    "test_size": 0.15,
    "seed": 42,
}

# Hyperparam keys persisted for the adapter to round-trip.
_PERSISTED_CONFIG_KEYS = {
    "model_name", "architecture", "embedding_dim", "hidden_dims", "hidden_dim",
    "dropout", "combine", "target", "output",
}


# ---------------------------------------------------------------------------
# Data: pivot player-row CSV → match-row table.
# ---------------------------------------------------------------------------
def build_match_table(comp_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot player-row CSV into one row per match: 10 champions + blue_win + patch.

    Preserves any timestamp column (``gameCreation`` or ``timestamp``) so the
    splitter can do time-aware ordering.
    """
    df = comp_df.copy()
    df["position"] = df["position"].astype(str).str.upper()
    df = df[df["position"].isin(ROLE_ORDER)]
    df["side"] = df["team_id"].map({100: "blue", 200: "red"})
    df = df.dropna(subset=["side"])

    pivot = (
        df.assign(slot=lambda x: x["side"] + "_" + x["position"])
          .pivot_table(index="match_id", columns="slot", values="champion_name", aggfunc="first")
    )
    needed = [f"{s}_{r}" for s in ("blue", "red") for r in ROLE_ORDER]
    pivot = pivot.dropna(subset=needed)

    blue_win = (
        df[df["side"] == "blue"]
          .groupby("match_id")["win"].first()
          .map(lambda v: int(bool(v)))
    )
    pivot["blue_win"] = blue_win.reindex(pivot.index)

    if "patch" in df.columns:
        pivot["patch"] = df.groupby("match_id")["patch"].first().reindex(pivot.index)
    if "gameCreation" in df.columns:
        pivot["gameCreation"] = df.groupby("match_id")["gameCreation"].first().reindex(pivot.index)
    elif "timestamp" in df.columns:
        pivot["gameCreation"] = df.groupby("match_id")["timestamp"].first().reindex(pivot.index)
    return pivot.reset_index()


def build_vocab(match_df: pd.DataFrame) -> dict:
    """Build the champion ↔ id vocab. PAD=0, UNK=1, real champs from id 2."""
    champs = set()
    for s in ("blue", "red"):
        for r in ROLE_ORDER:
            champs.update(match_df[f"{s}_{r}"].dropna().unique().tolist())
    sorted_champs = sorted(champs)
    champion_to_id = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
    for i, c in enumerate(sorted_champs):
        champion_to_id[c] = i + 2
    return {
        "champion_to_id": champion_to_id,
        "id_to_champion": {v: k for k, v in champion_to_id.items()},
        "pad_token": PAD_TOKEN,
        "unk_token": UNK_TOKEN,
        "role_order": ROLE_ORDER,
    }


def encode_match(row, champion_to_id: dict) -> tuple[list[int], list[int]]:
    blue = [champion_to_id.get(row[f"blue_{r}"], UNK_ID) for r in ROLE_ORDER]
    red = [champion_to_id.get(row[f"red_{r}"], UNK_ID) for r in ROLE_ORDER]
    return blue, red


def encode_df(df: pd.DataFrame, champion_to_id: dict):
    blue = np.zeros((len(df), 5), dtype=np.int64)
    red = np.zeros((len(df), 5), dtype=np.int64)
    for i, row in enumerate(df.itertuples(index=False)):
        row_d = dict(zip(df.columns, row))
        b, r = encode_match(row_d, champion_to_id)
        blue[i] = b
        red[i] = r
    y = df["blue_win"].astype(np.float32).values
    return blue, red, y


# ---------------------------------------------------------------------------
# Splitting: time-aware → patch → random.
# ---------------------------------------------------------------------------
def time_aware_split(match_df: pd.DataFrame, val_size: float, test_size: float, seed: int):
    """Return (train, val, test, split_method)."""
    if "gameCreation" in match_df.columns and match_df["gameCreation"].notna().any():
        try:
            df = match_df.sort_values("gameCreation").reset_index(drop=True)
            n = len(df)
            n_test = int(n * test_size)
            n_val = int(n * val_size)
            n_train = n - n_test - n_val
            return (df.iloc[:n_train].reset_index(drop=True),
                    df.iloc[n_train:n_train + n_val].reset_index(drop=True),
                    df.iloc[n_train + n_val:].reset_index(drop=True),
                    "time_gameCreation")
        except (TypeError, ValueError) as exc:
            logger.warning("gameCreation-based split failed (%s); trying patch.", exc)

    if "patch" in match_df.columns and match_df["patch"].notna().any():
        try:
            patch_minor = match_df["patch"].astype(str).str.split(".").str[1].astype(int)
            order = np.argsort(patch_minor.values)
            df = match_df.iloc[order].reset_index(drop=True)
            n = len(df)
            n_test = int(n * test_size)
            n_val = int(n * val_size)
            n_train = n - n_test - n_val
            return (df.iloc[:n_train].reset_index(drop=True),
                    df.iloc[n_train:n_train + n_val].reset_index(drop=True),
                    df.iloc[n_train + n_val:].reset_index(drop=True),
                    "time_patch")
        except (TypeError, ValueError) as exc:
            logger.warning("patch-based split failed (%s); falling back to random.", exc)

    # Random fallback (stratified on the label so val/test classes are balanced).
    train_val, test = train_test_split(
        match_df, test_size=test_size, random_state=seed, stratify=match_df["blue_win"]
    )
    rel_val = val_size / max(1.0 - test_size, 1e-9)
    train, val = train_test_split(
        train_val, test_size=rel_val, random_state=seed, stratify=train_val["blue_win"]
    )
    return (train.reset_index(drop=True), val.reset_index(drop=True),
            test.reset_index(drop=True), "stratified_random")


# ---------------------------------------------------------------------------
# Dataset with augmentation (slot dropout, unk dropout, side flip).
# ---------------------------------------------------------------------------
class DraftDataset(Dataset):
    def __init__(
        self,
        blue: np.ndarray,
        red: np.ndarray,
        y: np.ndarray,
        partial_slot_dropout_p: float = 0.0,
        unk_dropout_p: float = 0.0,
        side_flip_p: float = 0.0,
        unk_id: int = UNK_ID,
    ) -> None:
        self.blue = torch.tensor(blue, dtype=torch.long)
        self.red = torch.tensor(red, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.slot_dropout_p = partial_slot_dropout_p
        self.unk_dropout_p = unk_dropout_p
        self.side_flip_p = side_flip_p
        self.unk_id = unk_id

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx):
        b = self.blue[idx].clone()
        r = self.red[idx].clone()
        y = self.y[idx]

        if self.slot_dropout_p > 0:
            b_mask = torch.rand(5) < self.slot_dropout_p
            r_mask = torch.rand(5) < self.slot_dropout_p
            b[b_mask] = PAD_ID
            r[r_mask] = PAD_ID

        if self.unk_dropout_p > 0:
            # Apply only to non-PAD slots so we don't collide with slot dropout.
            b_unk = (torch.rand(5) < self.unk_dropout_p) & (b != PAD_ID)
            r_unk = (torch.rand(5) < self.unk_dropout_p) & (r != PAD_ID)
            b[b_unk] = self.unk_id
            r[r_unk] = self.unk_id

        if self.side_flip_p > 0 and torch.rand(1).item() < self.side_flip_p:
            b, r = r, b
            y = 1.0 - y

        return b, r, y


# ---------------------------------------------------------------------------
# PMI + SVD pretraining on the train split.
# ---------------------------------------------------------------------------
def pmi_svd_pretrain(train_blue: np.ndarray, train_red: np.ndarray,
                     num_champions: int, embedding_dim: int) -> np.ndarray:
    """Return [num_champions, embedding_dim] init matrix with PAD/UNK rows zeroed.

    Co-occurrence is built over same-team pairs (both blue and red), giving PMI
    a "champions that play well together" signal. This is a scaled-down version
    of v2's pretraining that avoids the ranking_negatives plumbing.
    """
    cooc = np.zeros((num_champions, num_champions), dtype=np.float64)
    for ids in (train_blue, train_red):
        for row in ids:
            real = [c for c in row if c != PAD_ID]
            for i in range(len(real)):
                for j in range(len(real)):
                    if i != j:
                        cooc[real[i], real[j]] += 1.0

    # PPMI
    total = cooc.sum() + 1e-9
    p_xy = cooc / total
    p_x = cooc.sum(axis=1) / total + 1e-12
    p_y = cooc.sum(axis=0) / total + 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        ppmi = np.log(p_xy / np.outer(p_x, p_y))
    ppmi = np.where(np.isfinite(ppmi) & (ppmi > 0), ppmi, 0.0)

    # SVD truncated to embedding_dim
    try:
        u, s, _ = np.linalg.svd(ppmi, full_matrices=False)
        emb = u[:, :embedding_dim] * np.sqrt(s[:embedding_dim])
    except np.linalg.LinAlgError:
        # SVD divergence on degenerate co-occurrence — fall back to small random.
        emb = np.random.normal(0, 0.05, (num_champions, embedding_dim))

    if emb.shape[1] < embedding_dim:
        # Pad with zeros if vocab is smaller than embedding_dim
        pad = np.zeros((emb.shape[0], embedding_dim - emb.shape[1]))
        emb = np.concatenate([emb, pad], axis=1)
    emb[PAD_ID] = 0.0
    emb[UNK_ID] = 0.0
    return emb.astype(np.float32)


# ---------------------------------------------------------------------------
# Listwise ranking auxiliary loss.
# ---------------------------------------------------------------------------
def _sample_ranking_batch(
    blue: torch.Tensor, red: torch.Tensor, num_champions: int, num_negatives: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """For each sample, hide one non-PAD slot and produce K=1+num_negatives candidates.

    Returns:
        cand_blue: [B, K, 5] LongTensor (the candidate inserted into the hidden slot)
        cand_red:  [B, K, 5] LongTensor
        side_is_blue: [B] bool (True if hidden slot was blue)
    """
    B = blue.shape[0]
    K = 1 + num_negatives
    cand_blue = blue.unsqueeze(1).expand(B, K, 5).clone()
    cand_red = red.unsqueeze(1).expand(B, K, 5).clone()
    side_is_blue = torch.zeros(B, dtype=torch.bool)

    for b in range(B):
        # 50/50 hide a blue or red slot, prefer non-PAD
        team = blue[b] if torch.rand(1).item() < 0.5 else red[b]
        non_pad = (team != PAD_ID).nonzero(as_tuple=False).squeeze(-1)
        if non_pad.numel() == 0:
            continue
        slot = non_pad[torch.randint(0, non_pad.numel(), (1,)).item()].item()
        is_blue = team is blue[b]
        side_is_blue[b] = is_blue
        true_id = int(team[slot].item())

        # Sample num_negatives distinct ids ∉ {PAD, UNK, true, picks already in match}
        used = set(blue[b].tolist()) | set(red[b].tolist()) | {PAD_ID, UNK_ID, true_id}
        negs: list[int] = []
        attempts = 0
        while len(negs) < num_negatives and attempts < 5 * num_negatives:
            cid = int(torch.randint(2, num_champions, (1,)).item())
            if cid not in used and cid not in negs:
                negs.append(cid)
            attempts += 1
        # If we can't find enough distinct negatives, pad with UNK (doesn't affect ranking).
        while len(negs) < num_negatives:
            negs.append(UNK_ID)

        cand_ids = [true_id] + negs
        for k, cid in enumerate(cand_ids):
            if is_blue:
                cand_blue[b, k, slot] = cid
            else:
                cand_red[b, k, slot] = cid

    return cand_blue, cand_red, side_is_blue


def _ranking_loss(
    model: WideDeepDraftNet,
    blue: torch.Tensor,
    red: torch.Tensor,
    num_champions: int,
    num_negatives: int,
) -> torch.Tensor:
    cand_blue, cand_red, side_is_blue = _sample_ranking_batch(
        blue, red, num_champions, num_negatives
    )
    B, K, _ = cand_blue.shape
    flat_b = cand_blue.reshape(B * K, 5)
    flat_r = cand_red.reshape(B * K, 5)
    logits = model(flat_b, flat_r).reshape(B, K)
    # When the hidden slot is red, lower blue-win logit is better → invert sign.
    sign = torch.where(side_is_blue, torch.ones(B), -torch.ones(B)).unsqueeze(-1)
    scores = logits * sign
    target = torch.zeros(B, dtype=torch.long)  # candidate 0 is the truth
    return torch.nn.functional.cross_entropy(scores, target)


# ---------------------------------------------------------------------------
# Eval utilities: AUC / acc / Brier / log_loss + Recall@k / MRR.
# ---------------------------------------------------------------------------
def _safe_auc(y, p):
    if len(set(y.tolist())) < 2:
        return None
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return None


def _safe_log_loss(y, p):
    if len(set(y.tolist())) < 2:
        return None
    try:
        return float(log_loss(y, p, labels=[0, 1]))
    except ValueError:
        return None


def _basic_eval(model: WideDeepDraftNet, blue, red, y) -> dict:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(blue, dtype=torch.long),
                       torch.tensor(red, dtype=torch.long))
        probs = torch.sigmoid(logits).cpu().numpy()
    return {
        "n": int(len(y)),
        "auc": _safe_auc(y, probs),
        "accuracy": float(accuracy_score(y, (probs > 0.5).astype(int))) if len(y) else None,
        "brier": float(brier_score_loss(y, probs)) if len(y) else None,
        "log_loss": _safe_log_loss(y, probs),
    }, probs


def _hidden_pick_metrics(
    model: WideDeepDraftNet, blue, red, num_champions: int, n_eval: int = 256, seed: int = 7
) -> dict:
    """Recall@1/3/5 + MRR over hidden-pick scoring with the W&D model alone.

    For each held-out match, randomly hide one non-PAD slot, score every legal
    champion id (excluding PAD/UNK/already-in-match), and check the rank of the
    true hidden champion.
    """
    if len(blue) == 0:
        return {"recall_at_1": None, "recall_at_3": None, "recall_at_5": None, "mrr": None, "n": 0}

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(blue), size=min(n_eval, len(blue)), replace=False)
    ranks = []
    model.eval()
    with torch.no_grad():
        for i in idx:
            b = blue[i].copy()
            r = red[i].copy()
            # Choose a side and a non-PAD slot to hide.
            team = b if rng.random() < 0.5 else r
            non_pad = np.where(team != PAD_ID)[0]
            if non_pad.size == 0:
                continue
            slot = int(rng.choice(non_pad))
            is_blue = team is b
            true_id = int(team[slot])
            used = set(b.tolist()) | set(r.tolist()) | {PAD_ID, UNK_ID}
            cand_ids = [c for c in range(num_champions) if c not in used] + [true_id]
            cand_arr = np.array(cand_ids, dtype=np.int64)

            cand_blue = np.tile(b, (len(cand_arr), 1))
            cand_red = np.tile(r, (len(cand_arr), 1))
            if is_blue:
                cand_blue[:, slot] = cand_arr
            else:
                cand_red[:, slot] = cand_arr
            logits = model(torch.from_numpy(cand_blue), torch.from_numpy(cand_red)).cpu().numpy()
            scores = logits if is_blue else -logits
            order = np.argsort(-scores)
            true_pos_in_cand = len(cand_arr) - 1  # the appended truth
            rank = int(np.where(order == true_pos_in_cand)[0][0]) + 1
            ranks.append(rank)

    if not ranks:
        return {"recall_at_1": None, "recall_at_3": None, "recall_at_5": None, "mrr": None, "n": 0}
    arr = np.array(ranks)
    return {
        "recall_at_1": float((arr <= 1).mean()),
        "recall_at_3": float((arr <= 3).mean()),
        "recall_at_5": float((arr <= 5).mean()),
        "mrr": float((1.0 / arr).mean()),
        "n": int(len(arr)),
    }


# ---------------------------------------------------------------------------
# Training loop.
# ---------------------------------------------------------------------------
def train_one_run(cfg: dict, args) -> dict:
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_csv = PROJECT_ROOT / "data" / "raw" / "compositions_s16.csv"
    if not raw_csv.exists():
        raise FileNotFoundError(
            f"{raw_csv} not found. Run preprocess_matches first or provide compositions_s16.csv."
        )

    logger.info("Loading %s", raw_csv)
    comp_df = pd.read_csv(raw_csv)
    if args.max_rows:
        comp_df = comp_df.head(args.max_rows)
    match_df = build_match_table(comp_df)
    if args.fast_dev_run:
        match_df = match_df.head(min(512, len(match_df)))
        cfg["epochs"] = 1
        cfg["pretrain_embeddings"] = False
        cfg["enable_calibration"] = False
        cfg["ranking_weight"] = 0.0  # save time
    logger.info("Built %d match rows", len(match_df))

    vocab = build_vocab(match_df)
    train_df, val_df, test_df, split_method = time_aware_split(
        match_df, cfg["val_size"], cfg["test_size"], cfg["seed"]
    )
    logger.info("Split (%s): train=%d val=%d test=%d",
                split_method, len(train_df), len(val_df), len(test_df))

    train_blue, train_red, train_y = encode_df(train_df, vocab["champion_to_id"])
    val_blue, val_red, val_y = encode_df(val_df, vocab["champion_to_id"])
    test_blue, test_red, test_y = encode_df(test_df, vocab["champion_to_id"])

    train_ds = DraftDataset(
        train_blue, train_red, train_y,
        partial_slot_dropout_p=cfg["partial_slot_dropout_p"],
        unk_dropout_p=cfg["unk_dropout_p"],
        side_flip_p=0.0 if args.disable_side_flip else cfg["side_flip_p"],
    )

    num_champions = max(vocab["champion_to_id"].values()) + 1
    model = WideDeepDraftNet(
        num_champions=num_champions,
        embedding_dim=cfg["embedding_dim"],
        hidden_dim=cfg["hidden_dim"],
        hidden_dims=tuple(cfg["hidden_dims"]),
        dropout=cfg["dropout"],
        architecture=cfg["architecture"],
        combine=cfg["combine"],
    )

    if cfg["pretrain_embeddings"] and not args.disable_pretraining:
        try:
            init = pmi_svd_pretrain(train_blue, train_red, num_champions, cfg["embedding_dim"])
            with torch.no_grad():
                model.champion_embedding.weight.copy_(torch.from_numpy(init))
            logger.info("PMI+SVD pretrain initialised champion embeddings")
        except Exception as exc:  # noqa: BLE001
            logger.warning("PMI+SVD pretrain failed (%s); using default init.", exc)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    bce = torch.nn.BCEWithLogitsLoss()

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)

    best_state = None
    best_val_auc = float("-inf")
    best_val_bce = float("inf")
    history = []
    no_improve = 0
    use_ranking = cfg["ranking_weight"] > 0 and cfg["ranking_negatives"] > 0

    for epoch in range(cfg["epochs"]):
        model.train()
        bce_total, rank_total, n = 0.0, 0.0, 0
        for blue, red, y in train_loader:
            opt.zero_grad()
            logits = model(blue, red)
            loss = bce(logits, y)
            if use_ranking:
                rl = _ranking_loss(model, blue, red, num_champions, cfg["ranking_negatives"])
                loss = loss + cfg["ranking_weight"] * rl
                rank_total += float(rl.item()) * y.shape[0]
            loss.backward()
            opt.step()
            bce_total += float(bce(logits, y).item()) * y.shape[0]
            n += y.shape[0]

        train_bce = bce_total / max(n, 1)
        train_rank = rank_total / max(n, 1) if use_ranking else None

        val_metrics, _ = _basic_eval(model, val_blue, val_red, val_y)
        val_auc = val_metrics["auc"]
        val_bce = val_metrics["log_loss"] if val_metrics["log_loss"] is not None else float("inf")

        improved = False
        if val_auc is not None:
            if val_auc > best_val_auc + 1e-6:
                best_val_auc = val_auc
                improved = True
        elif val_bce < best_val_bce - 1e-6:
            best_val_bce = val_bce
            improved = True

        if improved:
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        history.append({
            "epoch": epoch + 1,
            "train_bce": float(train_bce),
            "train_ranking_loss": train_rank,
            "val_auc": val_auc,
            "val_log_loss": val_metrics["log_loss"],
            "val_brier": val_metrics["brier"],
            "improved": improved,
        })
        logger.info(
            "Epoch %d  train_bce=%.4f  val_auc=%s  val_logloss=%s%s",
            epoch + 1, train_bce,
            f"{val_auc:.4f}" if val_auc is not None else "n/a",
            f"{val_metrics['log_loss']:.4f}" if val_metrics["log_loss"] is not None else "n/a",
            "  ★best" if improved else f"  (patience {no_improve}/{cfg['patience']})",
        )

        if no_improve >= cfg["patience"]:
            logger.info("Early stopping triggered after epoch %d", epoch + 1)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- Final eval on each split with best state ----
    train_metrics, _ = _basic_eval(model, train_blue, train_red, train_y)
    val_metrics, val_probs = _basic_eval(model, val_blue, val_red, val_y)
    test_metrics, test_probs = _basic_eval(model, test_blue, test_red, test_y)

    hidden_pick = _hidden_pick_metrics(
        model, test_blue, test_red, num_champions,
        n_eval=64 if args.fast_dev_run else 256,
    )

    # ---- Isotonic calibration on val ----
    calibrator = None
    cal_metrics = {}
    if cfg["enable_calibration"] and not args.disable_calibration and len(set(val_y.tolist())) > 1:
        try:
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(val_probs, val_y)
            cal_test_probs = np.clip(calibrator.predict(test_probs), 1e-6, 1 - 1e-6)
            cal_metrics = {
                "calibrated_brier": float(brier_score_loss(test_y, cal_test_probs)),
                "calibrated_log_loss": _safe_log_loss(test_y, cal_test_probs),
            }
        except (ValueError, RuntimeError) as exc:
            logger.warning("Isotonic calibration failed (%s); skipping", exc)
            calibrator = None
            cal_metrics = {}

    # ---- Save artifacts ----
    persisted_config = {k: v for k, v in cfg.items() if k in _PERSISTED_CONFIG_KEYS}
    metrics_blob = {
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "calibration": cal_metrics or None,
        "hidden_pick": hidden_pick,
        "history": history,
        "config": persisted_config,
        "training_config": cfg,
        "split_method": split_method,
        "dataset_sizes": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "vocab_size": int(num_champions),
        "notes": ("Wide & Deep v2 pairwise predicts blue-side win probability "
                  "from draft composition; calibrator (if present) is isotonic-regression."),
    }

    torch.save(model.state_dict(), OUTPUT_DIR / "wide_deep.pt")
    (OUTPUT_DIR / "wide_deep_vocab.json").write_text(json.dumps(vocab, indent=2))
    (OUTPUT_DIR / "wide_deep_config.json").write_text(json.dumps(persisted_config, indent=2))
    (OUTPUT_DIR / "wide_deep_metrics.json").write_text(json.dumps(metrics_blob, indent=2, default=float))
    if calibrator is not None:
        with open(OUTPUT_DIR / "wide_deep_calibrator.pkl", "wb") as f:
            pickle.dump(calibrator, f)
    else:
        # Remove a stale calibrator from a previous run so the adapter doesn't pick it up.
        stale = OUTPUT_DIR / "wide_deep_calibrator.pkl"
        if stale.exists():
            stale.unlink()
    logger.info("Saved artifacts to %s", OUTPUT_DIR)
    logger.info("Final test metrics: %s", test_metrics)
    logger.info("Hidden-pick metrics (W&D-only ranker): %s", hidden_pick)
    return metrics_blob


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Wide & Deep draft win-probability model.")
    p.add_argument("--epochs", type=int, default=None, help="Override CONFIG['epochs'].")
    p.add_argument("--batch-size", type=int, default=None, help="Override CONFIG['batch_size'].")
    p.add_argument("--max-rows", type=int, default=None, help="Cap input CSV rows for quick runs.")
    p.add_argument("--fast-dev-run", action="store_true",
                   help="1 epoch on a tiny subset; disables pretraining/calibration/ranking.")
    p.add_argument("--disable-side-flip", action="store_true", help="Disable side-flip augmentation.")
    p.add_argument("--disable-calibration", action="store_true", help="Skip isotonic calibration.")
    p.add_argument("--disable-pretraining", action="store_true", help="Skip PMI+SVD pretrain.")
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    cfg = dict(DEFAULT_CONFIG)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    train_one_run(cfg, args)


# Backward-compat alias for any external caller.
CONFIG = DEFAULT_CONFIG


if __name__ == "__main__":
    main()
