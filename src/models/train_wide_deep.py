"""Train the Wide & Deep draft win-probability model.

Usage:
    python -m src.models.train_wide_deep

Inputs:
    data/raw/compositions_s16.csv

Outputs (under data/processed/draft_models/):
    wide_deep.pt
    wide_deep_vocab.json
    wide_deep_config.json
    wide_deep_metrics.json
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from src.models.wide_deep import (
    PAD_ID,
    PAD_TOKEN,
    ROLE_ORDER,
    UNK_ID,
    UNK_TOKEN,
    WideDeepDraftNet,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "draft_models"

CONFIG = {
    "model_name": "wide_deep_draft_v1",
    "embedding_dim": 32,
    "hidden_dims": [128, 64],
    "dropout": 0.2,
    "champion_dropout": 0.15,
    "target": "blue_win",
    "output": "blue_win_prob",
    "batch_size": 256,
    "epochs": 12,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "test_size": 0.2,
    "seed": 42,
}


def build_match_table(comp_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot player-row CSV into one row per match: 10 champions + blue_win + patch."""
    df = comp_df.copy()
    df["position"] = df["position"].astype(str).str.upper()
    df = df[df["position"].isin(ROLE_ORDER)]
    df["side"] = df["team_id"].map({100: "blue", 200: "red"})
    df = df.dropna(subset=["side"])

    # One column per (side, role): "blue_TOP", "red_BOTTOM", ...
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

    patch = df.groupby("match_id")["patch"].first().reindex(pivot.index)
    pivot["patch"] = patch
    return pivot.reset_index()


def build_vocab(match_df: pd.DataFrame) -> dict:
    champs = set()
    for s in ("blue", "red"):
        for r in ROLE_ORDER:
            champs.update(match_df[f"{s}_{r}"].dropna().unique().tolist())
    sorted_champs = sorted(champs)
    champion_to_id = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
    for i, c in enumerate(sorted_champs):
        champion_to_id[c] = i + 2  # 0,1 reserved
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


class DraftDataset(Dataset):
    def __init__(self, blue: np.ndarray, red: np.ndarray, y: np.ndarray, champ_dropout: float = 0.0):
        self.blue = torch.tensor(blue, dtype=torch.long)
        self.red = torch.tensor(red, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.dropout = champ_dropout

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx):
        b = self.blue[idx].clone()
        r = self.red[idx].clone()
        if self.dropout > 0:
            b_mask = torch.rand(5) < self.dropout
            r_mask = torch.rand(5) < self.dropout
            b[b_mask] = PAD_ID
            r[r_mask] = PAD_ID
        return b, r, self.y[idx]


def split_by_match(match_df: pd.DataFrame, test_size: float, seed: int):
    """Time-aware split when patch is available, else random match-id split."""
    if match_df["patch"].notna().sum() > 0:
        try:
            patch_minor = match_df["patch"].astype(str).str.split(".").str[1].astype(int)
            sorted_idx = np.argsort(patch_minor.values)
            n_train = int(len(match_df) * (1 - test_size))
            train_idx = sorted_idx[:n_train]
            test_idx = sorted_idx[n_train:]
            return match_df.iloc[train_idx].reset_index(drop=True), match_df.iloc[test_idx].reset_index(drop=True)
        except Exception as e:  # noqa: BLE001
            logger.warning("Patch-based split failed (%s); falling back to random match split.", e)
    train, test = train_test_split(match_df, test_size=test_size, random_state=seed)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def main() -> None:
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_csv = PROJECT_ROOT / "data" / "raw" / "compositions_s16.csv"
    if not raw_csv.exists():
        raise FileNotFoundError(
            f"{raw_csv} not found. Run preprocess_matches first or provide compositions_s16.csv."
        )

    logger.info("Loading %s", raw_csv)
    comp_df = pd.read_csv(raw_csv)
    match_df = build_match_table(comp_df)
    logger.info("Built %d match rows", len(match_df))

    vocab = build_vocab(match_df)
    train_df, test_df = split_by_match(match_df, CONFIG["test_size"], CONFIG["seed"])
    logger.info("Split: train=%d, test=%d", len(train_df), len(test_df))

    def encode_df(df):
        blue = np.zeros((len(df), 5), dtype=np.int64)
        red = np.zeros((len(df), 5), dtype=np.int64)
        for i, row in enumerate(df.itertuples(index=False)):
            row_d = dict(zip(df.columns, row))
            b, r = encode_match(row_d, vocab["champion_to_id"])
            blue[i] = b
            red[i] = r
        y = df["blue_win"].astype(np.float32).values
        return blue, red, y

    train_blue, train_red, train_y = encode_df(train_df)
    test_blue, test_red, test_y = encode_df(test_df)

    train_ds = DraftDataset(train_blue, train_red, train_y, champ_dropout=CONFIG["champion_dropout"])
    test_ds = DraftDataset(test_blue, test_red, test_y, champ_dropout=0.0)

    num_champions = max(vocab["champion_to_id"].values()) + 1
    model = WideDeepDraftNet(
        num_champions=num_champions,
        embedding_dim=CONFIG["embedding_dim"],
        hidden_dims=tuple(CONFIG["hidden_dims"]),
        dropout=CONFIG["dropout"],
    )
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    loss_fn = torch.nn.BCEWithLogitsLoss()

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    for epoch in range(CONFIG["epochs"]):
        model.train()
        total = 0.0
        n = 0
        for blue, red, y in train_loader:
            opt.zero_grad()
            logits = model(blue, red)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            total += loss.item() * y.shape[0]
            n += y.shape[0]
        logger.info("Epoch %d  train_loss=%.4f", epoch + 1, total / max(n, 1))

    # Evaluate
    model.eval()
    with torch.no_grad():
        preds = torch.sigmoid(model(
            torch.tensor(test_blue, dtype=torch.long),
            torch.tensor(test_red, dtype=torch.long),
        )).numpy()
    metrics = {
        "roc_auc": float(roc_auc_score(test_y, preds)),
        "accuracy": float(accuracy_score(test_y, (preds > 0.5).astype(int))),
        "brier_score": float(brier_score_loss(test_y, preds)),
        "test_size": int(len(test_y)),
        "notes": "Wide & Deep predicts blue-side win probability from draft composition.",
    }
    logger.info("Test metrics: %s", metrics)

    torch.save(model.state_dict(), OUTPUT_DIR / "wide_deep.pt")
    (OUTPUT_DIR / "wide_deep_vocab.json").write_text(json.dumps(vocab, indent=2))
    (OUTPUT_DIR / "wide_deep_config.json").write_text(json.dumps({
        k: v for k, v in CONFIG.items()
        if k in {"model_name", "embedding_dim", "hidden_dims", "dropout",
                 "champion_dropout", "target", "output"}
    }, indent=2))
    (OUTPUT_DIR / "wide_deep_metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info("Saved artifacts to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
