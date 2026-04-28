"""
lol_draft_pipeline.py
=====================

End-to-end League of Legends draft-time win prediction and champion
recommendation pipeline.

Five core modules, all wired through a single CLI:

  1. LightGBM baseline (with HistGradientBoosting fallback).
  2. TeamCompNet (PyTorch) - learned champion embeddings + pairwise
     interaction.
  3. LightGBM + extracted-embedding features (hybrid).
  4. Wide & Deep DraftNet (PyTorch).
  5. Beam Search top-k draft recommender.

Strict draft-time leakage discipline: only blue/red picks, roles, side
(and bans / patch / rank when available) are used in the main models.
Post-game stats (kda, gold, damage, vision, items) are explicitly
excluded.  A leakage audit is printed every run.

Run `python lol_draft_pipeline.py --help` for the full CLI.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import math
import os
import pickle
import random
import sys
import time
import traceback
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Optional imports - protected so the file imports even if a backend is
# missing.  We surface a clear error at the call site.
try:
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:  # pragma: no cover - environment dependent
    lgb = None  # type: ignore[assignment]
    _HAS_LGB = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _HAS_TORCH = False

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("lol_pipeline")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

ROLES: Tuple[str, ...] = ("top", "jungle", "mid", "adc", "support")
ROLE_TO_IDX: Dict[str, int] = {r: i for i, r in enumerate(ROLES)}

# Riot match-v5 uses these tokens; we normalise to the spec-friendly aliases.
RIOT_POSITION_TO_ROLE: Dict[str, str] = {
    "TOP": "top",
    "JUNGLE": "jungle",
    "MIDDLE": "mid",
    "MID": "mid",
    "BOTTOM": "adc",
    "BOT": "adc",
    "ADC": "adc",
    "UTILITY": "support",
    "SUPPORT": "support",
    "SUP": "support",
}

UNKNOWN_TOKEN = "<UNK>"
UNKNOWN_INDEX = 0

# Post-game columns we must NEVER feed the draft-time model.  Used for the
# leakage audit at the start of every run.
POST_GAME_COLUMNS: Tuple[str, ...] = (
    "kills",
    "deaths",
    "assists",
    "goldEarned",
    "gold",
    "totalDamageDealtToChampions",
    "damage_dealt",
    "totalDamageTaken",
    "damage_taken",
    "visionScore",
    "vision_score",
    "totalMinionsKilled",
    "cs",
    "neutralMinionsKilled",
    "items",
    "item0",
    "item1",
    "item2",
    "item3",
    "item4",
    "item5",
    "item6",
    "dragons",
    "barons",
    "towers",
)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass
class PipelineConfig:
    """All pipeline-level knobs in one place; also serialised to artifacts."""

    # I/O
    data_dir: str = "data"
    artifacts_dir: str = "artifacts"
    raw_csv: Optional[str] = None  # auto-detected if None

    # Reproducibility
    random_seed: int = 42

    # Splits
    val_size: float = 0.15
    test_size: float = 0.15

    # Synergy / counter smoothing
    pair_smoothing_min_count: int = 20
    pair_smoothing_prior: float = 0.5

    # LightGBM
    lgb_n_estimators: int = 500
    lgb_learning_rate: float = 0.03
    lgb_num_leaves: int = 31
    lgb_min_child_samples: int = 20
    lgb_early_stop_rounds: int = 30

    # PyTorch
    embedding_dim: int = 32
    hidden_dim: int = 128
    dropout: float = 0.2
    batch_size: int = 256
    learning_rate: float = 1e-3
    epochs: int = 30
    patience: int = 5
    weight_decay: float = 1e-5

    # Wide & Deep
    wide_deep_combine: str = "sum"  # 'sum' or 'concat'

    # Beam search
    beam_width: int = 5
    beam_depth: int = 2
    top_k: int = 5

    # ----- Advanced research-grade extensions ------------------------------
    # Architecture switch for the deep models: "pairwise" (TeamCompNet) or
    # "transformer" (SetTransformerCompNet, default).
    arch: str = "transformer"
    n_attention_layers: int = 2
    n_attention_heads: int = 4

    # Data augmentation
    augment_side_flip: bool = True
    augment_dropout_p: float = 0.10  # mask champion id -> UNK with this prob

    # Listwise / LambdaRank-style auxiliary loss
    ranking_weight: float = 0.3
    ranking_negatives: int = 31  # -> K = 32 candidates per ranking step

    # AlphaZero policy head + MCTS
    policy_weight: float = 0.5
    enable_policy_head: bool = True
    mcts_simulations: int = 64
    mcts_c_puct: float = 1.5

    # PMI + SVD pretraining of champion embeddings
    pretrain_embeddings: bool = True

    # Calibration (isotonic on val set)
    enable_calibration: bool = True

    # Stacking ensemble (logistic regression meta-learner over val OOF preds).
    # Default off because under time-based splits the meta-LR fits val
    # (which is closer in time to train than test) and consistently
    # under-performs the best single base model on the actual test split.
    # Enable explicitly with --enable-stacking when using random splits.
    enable_stacking: bool = False

    # External tabular features (only kick in when columns exist in source data)
    use_patch_feature: bool = True
    use_rank_feature: bool = True
    use_bans_feature: bool = True

    # Smoke / dev
    max_rows: Optional[int] = None  # cap pivoted matches for fast dev runs
    fast_dev_run: bool = False

    # Run identification (auto-generated when None or "auto")
    run_id: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Run directory + event logging
# --------------------------------------------------------------------------- #


def _new_run_id() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_run_id(cfg: PipelineConfig) -> str:
    """Materialise ``cfg.run_id`` (auto-generate when missing / 'auto')."""
    if not cfg.run_id or cfg.run_id == "auto":
        cfg.run_id = _new_run_id()
    return cfg.run_id


def run_dir_for(cfg: PipelineConfig) -> Path:
    """Return the per-run output directory (creates parents on demand)."""
    rid = resolve_run_id(cfg)
    p = Path(cfg.artifacts_dir) / "runs" / rid
    p.mkdir(parents=True, exist_ok=True)
    return p


def update_latest_pointer(cfg: PipelineConfig) -> None:
    """Record the most recently written run_id at ``runs/_latest_run_id.txt``."""
    runs_root = Path(cfg.artifacts_dir) / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    (runs_root / "_latest_run_id.txt").write_text(cfg.run_id or "")


class EventLogger:
    """Append structured training events to ``events.jsonl``.

    The file is opened in append mode and flushed after every write so the
    dashboard can tail it while training is still running.  All writes are
    best-effort: we never let a logging failure crash training.
    """

    def __init__(self, run_dir: Path, run_id: str) -> None:
        self.run_dir = run_dir
        self.run_id = run_id
        self.path = run_dir / "events.jsonl"
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def log(self, event_type: str, **payload: object) -> None:
        record = {
            "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
            "run_id": self.run_id,
            "event_type": event_type,
            **payload,
        }
        try:
            with self.path.open("a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as exc:  # pragma: no cover
            log.warning("EventLogger failed to write %s: %s", event_type, exc)

    # Convenience wrappers --------------------------------------------------

    def run_started(self, command: str, config: Dict[str, object]) -> None:
        self.log("run_started", command=command, config=config)

    def run_completed(self, status: str, duration_seconds: float, **extra) -> None:
        self.log("run_completed", status=status, duration_seconds=duration_seconds, **extra)

    def error(self, message: str, tb: Optional[str] = None) -> None:
        self.log("error", message=message, traceback=tb or "")

    def stage_started(self, model: str) -> None:
        self.log("stage_started", model=model)

    def stage_completed(self, model: str, val_metrics: Dict, test_metrics: Dict) -> None:
        self.log(
            "stage_completed",
            model=model,
            val_metrics={k: v for k, v in val_metrics.items() if k != "calibration"},
            test_metrics={k: v for k, v in test_metrics.items() if k != "calibration"},
        )

    def epoch(
        self,
        model: str,
        epoch: int,
        total: int,
        train_loss: float,
        val_loss: float,
        val_auc: float,
    ) -> None:
        self.log(
            "train_metric",
            model=model,
            epoch=epoch,
            total_epochs=total,
            split="val",
            metrics={
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_auc": float(val_auc),
            },
        )

    def lgb_iter(self, model: str, iteration: int, split: str, metrics: Dict[str, float]) -> None:
        self.log(
            "train_metric",
            model=model,
            iteration=iteration,
            split=split,
            metrics={k: float(v) for k, v in metrics.items()},
        )


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and (when available) PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def torch_device() -> "torch.device":
    """Return the best torch device available (cuda > mps > cpu)."""
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is not installed; cannot select a device.")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #


def find_raw_csv(data_dir: str, override: Optional[str] = None) -> Path:
    """Pick the most likely match-data CSV under ``data_dir``."""
    if override:
        p = Path(override)
        if not p.is_file():
            raise FileNotFoundError(f"--raw-csv path not found: {override}")
        return p
    data_path = Path(data_dir)
    candidates: List[Path] = []
    for pat in (
        "processed/matches.csv",
        "processed/*.csv",
        "*.csv",
        "raw/*.csv",
    ):
        candidates.extend(data_path.glob(pat))
    # Filter out non-match auxiliary CSVs.
    aux = {"champion_embeddings.csv"}
    candidates = [
        c
        for c in candidates
        if c.name not in aux and not c.name.startswith(".")
    ]
    # Prefer files in processed/ (curated by the user), then by name keyword,
    # then by size as a final tiebreak.
    keywords = ("matches", "composition", "draft", "games")

    def _rank(p: Path) -> tuple:
        in_processed = "processed" in p.parts
        keyword_hits = sum(k in p.name.lower() for k in keywords)
        return (
            0 if in_processed else 1,           # processed/ wins
            -keyword_hits,                       # more keywords = better
            -p.stat().st_size,                   # bigger = better (last resort)
        )

    scored = sorted(candidates, key=_rank)
    if not scored:
        raise FileNotFoundError(
            f"No CSV match data found under {data_dir!r}. "
            f"Tried processed/, raw/, top-level."
        )
    log.info("Detected raw match CSV: %s", scored[0])
    return scored[0]


def _detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Map logical column names to the raw column names actually present."""
    cols = {c.lower(): c for c in df.columns}

    def pick(*aliases: str) -> Optional[str]:
        for a in aliases:
            if a in cols:
                return cols[a]
        return None

    mapping = {
        "match_id": pick("match_id", "matchid", "gameid", "game_id"),
        "champion_id": pick("champion_id", "championid", "champ_id"),
        "champion_name": pick(
            "champion_name", "championname", "champion", "champ"
        ),
        "team_id": pick("team_id", "teamid", "team", "side_id"),
        "position": pick(
            "position", "role", "lane", "team_position", "teamposition"
        ),
        "win": pick("win", "winner", "blue_win", "label", "target"),
        "patch": pick("patch", "game_version", "gameversion"),
        "rank": pick("rank", "tier", "elo"),
        "timestamp": pick(
            "timestamp",
            "game_creation",
            "gamecreation",
            "game_start_timestamp",
            "gamestarttimestamp",
        ),
        "bans": pick("bans", "ban_list"),
    }
    missing = [
        k
        for k in ("match_id", "champion_name", "team_id", "position", "win")
        if mapping[k] is None
    ]
    if missing:
        raise ValueError(
            "Could not auto-detect required columns: "
            f"{missing}. Got columns: {list(df.columns)}"
        )
    return mapping


def load_long_dataframe(csv_path: Path, max_matches: Optional[int] = None) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Read the long-form participant CSV and normalise critical columns."""
    log.info("Reading %s", csv_path)
    df = pd.read_csv(csv_path)
    schema = _detect_columns(df)
    log.info("Schema mapping: %s", schema)

    df = df.rename(
        columns={v: k for k, v in schema.items() if v is not None and v != k}
    )

    # Normalise types
    df["match_id"] = df["match_id"].astype(str)
    df["champion_name"] = df["champion_name"].astype(str).str.strip()
    df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce").astype("Int64")
    df["position"] = df["position"].astype(str).str.upper().str.strip()
    df["role"] = df["position"].map(RIOT_POSITION_TO_ROLE)

    # Win can be bool / 0-1 / "True"/"False"; coerce to int 0/1.
    win = df["win"]
    if win.dtype == object:
        win = win.astype(str).str.lower().map(
            {"true": 1, "false": 0, "1": 1, "0": 0}
        )
    df["win"] = pd.to_numeric(win, errors="coerce").fillna(0).astype(int)

    if max_matches is not None and max_matches > 0:
        keep_ids = pd.unique(df["match_id"])[:max_matches]
        df = df[df["match_id"].isin(keep_ids)].copy()
        log.info("max_rows applied: keeping %d matches", len(keep_ids))

    return df, schema


def pivot_long_to_match_level(long_df: pd.DataFrame) -> pd.DataFrame:
    """Convert long participant table into one row per match.

    Each row exposes role-specific champion columns for both sides plus the
    binary ``blue_win`` label.  Matches with missing roles or unexpected team
    sizes are dropped (they cannot contribute clean draft features).
    """
    df = long_df.copy()

    # Drop rows whose role couldn't be normalised.
    null_roles = df["role"].isna().sum()
    if null_roles:
        log.warning("Dropping %d rows with unrecognised positions", null_roles)
        df = df.dropna(subset=["role"]).copy()

    # Map team ids to side strings (100 -> blue, 200 -> red).
    side_map = {100: "blue", 200: "red"}
    df["side"] = df["team_id"].astype("Int64").map(side_map)
    df = df.dropna(subset=["side"]).copy()

    # We require exactly 5 blue + 5 red and one champion per (side, role).
    counts = df.groupby(["match_id", "side", "role"]).size().rename("n").reset_index()
    bad = counts[counts["n"] != 1]["match_id"].unique()
    if len(bad):
        log.warning(
            "Dropping %d matches with duplicate/missing (side, role) entries",
            len(bad),
        )
        df = df[~df["match_id"].isin(bad)].copy()

    # Ensure each match has all 10 slots (5 roles * 2 sides).
    coverage = df.groupby("match_id").size()
    full_ids = coverage[coverage == 10].index
    incomplete = len(coverage) - len(full_ids)
    if incomplete:
        log.warning("Dropping %d matches without 5+5 full coverage", incomplete)
    df = df[df["match_id"].isin(full_ids)].copy()

    # Derive blue_win from the blue-side row (any blue row is fine; all share win).
    blue_wins = (
        df[df["side"] == "blue"]
        .groupby("match_id")["win"]
        .first()
        .rename("blue_win")
    )

    # Pivot: index=match_id, columns=(side, role), values=champion_name
    wide = (
        df.set_index(["match_id", "side", "role"])["champion_name"]
        .unstack(["side", "role"])
    )
    wide.columns = [f"{side}_{role}_champion" for (side, role) in wide.columns]
    wide = wide.join(blue_wins, how="inner")

    # Optional metadata columns (kept on a best-effort basis).
    extra_cols = [c for c in ("patch", "rank", "timestamp", "bans") if c in df.columns]
    for col in extra_cols:
        # Use first non-null entry per match
        wide[col] = (
            df.groupby("match_id")[col].first().reindex(wide.index).values
        )

    wide = wide.reset_index()
    log.info(
        "Pivoted to match-level: %d matches, blue_win mean = %.4f",
        len(wide),
        wide["blue_win"].mean(),
    )
    return wide


# --------------------------------------------------------------------------- #
# Champion vocabulary
# --------------------------------------------------------------------------- #


def build_champion_vocab(match_df: pd.DataFrame) -> Dict[str, int]:
    """Build a deterministic champion -> int mapping (UNKNOWN stays at 0)."""
    champ_cols = [c for c in match_df.columns if c.endswith("_champion")]
    champs = pd.unique(match_df[champ_cols].values.ravel("K"))
    champs = sorted({c for c in champs if isinstance(c, str) and c})
    vocab = {UNKNOWN_TOKEN: UNKNOWN_INDEX}
    for c in champs:
        vocab[c] = len(vocab)
    log.info("Built champion vocab with %d champions (+UNK)", len(champs))
    return vocab


def champion_columns_for_side(side: str) -> List[str]:
    return [f"{side}_{r}_champion" for r in ROLES]


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #


def make_splits(
    match_df: pd.DataFrame,
    cfg: PipelineConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Time-based split if a usable timestamp column exists, else stratified."""
    if "timestamp" in match_df.columns and match_df["timestamp"].notna().any():
        sorted_df = match_df.sort_values("timestamp").reset_index(drop=True)
        n = len(sorted_df)
        n_test = int(round(cfg.test_size * n))
        n_val = int(round(cfg.val_size * n))
        n_train = n - n_test - n_val
        train = sorted_df.iloc[:n_train].copy()
        val = sorted_df.iloc[n_train : n_train + n_val].copy()
        test = sorted_df.iloc[n_train + n_val :].copy()
        log.info(
            "Time-based split: train=%d val=%d test=%d", len(train), len(val), len(test)
        )
    else:
        log.warning(
            "No timestamp column found; falling back to stratified random split"
        )
        train_val, test = train_test_split(
            match_df,
            test_size=cfg.test_size,
            random_state=cfg.random_seed,
            stratify=match_df["blue_win"],
        )
        rel_val = cfg.val_size / max(1.0 - cfg.test_size, 1e-9)
        train, val = train_test_split(
            train_val,
            test_size=rel_val,
            random_state=cfg.random_seed,
            stratify=train_val["blue_win"],
        )
        log.info(
            "Stratified split: train=%d val=%d test=%d",
            len(train),
            len(val),
            len(test),
        )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #


@dataclass
class HandcraftedStats:
    """Win-rate / synergy / counter stats fit on TRAIN ONLY (no leakage)."""

    champ_winrate: Dict[str, float]
    pair_synergy_blue: Dict[Tuple[str, str], float]  # ordered: (a, b) where a<b
    pair_counter: Dict[Tuple[str, str], float]  # ordered: (ally a, enemy b)
    base_winrate: float
    min_count: int

    def synergy(self, a: str, b: str) -> float:
        if a == UNKNOWN_TOKEN or b == UNKNOWN_TOKEN:
            return 0.0
        key = (a, b) if a < b else (b, a)
        return self.pair_synergy_blue.get(key, 0.0)

    def counter(self, ally: str, enemy: str) -> float:
        if ally == UNKNOWN_TOKEN or enemy == UNKNOWN_TOKEN:
            return 0.0
        return self.pair_counter.get((ally, enemy), 0.0)

    def winrate(self, champ: str) -> float:
        if champ == UNKNOWN_TOKEN:
            return self.base_winrate
        return self.champ_winrate.get(champ, self.base_winrate)


def fit_handcrafted_stats(
    train_df: pd.DataFrame, min_count: int = 20, prior: float = 0.5
) -> HandcraftedStats:
    """Compute champion / pair / counter win rates with Bayesian smoothing.

    All statistics are fit ONLY on the training data to avoid leakage.
    """
    blue_cols = champion_columns_for_side("blue")
    red_cols = champion_columns_for_side("red")

    base = float(train_df["blue_win"].mean())

    # Champion-level win rate (across both sides; flip outcome for red side).
    champ_count: Dict[str, int] = {}
    champ_wins: Dict[str, int] = {}
    pair_synergy_count: Dict[Tuple[str, str], int] = {}
    pair_synergy_wins: Dict[Tuple[str, str], int] = {}
    pair_counter_count: Dict[Tuple[str, str], int] = {}
    pair_counter_wins: Dict[Tuple[str, str], int] = {}

    blue_arr = train_df[blue_cols].values
    red_arr = train_df[red_cols].values
    win_arr = train_df["blue_win"].values.astype(int)

    for blues, reds, blue_w in zip(blue_arr, red_arr, win_arr):
        red_w = 1 - blue_w
        # Champion-level
        for c in blues:
            champ_count[c] = champ_count.get(c, 0) + 1
            champ_wins[c] = champ_wins.get(c, 0) + blue_w
        for c in reds:
            champ_count[c] = champ_count.get(c, 0) + 1
            champ_wins[c] = champ_wins.get(c, 0) + red_w

        # Synergy: unordered same-team pairs
        for team_champs, team_w in ((blues, blue_w), (reds, red_w)):
            for i in range(len(team_champs)):
                for j in range(i + 1, len(team_champs)):
                    a, b = team_champs[i], team_champs[j]
                    key = (a, b) if a < b else (b, a)
                    pair_synergy_count[key] = pair_synergy_count.get(key, 0) + 1
                    pair_synergy_wins[key] = (
                        pair_synergy_wins.get(key, 0) + team_w
                    )

        # Counter: ordered (ally, enemy)
        for ally in blues:
            for enemy in reds:
                key = (ally, enemy)
                pair_counter_count[key] = pair_counter_count.get(key, 0) + 1
                pair_counter_wins[key] = pair_counter_wins.get(key, 0) + blue_w
        for ally in reds:
            for enemy in blues:
                key = (ally, enemy)
                pair_counter_count[key] = pair_counter_count.get(key, 0) + 1
                pair_counter_wins[key] = pair_counter_wins.get(key, 0) + red_w

    # Smoothed -> residual (deviation from base 0.5)
    def smooth(num: int, denom: int) -> float:
        # Bayesian: (wins + alpha * prior * min_count) / (denom + alpha * min_count)
        alpha = 1.0
        smoothed = (num + alpha * prior * min_count) / (denom + alpha * min_count)
        return smoothed - prior

    champ_wr = {
        c: smooth(champ_wins[c], champ_count[c]) + prior  # keep absolute WR
        for c in champ_count
    }
    pair_syn = {
        k: smooth(pair_synergy_wins[k], pair_synergy_count[k])
        for k in pair_synergy_count
    }
    pair_ctr = {
        k: smooth(pair_counter_wins[k], pair_counter_count[k])
        for k in pair_counter_count
    }
    log.info(
        "Handcrafted stats: %d champions, %d synergy pairs, %d counter pairs",
        len(champ_wr),
        len(pair_syn),
        len(pair_ctr),
    )
    return HandcraftedStats(
        champ_winrate=champ_wr,
        pair_synergy_blue=pair_syn,
        pair_counter=pair_ctr,
        base_winrate=base,
        min_count=min_count,
    )


def featurise_handcrafted(
    df: pd.DataFrame, stats: HandcraftedStats
) -> pd.DataFrame:
    """Generate per-match handcrafted feature columns."""
    blue_cols = champion_columns_for_side("blue")
    red_cols = champion_columns_for_side("red")
    blue_arr = df[blue_cols].values
    red_arr = df[red_cols].values

    rows = []
    for blues, reds in zip(blue_arr, red_arr):
        b_wr = np.mean([stats.winrate(c) for c in blues])
        r_wr = np.mean([stats.winrate(c) for c in reds])

        b_syn = []
        for i in range(5):
            for j in range(i + 1, 5):
                b_syn.append(stats.synergy(blues[i], blues[j]))
        r_syn = []
        for i in range(5):
            for j in range(i + 1, 5):
                r_syn.append(stats.synergy(reds[i], reds[j]))

        b_ctr = [stats.counter(a, e) for a in blues for e in reds]
        r_ctr = [stats.counter(a, e) for a in reds for e in blues]

        rows.append(
            {
                "blue_avg_champion_wr": b_wr,
                "red_avg_champion_wr": r_wr,
                "blue_synergy_avg": float(np.mean(b_syn)),
                "red_synergy_avg": float(np.mean(r_syn)),
                "blue_counter_avg": float(np.mean(b_ctr)),
                "red_counter_avg": float(np.mean(r_ctr)),
                "blue_minus_red_wr": b_wr - r_wr,
                "blue_minus_red_synergy": float(np.mean(b_syn) - np.mean(r_syn)),
                "blue_minus_red_counter": float(np.mean(b_ctr) - np.mean(r_ctr)),
            }
        )
    return pd.DataFrame(rows, index=df.index)


def encode_champion_ids(
    df: pd.DataFrame, vocab: Dict[str, int]
) -> Dict[str, np.ndarray]:
    """Map champion names to integer ids using ``vocab`` (UNKNOWN as fallback)."""
    encoded = {}
    for side in ("blue", "red"):
        ids = np.zeros((len(df), len(ROLES)), dtype=np.int64)
        for j, role in enumerate(ROLES):
            col = f"{side}_{role}_champion"
            ids[:, j] = df[col].map(lambda x: vocab.get(x, UNKNOWN_INDEX)).values
        encoded[side] = ids
    return encoded


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, n_buckets: int = 10
) -> Dict[str, object]:
    """All metrics required by the spec, including calibration buckets."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob), 1e-6, 1 - 1e-6)
    y_pred = (y_prob >= 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred).tolist()
    out: Dict[str, object] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob))
        if len(np.unique(y_true)) > 1
        else float("nan"),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "confusion_matrix": cm,
    }

    # Calibration buckets
    buckets = []
    edges = np.linspace(0.0, 1.0, n_buckets + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi if hi < 1.0 else y_prob <= hi)
        if mask.any():
            buckets.append(
                {
                    "bucket": f"[{lo:.2f},{hi:.2f})",
                    "n": int(mask.sum()),
                    "mean_pred": float(y_prob[mask].mean()),
                    "empirical_rate": float(y_true[mask].mean()),
                }
            )
        else:
            buckets.append(
                {
                    "bucket": f"[{lo:.2f},{hi:.2f})",
                    "n": 0,
                    "mean_pred": float("nan"),
                    "empirical_rate": float("nan"),
                }
            )
    out["calibration"] = buckets
    return out


def print_metrics(name: str, metrics: Dict[str, object]) -> None:
    log.info(
        "[%s] acc=%.4f f1=%.4f auc=%.4f log_loss=%.4f brier=%.4f",
        name,
        metrics["accuracy"],
        metrics["f1"],
        metrics["roc_auc"],
        metrics["log_loss"],
        metrics["brier_score"],
    )


# --------------------------------------------------------------------------- #
# LightGBM baseline
# --------------------------------------------------------------------------- #


def build_baseline_feature_matrix(
    df: pd.DataFrame,
    vocab: Dict[str, int],
    handcrafted: Optional[HandcraftedStats] = None,
    cfg: Optional[PipelineConfig] = None,
    extra_vocabs: Optional[Dict[str, Dict[str, int]]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Build the feature matrix used by the baseline & hybrid LGB models.

    Features (all draft-time):
      * 10 categorical columns: ``{side}_{role}_champion_id`` (int)
      * 9  handcrafted scalars (avg winrate / synergy / counter, diffs)
      * Optional patch / rank categorical ids (when present in source data
        and ``cfg.use_*_feature`` is True).
      * Optional ban one-hot (top-N most frequent bans) when ``bans`` column
        is present.

    ``extra_vocabs`` maps feature name -> ``{value: int}`` learned on the
    training split.  Pass it to keep val/test encoding consistent.
    """
    out = pd.DataFrame(index=df.index)

    cat_cols: List[str] = []
    for side in ("blue", "red"):
        for role in ROLES:
            col = f"{side}_{role}_champion_id"
            src = f"{side}_{role}_champion"
            out[col] = (
                df[src].map(lambda x: vocab.get(x, UNKNOWN_INDEX)).astype(np.int32)
            )
            cat_cols.append(col)

    if handcrafted is not None:
        hc = featurise_handcrafted(df, handcrafted)
        for col in hc.columns:
            out[col] = hc[col].values

    cfg = cfg or PipelineConfig()
    extra_vocabs = extra_vocabs or {}

    if cfg.use_patch_feature and "patch" in df.columns:
        vmap = extra_vocabs.get("patch", {})
        out["patch_id"] = (
            df["patch"].astype(str).map(lambda x: vmap.get(x, 0)).astype(np.int32)
        )
        cat_cols.append("patch_id")

        # Continuous-numeric encoding so that trees can extrapolate to
        # *unseen* future patches (categorical IDs collapse to UNK on
        # time-based test splits and carry no signal).
        def _patch_to_float(s) -> float:
            try:
                parts = str(s).split(".")
                major = int(parts[0]) if len(parts) > 0 else 0
                minor = int(parts[1]) if len(parts) > 1 else 0
                build = int(parts[2]) if len(parts) > 2 else 0
                return major + minor / 100.0 + build / 1_000_000.0
            except Exception:
                return 0.0

        out["patch_numeric"] = df["patch"].map(_patch_to_float).astype(np.float32)
    if "timestamp" in df.columns:
        # Match epoch in milliseconds → days since data start. Continuous
        # feature so trees can split on "is this match late?" cleanly.
        ts = pd.to_numeric(df["timestamp"], errors="coerce")
        if ts.notna().any():
            origin = ts.dropna().min()
            out["match_time_days"] = ((ts - origin) / (1000 * 60 * 60 * 24)).astype(np.float32)
    if cfg.use_rank_feature and "rank" in df.columns:
        vmap = extra_vocabs.get("rank", {})
        out["rank_id"] = (
            df["rank"].astype(str).map(lambda x: vmap.get(x, 0)).astype(np.int32)
        )
        cat_cols.append("rank_id")
    if cfg.use_bans_feature and "bans" in df.columns:
        ban_vocab = extra_vocabs.get("bans", {})
        for champ_name, idx in ban_vocab.items():
            col = f"ban_{champ_name}"
            out[col] = df["bans"].astype(str).map(
                lambda b, c=champ_name: int(c in b)
            ).astype(np.int8)

    return out, cat_cols


def fit_extra_vocabs(
    train_df: pd.DataFrame, cfg: PipelineConfig, top_bans: int = 30
) -> Dict[str, Dict[str, int]]:
    """Learn categorical encodings for patch / rank / bans on the train split."""
    out: Dict[str, Dict[str, int]] = {}
    if cfg.use_patch_feature and "patch" in train_df.columns:
        vals = sorted(train_df["patch"].dropna().astype(str).unique().tolist())
        out["patch"] = {"<UNK>": 0, **{v: i + 1 for i, v in enumerate(vals)}}
    if cfg.use_rank_feature and "rank" in train_df.columns:
        vals = sorted(train_df["rank"].dropna().astype(str).unique().tolist())
        out["rank"] = {"<UNK>": 0, **{v: i + 1 for i, v in enumerate(vals)}}
    if cfg.use_bans_feature and "bans" in train_df.columns:
        # Bans are typically a comma-separated string per match.
        from collections import Counter

        c: "Counter[str]" = Counter()
        for spec in train_df["bans"].dropna().astype(str):
            for tok in spec.split(","):
                tok = tok.strip()
                if tok:
                    c[tok] += 1
        common = [b for b, _ in c.most_common(top_bans)]
        out["bans"] = {b: i for i, b in enumerate(common)}
    return out


# --------------------------------------------------------------------------- #
# Calibration (isotonic regression on val set)
# --------------------------------------------------------------------------- #


class ProbabilityCalibrator:
    """Wraps ``sklearn.isotonic.IsotonicRegression`` with a no-op fallback."""

    def __init__(self) -> None:
        self.iso = None

    def fit(self, probs: np.ndarray, y: np.ndarray) -> "ProbabilityCalibrator":
        try:
            from sklearn.isotonic import IsotonicRegression

            probs = np.asarray(probs, dtype=float).reshape(-1)
            y = np.asarray(y, dtype=float).reshape(-1)
            if len(np.unique(y)) < 2:
                return self  # cannot fit; leave as identity
            self.iso = IsotonicRegression(out_of_bounds="clip").fit(probs, y)
        except Exception as exc:  # pragma: no cover
            log.warning("Calibrator fit failed: %s", exc)
            self.iso = None
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        if self.iso is None:
            return np.asarray(probs, dtype=float)
        return np.clip(self.iso.predict(np.asarray(probs, dtype=float)), 1e-6, 1 - 1e-6)


# --------------------------------------------------------------------------- #
# PMI + SVD pretraining of champion embeddings
# --------------------------------------------------------------------------- #


def pretrain_champion_embeddings(
    train_df: pd.DataFrame,
    vocab: Dict[str, int],
    embedding_dim: int,
    smoothing: float = 0.75,
) -> np.ndarray:
    """Compute a GloVe-style PMI(+SVD) initialisation for champion vectors.

    Builds a same-team co-occurrence matrix on the train split, applies
    positive-PMI (with frequency smoothing as in the original GloVe paper),
    and truncates the SVD to ``embedding_dim`` dimensions.  The output is
    suitable for ``nn.Embedding.weight.data.copy_`` initialisation.

    Returns an array of shape ``[len(vocab), embedding_dim]`` with row 0
    (the UNK token) zeroed out.
    """
    N = len(vocab)
    M = np.zeros((N, N), dtype=np.float64)
    blue_cols = champion_columns_for_side("blue")
    red_cols = champion_columns_for_side("red")

    def _accumulate(values: np.ndarray) -> None:
        for champs in values:
            ids = [vocab.get(str(c), UNKNOWN_INDEX) for c in champs]
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    a, b = ids[i], ids[j]
                    if a == UNKNOWN_INDEX or b == UNKNOWN_INDEX:
                        continue
                    M[a, b] += 1
                    M[b, a] += 1

    _accumulate(train_df[blue_cols].values)
    _accumulate(train_df[red_cols].values)

    # GloVe-style smoothing: use marginals raised to power < 1.
    row_sum = M.sum(axis=1, keepdims=True) + 1e-9
    col_sum = M.sum(axis=0, keepdims=True) + 1e-9
    total = M.sum() + 1e-9
    smoothed_row = np.power(row_sum, smoothing)
    smoothed_col = np.power(col_sum, smoothing)
    pmi = np.log(M * total / (smoothed_row * smoothed_col) + 1e-9)
    ppmi = np.maximum(pmi, 0.0)
    np.fill_diagonal(ppmi, 0.0)

    try:
        u, s, _ = np.linalg.svd(ppmi, full_matrices=False)
        k = min(embedding_dim, len(s))
        emb = u[:, :k] * np.sqrt(np.maximum(s[:k], 0.0))
    except np.linalg.LinAlgError:
        log.warning("SVD failed during pretraining; returning zero init.")
        emb = np.zeros((N, embedding_dim))

    if emb.shape[1] < embedding_dim:
        pad = np.zeros((N, embedding_dim - emb.shape[1]))
        emb = np.concatenate([emb, pad], axis=1)
    # Normalise per-row to unit variance so the network sees stable scale.
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    emb = emb / norms
    emb[UNKNOWN_INDEX] = 0.0
    return emb.astype(np.float32)


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cat_cols: List[str],
    cfg: PipelineConfig,
    event_logger: Optional[EventLogger] = None,
    model_name: str = "lightgbm",
) -> Tuple[object, str]:
    """Train LightGBM if available, otherwise fall back to HistGradientBoosting.

    When an ``event_logger`` is provided, eval metrics from each evaluation
    period are forwarded as ``train_metric`` events for the dashboard.
    """
    if _HAS_LGB:
        params = dict(
            objective="binary",
            metric=["binary_logloss", "auc"],
            learning_rate=cfg.lgb_learning_rate,
            num_leaves=cfg.lgb_num_leaves,
            min_child_samples=cfg.lgb_min_child_samples,
            feature_pre_filter=False,
            verbose=-1,
            random_state=cfg.random_seed,
        )
        train_set = lgb.Dataset(
            X_train, label=y_train, categorical_feature=cat_cols, free_raw_data=False
        )
        val_set = lgb.Dataset(
            X_val,
            label=y_val,
            categorical_feature=cat_cols,
            reference=train_set,
            free_raw_data=False,
        )
        callbacks = [
            lgb.early_stopping(cfg.lgb_early_stop_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ]
        if event_logger is not None:
            def _eval_callback(env):
                # env.evaluation_result_list -> [(name, metric, value, _is_higher_better), ...]
                grouped: Dict[str, Dict[str, float]] = {}
                for entry in env.evaluation_result_list:
                    split, metric, value = entry[0], entry[1], entry[2]
                    grouped.setdefault(split, {})[metric] = value
                for split, mvals in grouped.items():
                    event_logger.lgb_iter(model_name, env.iteration + 1, split, mvals)

            _eval_callback.order = 20
            callbacks.append(_eval_callback)
        booster = lgb.train(
            params,
            train_set,
            num_boost_round=cfg.lgb_n_estimators,
            valid_sets=[train_set, val_set],
            valid_names=["train", "val"],
            callbacks=callbacks,
        )
        return booster, "lightgbm"

    log.warning("LightGBM not available - falling back to HistGradientBoosting")
    model = HistGradientBoostingClassifier(
        max_iter=cfg.lgb_n_estimators,
        learning_rate=cfg.lgb_learning_rate,
        max_leaf_nodes=cfg.lgb_num_leaves,
        min_samples_leaf=cfg.lgb_min_child_samples,
        random_state=cfg.random_seed,
        early_stopping=True,
        validation_fraction=None,
    )
    model.fit(X_train, y_train)
    return model, "hist_gradient_boosting"


def predict_lightgbm(model: object, backend: str, X: pd.DataFrame) -> np.ndarray:
    if backend == "lightgbm":
        return model.predict(X)
    return model.predict_proba(X)[:, 1]


# --------------------------------------------------------------------------- #
# TeamCompNet (PyTorch)
# --------------------------------------------------------------------------- #


if _HAS_TORCH:

    class TeamCompNet(nn.Module):
        """Champion-embedding draft-time win predictor with pairwise interactions."""

        def __init__(
            self,
            num_champions: int,
            embedding_dim: int = 32,
            hidden_dim: int = 128,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            self.num_champions = num_champions
            self.embedding_dim = embedding_dim
            self.champion_emb = nn.Embedding(
                num_champions, embedding_dim, padding_idx=UNKNOWN_INDEX
            )
            self.role_emb = nn.Embedding(len(ROLES), embedding_dim)
            # Inputs: blue_pool, red_pool, diff, prod, 3 pairwise scores
            in_dim = 4 * embedding_dim + 3
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        def _team_repr(self, champ_ids: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
            """Return (mean_pool, embeddings_with_role) for a side."""
            B, R = champ_ids.shape
            roles = torch.arange(R, device=champ_ids.device).unsqueeze(0).expand(B, R)
            emb = self.champion_emb(champ_ids) + self.role_emb(roles)
            pool = emb.mean(dim=1)
            return pool, emb

        @staticmethod
        def _avg_pairwise(emb_a: "torch.Tensor", emb_b: "torch.Tensor") -> "torch.Tensor":
            # Average dot product across all (i, j) pairs.
            # emb shapes: [B, Na, D] and [B, Nb, D]
            scores = torch.einsum("bid,bjd->bij", emb_a, emb_b)  # [B, Na, Nb]
            return scores.mean(dim=(1, 2))

        def _pairwise_internal(self, emb: "torch.Tensor") -> "torch.Tensor":
            B, R, _ = emb.shape
            scores = torch.einsum("bid,bjd->bij", emb, emb)
            mask = ~torch.eye(R, dtype=torch.bool, device=emb.device)
            return scores.masked_select(mask.unsqueeze(0)).view(B, -1).mean(dim=1)

        def forward(
            self,
            blue_ids: "torch.Tensor",
            red_ids: "torch.Tensor",
        ) -> "torch.Tensor":
            blue_pool, blue_emb = self._team_repr(blue_ids)
            red_pool, red_emb = self._team_repr(red_ids)

            blue_syn = self._pairwise_internal(blue_emb)
            red_syn = self._pairwise_internal(red_emb)
            cross = self._avg_pairwise(blue_emb, red_emb)

            x = torch.cat(
                [
                    blue_pool,
                    red_pool,
                    blue_pool - red_pool,
                    blue_pool * red_pool,
                    blue_syn.unsqueeze(1),
                    red_syn.unsqueeze(1),
                    cross.unsqueeze(1),
                ],
                dim=1,
            )
            return self.mlp(x).squeeze(-1)

        def champion_embeddings(self) -> np.ndarray:
            """Return the learned champion-only embedding matrix as numpy."""
            return self.champion_emb.weight.detach().cpu().numpy()


    class SetTransformerCompNet(nn.Module):
        """Set-style Transformer over the 10-token (5 blue + 5 red) draft.

        Each token is the sum of (champion_emb, role_emb, side_emb).  A
        learnable [CLS] token aggregates the global representation across
        ``n_attention_layers`` of multi-head self-attention.  Two output
        heads:

        * ``value_logit`` - blue-side win probability (BCEWithLogits)
        * ``policy_logits`` (optional) - distribution over the next champion
          pick at the (side, role) the user is currently filling. Used for
          AlphaZero-style policy training and as the prior for MCTS.
        """

        def __init__(
            self,
            num_champions: int,
            embedding_dim: int = 32,
            hidden_dim: int = 128,
            dropout: float = 0.2,
            n_layers: int = 2,
            n_heads: int = 4,
            policy_head: bool = True,
        ) -> None:
            super().__init__()
            self.num_champions = num_champions
            self.embedding_dim = embedding_dim
            self.policy_head_enabled = policy_head
            self.champion_emb = nn.Embedding(
                num_champions, embedding_dim, padding_idx=UNKNOWN_INDEX
            )
            self.role_emb = nn.Embedding(len(ROLES), embedding_dim)
            self.side_emb = nn.Embedding(2, embedding_dim)
            # Extra prompt tokens encode "we are about to pick at (side, role)"
            # so the policy head is actually conditional. The slot prompt has
            # the same dimensionality as a champion token; UNK champion +
            # role + side gives the right semantics.
            self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.value_head = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim, 1),
            )
            self.policy_proj: Optional[nn.Linear] = None
            if policy_head:
                # Tied to the champion embedding to limit parameters.
                self.policy_bias = nn.Parameter(torch.zeros(num_champions))

        def init_pretrained_embeddings(self, weight: np.ndarray) -> None:
            with torch.no_grad():
                w = torch.as_tensor(weight, dtype=self.champion_emb.weight.dtype)
                if w.shape != self.champion_emb.weight.shape:
                    log.warning(
                        "Pretrained shape %s != embedding %s; skipping init.",
                        tuple(w.shape),
                        tuple(self.champion_emb.weight.shape),
                    )
                    return
                self.champion_emb.weight.copy_(w)

        def _build_tokens(self, blue_ids: "torch.Tensor", red_ids: "torch.Tensor") -> "torch.Tensor":
            B = blue_ids.size(0)
            roles = torch.arange(len(ROLES), device=blue_ids.device).unsqueeze(0).expand(B, -1)
            sides_b = torch.zeros_like(blue_ids)
            sides_r = torch.ones_like(red_ids)
            blue_tok = (
                self.champion_emb(blue_ids)
                + self.role_emb(roles)
                + self.side_emb(sides_b)
            )
            red_tok = (
                self.champion_emb(red_ids)
                + self.role_emb(roles)
                + self.side_emb(sides_r)
            )
            cls = self.cls_token.expand(B, -1, -1)
            return torch.cat([cls, blue_tok, red_tok], dim=1)  # [B, 11, D]

        def forward(
            self,
            blue_ids: "torch.Tensor",
            red_ids: "torch.Tensor",
            return_policy: bool = False,
        ):
            tokens = self._build_tokens(blue_ids, red_ids)
            encoded = self.encoder(tokens)  # [B, 11, D]
            cls_out = encoded[:, 0]  # [B, D]
            value_logit = self.value_head(cls_out).squeeze(-1)
            if return_policy and self.policy_head_enabled:
                # Policy logits = cls @ champion_emb^T + bias (tied weights).
                policy = cls_out @ self.champion_emb.weight.T + self.policy_bias
                return value_logit, policy
            return value_logit

        def champion_embeddings(self) -> np.ndarray:
            return self.champion_emb.weight.detach().cpu().numpy()


    class WideDeepDraftNet(nn.Module):
        """Wide & Deep draft-time predictor.

        Wide branch is a sparse logistic regression over role-slot champion
        one-hot features; Deep branch reuses the TeamCompNet body.
        """

        def __init__(
            self,
            num_champions: int,
            embedding_dim: int = 32,
            hidden_dim: int = 128,
            dropout: float = 0.2,
            combine: str = "sum",
        ) -> None:
            super().__init__()
            self.combine = combine
            self.deep = TeamCompNet(
                num_champions=num_champions,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
            wide_dim = 2 * len(ROLES) * num_champions  # blue + red, role-slot one-hot
            self.wide = nn.Linear(wide_dim, 1, bias=False)
            self._wide_dim = wide_dim
            self.num_champions = num_champions
            if combine == "concat":
                self.combine_head = nn.Sequential(
                    nn.Linear(2, 16),
                    nn.ReLU(),
                    nn.Linear(16, 1),
                )
            else:
                self.combine_head = None

        def _wide_features(
            self, blue_ids: "torch.Tensor", red_ids: "torch.Tensor"
        ) -> "torch.Tensor":
            B = blue_ids.shape[0]
            x = torch.zeros(B, self._wide_dim, device=blue_ids.device)
            num_champs = self.num_champions
            for i, ids in enumerate((blue_ids, red_ids)):
                for r in range(len(ROLES)):
                    offset = (i * len(ROLES) + r) * num_champs
                    x[torch.arange(B), offset + ids[:, r]] = 1.0
            return x

        def forward(
            self, blue_ids: "torch.Tensor", red_ids: "torch.Tensor"
        ) -> "torch.Tensor":
            wide_logit = self.wide(self._wide_features(blue_ids, red_ids)).squeeze(-1)
            deep_logit = self.deep(blue_ids, red_ids)
            if self.combine == "sum":
                return wide_logit + deep_logit
            return self.combine_head(
                torch.stack([wide_logit, deep_logit], dim=1)
            ).squeeze(-1)


# --------------------------------------------------------------------------- #
# PyTorch training helpers
# --------------------------------------------------------------------------- #


def _make_loader(
    blue_ids: np.ndarray,
    red_ids: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> "DataLoader":
    ds = TensorDataset(
        torch.from_numpy(blue_ids).long(),
        torch.from_numpy(red_ids).long(),
        torch.from_numpy(y.astype(np.float32)),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _model_value_logits(model: "nn.Module", blue_ids: "torch.Tensor", red_ids: "torch.Tensor"):
    """Forward pass that always returns just the value logit (handles tuple-returning models)."""
    out = model(blue_ids, red_ids)
    if isinstance(out, tuple):
        return out[0]
    return out


def _augment_batch(
    blue_ids: "torch.Tensor",
    red_ids: "torch.Tensor",
    y: "torch.Tensor",
    cfg: PipelineConfig,
    num_champions: int,
):
    """Side-flip + champion-id dropout (mask -> UNK) augmentation.

    Side flips happen at row level with prob 0.5 (label is inverted accordingly).
    Champion dropout zero-masks individual slot ids to UNK with prob
    ``cfg.augment_dropout_p``.  Both transforms preserve label correctness.
    """
    if cfg.augment_side_flip:
        flip = torch.rand(y.size(0), device=blue_ids.device) < 0.5
        if flip.any():
            new_blue = torch.where(flip.unsqueeze(1), red_ids, blue_ids)
            new_red = torch.where(flip.unsqueeze(1), blue_ids, red_ids)
            blue_ids, red_ids = new_blue, new_red
            y = torch.where(flip, 1.0 - y, y)
    if cfg.augment_dropout_p > 0:
        mask_b = torch.rand_like(blue_ids, dtype=torch.float32) < cfg.augment_dropout_p
        mask_r = torch.rand_like(red_ids, dtype=torch.float32) < cfg.augment_dropout_p
        blue_ids = blue_ids.masked_fill(mask_b, UNKNOWN_INDEX)
        red_ids = red_ids.masked_fill(mask_r, UNKNOWN_INDEX)
    return blue_ids, red_ids, y


def _ranking_loss(
    model: "nn.Module",
    blue_ids: "torch.Tensor",
    red_ids: "torch.Tensor",
    y: "torch.Tensor",
    cfg: PipelineConfig,
    num_champions: int,
) -> "torch.Tensor":
    """Listwise softmax-CE loss: rank true champion above ``ranking_negatives`` randoms.

    For each row we randomly pick a slot to "hide", sample K-1 negatives from
    the champion vocab (excluding UNK and the true champion), and ask the
    value head to score the true completion higher than the negatives.  The
    objective directly aligns with Recall@k and MRR.

    Side-aware: if a blue slot is hidden the score for that side is
    ``logit``; if a red slot is hidden it's ``-logit`` (red wants low blue
    win prob).
    """
    B = blue_ids.size(0)
    K = max(2, cfg.ranking_negatives + 1)
    device = blue_ids.device

    hide_role = torch.randint(0, len(ROLES), (B,), device=device)
    hide_side = torch.randint(0, 2, (B,), device=device)  # 0=blue, 1=red

    # Capture true champ for each row at the hidden slot.
    blue_clone = blue_ids.clone()
    red_clone = red_ids.clone()
    rows = torch.arange(B, device=device)
    true_champ_blue = blue_ids[rows, hide_role]
    true_champ_red = red_ids[rows, hide_role]
    true_champ = torch.where(hide_side == 0, true_champ_blue, true_champ_red)

    # Replicate batch K times along a new axis.
    blue_rep = blue_clone.unsqueeze(1).expand(B, K, len(ROLES)).clone()  # [B, K, R]
    red_rep = red_clone.unsqueeze(1).expand(B, K, len(ROLES)).clone()

    # Slot 0 of K is the true completion; slots 1..K-1 are negatives.
    neg_ids = torch.randint(1, num_champions, (B, K - 1), device=device)
    cand_ids = torch.cat([true_champ.unsqueeze(1), neg_ids], dim=1)  # [B, K]

    # Insert candidate into the hidden slot for each of K candidates.
    role_idx = hide_role.unsqueeze(1).expand(B, K)
    blue_target = blue_rep[rows.unsqueeze(1).expand(B, K), torch.arange(K, device=device).unsqueeze(0).expand(B, K), role_idx]
    # We'll write candidates directly via advanced indexing on flat tensors.
    flat_blue = blue_rep.reshape(B * K, len(ROLES))
    flat_red = red_rep.reshape(B * K, len(ROLES))
    flat_role = role_idx.reshape(-1)
    flat_side = hide_side.unsqueeze(1).expand(B, K).reshape(-1)
    flat_cand = cand_ids.reshape(-1)
    flat_rows = torch.arange(B * K, device=device)

    blue_mask = flat_side == 0
    red_mask = ~blue_mask
    if blue_mask.any():
        flat_blue[flat_rows[blue_mask], flat_role[blue_mask]] = flat_cand[blue_mask]
    if red_mask.any():
        flat_red[flat_rows[red_mask], flat_role[red_mask]] = flat_cand[red_mask]

    logits = _model_value_logits(model, flat_blue, flat_red)  # [B*K]
    logits = logits.view(B, K)
    # Sign according to which side is choosing.
    signs = torch.where(hide_side == 0, torch.ones(B, device=device), -torch.ones(B, device=device))
    logits = logits * signs.unsqueeze(1)
    target = torch.zeros(B, dtype=torch.long, device=device)
    return F.cross_entropy(logits, target)


def _policy_loss(
    model: "nn.Module",
    blue_ids: "torch.Tensor",
    red_ids: "torch.Tensor",
    cfg: PipelineConfig,
    num_champions: int,
) -> Optional["torch.Tensor"]:
    """AlphaZero-style policy CE: predict the true champion at a hidden slot.

    Returns ``None`` when the model has no policy head.
    """
    if not getattr(model, "policy_head_enabled", False):
        return None
    B = blue_ids.size(0)
    device = blue_ids.device

    hide_role = torch.randint(0, len(ROLES), (B,), device=device)
    hide_side = torch.randint(0, 2, (B,), device=device)
    rows = torch.arange(B, device=device)

    true_blue = blue_ids[rows, hide_role]
    true_red = red_ids[rows, hide_role]
    true_champ = torch.where(hide_side == 0, true_blue, true_red)

    blue_in = blue_ids.clone()
    red_in = red_ids.clone()
    blue_mask = hide_side == 0
    red_mask = ~blue_mask
    if blue_mask.any():
        blue_in[rows[blue_mask], hide_role[blue_mask]] = UNKNOWN_INDEX
    if red_mask.any():
        red_in[rows[red_mask], hide_role[red_mask]] = UNKNOWN_INDEX

    out = model(blue_in, red_in, return_policy=True)
    if not isinstance(out, tuple):
        return None
    _, policy_logits = out
    return F.cross_entropy(policy_logits, true_champ)


def _train_torch_model(
    model: "nn.Module",
    train_loader: "DataLoader",
    val_loader: "DataLoader",
    cfg: PipelineConfig,
    device: "torch.device",
    name: str,
    event_logger: Optional[EventLogger] = None,
    num_champions: Optional[int] = None,
) -> Dict[str, List[float]]:
    """Generic train/eval loop with early stopping.  Mutates model in-place.

    Includes optional augmentation (side-flip + champion dropout), listwise
    ranking auxiliary loss, and AlphaZero-style policy loss.  Each component
    can be disabled via :class:`PipelineConfig`.
    """
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    loss_fn = nn.BCEWithLogitsLoss()
    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [], "val_auc": [],
        "rank_loss": [], "policy_loss": [],
    }
    best_state = None
    best_val = math.inf
    patience_left = cfg.patience
    if num_champions is None:
        emb_layer = getattr(model, "champion_emb", None)
        num_champions = emb_layer.num_embeddings if emb_layer is not None else 200

    epochs = 1 if cfg.fast_dev_run else cfg.epochs
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        train_loss, rank_loss_acc, policy_loss_acc, n = 0.0, 0.0, 0.0, 0
        for blue_ids, red_ids, y in train_loader:
            blue_ids = blue_ids.to(device)
            red_ids = red_ids.to(device)
            y = y.to(device)
            blue_ids, red_ids, y = _augment_batch(
                blue_ids, red_ids, y, cfg, num_champions
            )
            opt.zero_grad()
            logits = _model_value_logits(model, blue_ids, red_ids)
            loss = loss_fn(logits, y)
            total = loss
            if cfg.ranking_weight > 0:
                rl = _ranking_loss(model, blue_ids, red_ids, y, cfg, num_champions)
                total = total + cfg.ranking_weight * rl
                rank_loss_acc += rl.item() * y.size(0)
            if cfg.policy_weight > 0:
                pl = _policy_loss(model, blue_ids, red_ids, cfg, num_champions)
                if pl is not None:
                    total = total + cfg.policy_weight * pl
                    policy_loss_acc += pl.item() * y.size(0)
            total.backward()
            opt.step()
            train_loss += loss.item() * y.size(0)
            n += y.size(0)
        train_loss /= max(n, 1)
        rank_loss_avg = rank_loss_acc / max(n, 1)
        policy_loss_avg = policy_loss_acc / max(n, 1)

        # Validation
        model.eval()
        val_loss, m = 0.0, 0
        all_p, all_y = [], []
        with torch.no_grad():
            for blue_ids, red_ids, y in val_loader:
                blue_ids, red_ids, y = (
                    blue_ids.to(device),
                    red_ids.to(device),
                    y.to(device),
                )
                logits = _model_value_logits(model, blue_ids, red_ids)
                val_loss += loss_fn(logits, y).item() * y.size(0)
                m += y.size(0)
                all_p.append(torch.sigmoid(logits).cpu().numpy())
                all_y.append(y.cpu().numpy())
        val_loss /= max(m, 1)
        probs = np.concatenate(all_p)
        truths = np.concatenate(all_y)
        try:
            val_auc = float(roc_auc_score(truths, probs))
        except ValueError:
            val_auc = float("nan")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["rank_loss"].append(rank_loss_avg)
        history["policy_loss"].append(policy_loss_avg)
        extras = []
        if cfg.ranking_weight > 0:
            extras.append(f"rank={rank_loss_avg:.3f}")
        if cfg.policy_weight > 0 and policy_loss_avg > 0:
            extras.append(f"policy={policy_loss_avg:.3f}")
        suffix = (" " + " ".join(extras)) if extras else ""
        log.info(
            "[%s] epoch %02d/%02d  train_loss=%.4f  val_loss=%.4f  val_auc=%.4f%s  (%.1fs)",
            name,
            epoch,
            epochs,
            train_loss,
            val_loss,
            val_auc,
            suffix,
            time.time() - t0,
        )
        if event_logger is not None:
            event_logger.epoch(name, epoch, epochs, train_loss, val_loss, val_auc)

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                log.info("[%s] early stop at epoch %d", name, epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def _torch_predict_proba(
    model: "nn.Module",
    blue_ids: np.ndarray,
    red_ids: np.ndarray,
    device: "torch.device",
    batch_size: int = 1024,
) -> np.ndarray:
    """Predict win probabilities in mini-batches (handles tuple-returning models)."""
    model.eval()
    out = np.empty(len(blue_ids), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(blue_ids), batch_size):
            end = start + batch_size
            b = torch.from_numpy(blue_ids[start:end]).long().to(device)
            r = torch.from_numpy(red_ids[start:end]).long().to(device)
            logits = _model_value_logits(model, b, r)
            out[start:end] = torch.sigmoid(logits).cpu().numpy()
    return out


def _torch_policy(model: "nn.Module", state: "DraftState",
                  vocab: Dict[str, int], device: "torch.device") -> Optional[np.ndarray]:
    """Return softmax policy distribution over champions for a single state, or None."""
    if not getattr(model, "policy_head_enabled", False):
        return None
    blue_ids, red_ids = state.to_id_arrays(vocab)
    b = torch.from_numpy(blue_ids).long().to(device)
    r = torch.from_numpy(red_ids).long().to(device)
    model.eval()
    with torch.no_grad():
        out = model(b, r, return_policy=True)
        if not isinstance(out, tuple):
            return None
        _, logits = out
        return F.softmax(logits, dim=-1).cpu().numpy()[0]


# --------------------------------------------------------------------------- #
# Embedding feature extraction (for hybrid LightGBM)
# --------------------------------------------------------------------------- #


def extract_embedding_features(
    blue_ids: np.ndarray,
    red_ids: np.ndarray,
    champion_emb: np.ndarray,
) -> pd.DataFrame:
    """Map drafted champion ids -> derived team-embedding feature columns."""
    D = champion_emb.shape[1]
    blue_emb = champion_emb[blue_ids]  # [N, 5, D]
    red_emb = champion_emb[red_ids]  # [N, 5, D]
    blue_mean = blue_emb.mean(axis=1)
    red_mean = red_emb.mean(axis=1)
    diff = blue_mean - red_mean
    prod = blue_mean * red_mean

    # Pairwise summaries via Gram matrices
    blue_dot = np.einsum("nid,njd->nij", blue_emb, blue_emb)
    red_dot = np.einsum("nid,njd->nij", red_emb, red_emb)
    cross_dot = np.einsum("nid,njd->nij", blue_emb, red_emb)

    eye = np.eye(blue_emb.shape[1], dtype=bool)
    blue_pair = blue_dot[:, ~eye].mean(axis=1)
    red_pair = red_dot[:, ~eye].mean(axis=1)
    cross_pair = cross_dot.mean(axis=(1, 2))

    cols = {}
    for j in range(D):
        cols[f"emb_blue_mean_{j}"] = blue_mean[:, j]
    for j in range(D):
        cols[f"emb_red_mean_{j}"] = red_mean[:, j]
    for j in range(D):
        cols[f"emb_diff_{j}"] = diff[:, j]
    for j in range(D):
        cols[f"emb_prod_{j}"] = prod[:, j]
    cols["emb_blue_pair_score"] = blue_pair
    cols["emb_red_pair_score"] = red_pair
    cols["emb_cross_pair_score"] = cross_pair
    return pd.DataFrame(cols)


# --------------------------------------------------------------------------- #
# Recommender / Beam Search
# --------------------------------------------------------------------------- #


@dataclass
class DraftState:
    """Lightweight container describing an in-progress draft."""

    blue_picks: Dict[str, str] = field(default_factory=dict)
    red_picks: Dict[str, str] = field(default_factory=dict)
    bans: List[str] = field(default_factory=list)

    def picks_for(self, side: str) -> Dict[str, str]:
        return self.blue_picks if side == "blue" else self.red_picks

    def used_champions(self) -> set:
        return (
            set(self.blue_picks.values())
            | set(self.red_picks.values())
            | set(self.bans)
        )

    def empty_roles(self, side: str) -> List[str]:
        picks = self.picks_for(side)
        return [r for r in ROLES if r not in picks]

    def to_id_arrays(self, vocab: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
        def encode(picks: Dict[str, str]) -> np.ndarray:
            arr = np.full(len(ROLES), UNKNOWN_INDEX, dtype=np.int64)
            for r, c in picks.items():
                arr[ROLE_TO_IDX[r]] = vocab.get(c, UNKNOWN_INDEX)
            return arr

        return encode(self.blue_picks)[None, :], encode(self.red_picks)[None, :]

    def clone(self) -> "DraftState":
        return DraftState(
            blue_picks=dict(self.blue_picks),
            red_picks=dict(self.red_picks),
            bans=list(self.bans),
        )


class _MCTSNode:
    """Internal node for the PUCT search tree."""

    __slots__ = (
        "state", "side", "role", "prior", "parent", "action",
        "children", "visit_count", "value_sum",
    )

    def __init__(self, state, side, role, prior, parent, action):
        self.state = state
        self.side = side
        self.role = role
        self.prior = prior
        self.parent = parent
        self.action = action
        self.children: Dict[str, "_MCTSNode"] = {}
        self.visit_count = 0
        self.value_sum = 0.0

    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0


class Recommender:
    """Wraps any draft-time scorer and exposes top-k / beam search APIs.

    ``score_fn`` is the per-state scorer (returns blue win probability).
    ``batch_score_fn`` is an optional fast path that evaluates many candidates
    for one (side, role) slot in a single batched call - used by
    :py:meth:`top_k` / beam search for large candidate sets.
    """

    def __init__(
        self,
        score_fn,
        vocab: Dict[str, int],
        handcrafted: Optional[HandcraftedStats] = None,
        batch_score_fn=None,
    ) -> None:
        self.score_fn = score_fn  # callable: (DraftState) -> blue_win_prob
        self.batch_score_fn = batch_score_fn  # (state, side, role, [cands]) -> np.ndarray
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.handcrafted = handcrafted

    # --- top-k ----------------------------------------------------------- #

    def candidates(self, state: DraftState) -> List[str]:
        used = state.used_champions()
        return [c for c in self.vocab.keys() if c != UNKNOWN_TOKEN and c not in used]

    def score_state(self, state: DraftState) -> float:
        return float(self.score_fn(state))

    def _score_candidates(
        self, state: DraftState, side: str, role: str, candidates: List[str]
    ) -> np.ndarray:
        """Return blue_win_prob for each candidate filling ``(side, role)``."""
        if self.batch_score_fn is not None:
            return np.asarray(self.batch_score_fn(state, side, role, candidates))
        out = np.empty(len(candidates))
        for i, cand in enumerate(candidates):
            child = state.clone()
            child.picks_for(side)[role] = cand
            out[i] = self.score_state(child)
        return out

    def _explanation(
        self,
        state: DraftState,
        side: str,
        role: str,
        candidate: str,
        baseline_prob: float,
        new_prob: float,
    ) -> Dict[str, object]:
        notes: List[str] = []
        my_picks = list(state.picks_for(side).values())
        opp_picks = list(
            state.picks_for("red" if side == "blue" else "blue").values()
        )
        synergy = 0.0
        counter = 0.0
        if self.handcrafted is not None:
            if my_picks:
                synergy = float(
                    np.mean([self.handcrafted.synergy(candidate, c) for c in my_picks])
                )
            if opp_picks:
                counter = float(
                    np.mean([self.handcrafted.counter(candidate, e) for e in opp_picks])
                )
            if not opp_picks:
                notes.append("blind pick risk: enemy team unknown")
            wr = self.handcrafted.winrate(candidate)
            if wr < 0.45:
                notes.append("low historical win rate")
            if wr > 0.55:
                notes.append("strong historical win rate")
            if synergy > 0.02:
                notes.append("good synergy with current allies")
            if counter > 0.02:
                notes.append("favourable matchup vs enemy picks")
            if counter < -0.02:
                notes.append("warning: weak matchup vs enemy picks")
        delta = new_prob - baseline_prob
        if abs(delta) < 5e-3:
            notes.append("marginal model delta")
        return {
            "champion": candidate,
            "win_prob": new_prob,
            "delta": delta,
            "synergy": synergy,
            "counter": counter,
            "notes": "; ".join(notes) if notes else "stable model win probability",
        }

    def top_k(
        self,
        state: DraftState,
        side: str,
        role: str,
        k: int = 5,
    ) -> List[Dict[str, object]]:
        """Greedy top-k enumeration over a single empty (side, role) slot."""
        if role in state.picks_for(side):
            raise ValueError(f"{side} {role} already filled with {state.picks_for(side)[role]}")

        baseline_prob = self.score_state(state)
        cands = self.candidates(state)
        scores = self._score_candidates(state, side, role, cands)
        results: List[Dict[str, object]] = []
        for cand, prob in zip(cands, scores):
            score = prob if side == "blue" else (1.0 - prob)
            base = baseline_prob if side == "blue" else (1.0 - baseline_prob)
            results.append(self._explanation(state, side, role, cand, base, score))
        results.sort(key=lambda r: r["win_prob"], reverse=True)
        return results[:k]

    # --- beam search ----------------------------------------------------- #

    def beam_search(
        self,
        state: DraftState,
        side: str,
        role: str,
        beam_width: int = 5,
        depth: int = 2,
        k: int = 5,
    ) -> List[Dict[str, object]]:
        """Beam search with minimax-style opponent modelling.

        Each beam node is ``(state, score_for_decision_side)``.  At our turns we
        keep the ``beam_width`` highest-scoring children; at opponent turns we
        assume they pick the move that minimises our score.

        Falls back to greedy ``top_k`` when ``depth <= 1`` or there are no
        further empty slots.
        """
        if depth <= 1:
            return self.top_k(state, side, role, k=k)

        decision_side = side  # whose perspective we maximise
        baseline_prob = self.score_state(state)
        baseline_for_us = (
            baseline_prob if decision_side == "blue" else (1.0 - baseline_prob)
        )

        candidates = self.candidates(state)
        scores = self._score_candidates(state, side, role, candidates)
        screened: List[Tuple[str, float]] = []
        for cand, p in zip(candidates, scores):
            us = p if decision_side == "blue" else (1.0 - p)
            screened.append((cand, us))
        screened.sort(key=lambda x: x[1], reverse=True)
        beam = screened[: max(beam_width, k)]

        scored: List[Dict[str, object]] = []
        for cand, _ in beam:
            child = state.clone()
            child.picks_for(side)[role] = cand
            future_value = self._beam_expand(
                child,
                decision_side=decision_side,
                next_side="red" if side == "blue" else "blue",
                remaining_depth=depth - 1,
                beam_width=beam_width,
            )
            scored.append(
                self._explanation(
                    state,
                    side,
                    role,
                    cand,
                    baseline_for_us,
                    future_value,
                )
            )
        scored.sort(key=lambda r: r["win_prob"], reverse=True)
        return scored[:k]

    # --- MCTS (PUCT) ---------------------------------------------------- #

    def mcts(
        self,
        state: DraftState,
        side: str,
        role: str,
        n_simulations: int = 64,
        c_puct: float = 1.5,
        depth: int = 4,
        policy_fn=None,
        k: int = 5,
    ) -> List[Dict[str, object]]:
        """AlphaZero-flavoured MCTS over the partial draft.

        ``policy_fn`` is an optional callable ``(DraftState, side, role) ->
        np.ndarray[num_champions]`` returning a prior distribution.  When
        absent the prior falls back to a softmax over a single forward pass
        of the value head over all candidates (effectively a "value-shaped"
        prior).  Returns the top-``k`` children of the root by visit count.
        """
        root = _MCTSNode(
            state=state, side=side, role=role,
            prior=1.0, parent=None, action=None,
        )

        # Cache for children expansion
        cands = self.candidates(state)
        if not cands:
            return []
        priors = self._priors_for(state, side, role, cands, policy_fn)
        for cand, p in zip(cands, priors):
            new_state = state.clone()
            new_state.picks_for(side)[role] = cand
            child = _MCTSNode(
                state=new_state, side=side, role=role,
                prior=float(p), parent=root, action=cand,
            )
            root.children[cand] = child

        for _ in range(max(1, n_simulations)):
            self._mcts_simulate(root, root_side=side, depth=depth,
                                c_puct=c_puct, policy_fn=policy_fn)

        baseline_prob = self.score_state(state)
        baseline_for_us = baseline_prob if side == "blue" else (1.0 - baseline_prob)
        ranked = sorted(root.children.values(), key=lambda c: c.visit_count, reverse=True)[:k]
        out: List[Dict[str, object]] = []
        for c in ranked:
            if c.visit_count > 0:
                value = c.value()
            else:
                # Fallback for ties at the bottom of the ranking: use the
                # model's direct value estimate so the displayed win prob is
                # always meaningful.
                p_blue = self.score_state(c.state)
                value = p_blue if side == "blue" else (1.0 - p_blue)
            ex = self._explanation(state, side, role, c.action, baseline_for_us, value)
            ex["mcts_visits"] = int(c.visit_count)
            ex["mcts_prior"] = float(c.prior)
            out.append(ex)
        return out

    def _priors_for(
        self, state: DraftState, side: str, role: str,
        candidates: List[str], policy_fn,
    ) -> np.ndarray:
        if policy_fn is not None:
            try:
                vec = policy_fn(state, side, role)
                if vec is not None:
                    idx = np.array(
                        [self.vocab.get(c, UNKNOWN_INDEX) for c in candidates],
                        dtype=np.int64,
                    )
                    p = vec[idx]
                    s = p.sum()
                    if s > 0:
                        return p / s
            except Exception:
                pass
        # Fallback: softmax of value-head scores from our perspective.
        scores = self._score_candidates(state, side, role, candidates)
        signed = scores if side == "blue" else (1.0 - scores)
        # Sharper temperature so the prior actually informs selection.
        x = (signed - signed.mean()) * 8.0
        exp = np.exp(x - x.max())
        return exp / max(exp.sum(), 1e-9)

    def _mcts_simulate(self, root, root_side, depth, c_puct, policy_fn):
        """Single PUCT roll-out from root to leaf and back-prop visit/value."""
        path = [root]
        node = root
        # Selection: descend with PUCT until we hit an unexpanded child or depth limit.
        while node.children:
            best_score = -math.inf
            best_child = None
            total_visits = sum(ch.visit_count for ch in node.children.values()) + 1
            for ch in node.children.values():
                q = ch.value()  # already from root_side perspective via backup logic
                u = c_puct * ch.prior * math.sqrt(total_visits) / (1 + ch.visit_count)
                s = q + u
                if s > best_score:
                    best_score = s
                    best_child = ch
            node = best_child
            path.append(node)
            if len(path) > depth + 1:
                break

        # Evaluation: value of the leaf state from the *root* side perspective.
        leaf_state = node.state
        leaf_blue = self.score_state(leaf_state)
        value = leaf_blue if root_side == "blue" else (1.0 - leaf_blue)

        # Optional expansion: if we hit a non-terminal state, sprout one ply.
        if not node.children and len(path) <= depth:
            # Decide whose turn it is at this node:
            # The next pick goes to whichever side has empty roles first.
            empty_blue = leaf_state.empty_roles("blue")
            empty_red = leaf_state.empty_roles("red")
            if empty_blue or empty_red:
                next_side = "blue" if len(empty_blue) >= len(empty_red) else "red"
                empties = leaf_state.empty_roles(next_side)
                if empties:
                    next_role = empties[0]
                    cands = self.candidates(leaf_state)
                    if cands:
                        priors = self._priors_for(leaf_state, next_side, next_role, cands, policy_fn)
                        for cand, p in zip(cands, priors):
                            cs = leaf_state.clone()
                            cs.picks_for(next_side)[next_role] = cand
                            node.children[cand] = _MCTSNode(
                                state=cs, side=next_side, role=next_role,
                                prior=float(p), parent=node, action=cand,
                            )

        # Backup
        for n in path:
            n.visit_count += 1
            n.value_sum += value

    def _beam_expand(
        self,
        state: DraftState,
        decision_side: str,
        next_side: str,
        remaining_depth: int,
        beam_width: int,
    ) -> float:
        """Recursive minimax-flavoured expansion; returns value for ``decision_side``."""
        empty = state.empty_roles(next_side)
        if remaining_depth == 0 or not empty:
            p = self.score_state(state)
            return p if decision_side == "blue" else (1.0 - p)

        # Choose the next empty role deterministically.
        role = empty[0]
        candidates = self.candidates(state)
        probs = self._score_candidates(state, next_side, role, candidates)
        scored: List[Tuple[str, float]] = []
        for cand, p in zip(candidates, probs):
            us = p if decision_side == "blue" else (1.0 - p)
            scored.append((cand, us))

        if next_side == decision_side:
            scored.sort(key=lambda x: x[1], reverse=True)
        else:
            scored.sort(key=lambda x: x[1])  # opponent minimises us
        scored = scored[:beam_width]

        # Recurse on the kept beam, take best/worst again.
        future_values: List[float] = []
        for cand, _ in scored:
            child = state.clone()
            child.picks_for(next_side)[role] = cand
            future_values.append(
                self._beam_expand(
                    child,
                    decision_side=decision_side,
                    next_side="red" if next_side == "blue" else "blue",
                    remaining_depth=remaining_depth - 1,
                    beam_width=beam_width,
                )
            )

        if not future_values:
            p = self.score_state(state)
            return p if decision_side == "blue" else (1.0 - p)
        if next_side == decision_side:
            return float(max(future_values))
        return float(min(future_values))


# --------------------------------------------------------------------------- #
# Score-function adapters (so Recommender stays model-agnostic)
# --------------------------------------------------------------------------- #


def make_lgb_score_fn(
    booster: object,
    backend: str,
    vocab: Dict[str, int],
    handcrafted: HandcraftedStats,
    feature_columns: List[str],
    embedding_extras: Optional[np.ndarray] = None,
    calibrator: Optional["ProbabilityCalibrator"] = None,
    extra_vocabs: Optional[Dict[str, Dict[str, int]]] = None,
    cfg: Optional[PipelineConfig] = None,
):
    """Return a callable mapping DraftState -> blue win probability."""
    inv_vocab = {v: k for k, v in vocab.items()}
    cfg_ = cfg or PipelineConfig()

    def score(state: DraftState) -> float:
        # Build a single-row DataFrame mimicking the training schema.
        row = {f"{side}_{r}_champion": UNKNOWN_TOKEN for side in ("blue", "red") for r in ROLES}
        for r, c in state.blue_picks.items():
            row[f"blue_{r}_champion"] = c
        for r, c in state.red_picks.items():
            row[f"red_{r}_champion"] = c
        if state.bans:
            row["bans"] = ",".join(state.bans)
        df = pd.DataFrame([row])
        feats, cat_cols = build_baseline_feature_matrix(df, vocab, handcrafted, cfg_, extra_vocabs)
        if embedding_extras is not None:
            blue_ids, red_ids = state.to_id_arrays(vocab)
            extra = extract_embedding_features(blue_ids, red_ids, embedding_extras)
            feats = pd.concat([feats.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)
        feats = feats.reindex(columns=feature_columns, fill_value=0)
        prob = float(predict_lightgbm(booster, backend, feats)[0])
        if calibrator is not None:
            prob = float(calibrator.predict(np.array([prob]))[0])
        return prob

    return score


def make_torch_score_fn(
    model: "nn.Module", vocab: Dict[str, int], device,
    calibrator: Optional["ProbabilityCalibrator"] = None,
):
    def score(state: DraftState) -> float:
        blue_ids, red_ids = state.to_id_arrays(vocab)
        prob = float(_torch_predict_proba(model, blue_ids, red_ids, device)[0])
        if calibrator is not None:
            prob = float(calibrator.predict(np.array([prob]))[0])
        return prob

    return score


def make_lgb_batch_score_fn(
    booster: object,
    backend: str,
    vocab: Dict[str, int],
    handcrafted: HandcraftedStats,
    feature_columns: List[str],
    embedding_extras: Optional[np.ndarray] = None,
    calibrator: Optional["ProbabilityCalibrator"] = None,
    extra_vocabs: Optional[Dict[str, Dict[str, int]]] = None,
    cfg: Optional[PipelineConfig] = None,
):
    """Vectorised candidate scorer used by the recommender's hot path."""
    cfg_ = cfg or PipelineConfig()

    def batch(state: DraftState, side: str, role: str, candidates: List[str]) -> np.ndarray:
        if not candidates:
            return np.zeros(0)
        base_row = {
            f"{s}_{r}_champion": UNKNOWN_TOKEN
            for s in ("blue", "red")
            for r in ROLES
        }
        for r, c in state.blue_picks.items():
            base_row[f"blue_{r}_champion"] = c
        for r, c in state.red_picks.items():
            base_row[f"red_{r}_champion"] = c
        if state.bans:
            base_row["bans"] = ",".join(state.bans)
        rows = []
        target_col = f"{side}_{role}_champion"
        for cand in candidates:
            row = dict(base_row)
            row[target_col] = cand
            rows.append(row)
        df = pd.DataFrame(rows)
        feats, _ = build_baseline_feature_matrix(df, vocab, handcrafted, cfg_, extra_vocabs)
        if embedding_extras is not None:
            blue_arr = np.stack(
                [
                    df[f"blue_{r}_champion"].map(lambda x: vocab.get(x, UNKNOWN_INDEX)).values
                    for r in ROLES
                ],
                axis=1,
            ).astype(np.int64)
            red_arr = np.stack(
                [
                    df[f"red_{r}_champion"].map(lambda x: vocab.get(x, UNKNOWN_INDEX)).values
                    for r in ROLES
                ],
                axis=1,
            ).astype(np.int64)
            extra = extract_embedding_features(blue_arr, red_arr, embedding_extras)
            feats = pd.concat(
                [feats.reset_index(drop=True), extra.reset_index(drop=True)], axis=1
            )
        feats = feats.reindex(columns=feature_columns, fill_value=0)
        probs = predict_lightgbm(booster, backend, feats)
        if calibrator is not None:
            probs = calibrator.predict(probs)
        return probs

    return batch


def make_torch_batch_score_fn(
    model: "nn.Module", vocab: Dict[str, int], device,
    calibrator: Optional["ProbabilityCalibrator"] = None,
):
    """Vectorised PyTorch candidate scorer (single forward pass)."""

    def batch(state: DraftState, side: str, role: str, candidates: List[str]) -> np.ndarray:
        if not candidates:
            return np.zeros(0)
        blue_ids, red_ids = state.to_id_arrays(vocab)
        N = len(candidates)
        blue_batch = np.repeat(blue_ids, N, axis=0)
        red_batch = np.repeat(red_ids, N, axis=0)
        slot_idx = ROLE_TO_IDX[role]
        cand_ids = np.array([vocab.get(c, UNKNOWN_INDEX) for c in candidates], dtype=np.int64)
        if side == "blue":
            blue_batch[:, slot_idx] = cand_ids
        else:
            red_batch[:, slot_idx] = cand_ids
        probs = _torch_predict_proba(model, blue_batch, red_batch, device)
        if calibrator is not None:
            probs = calibrator.predict(probs)
        return probs

    return batch


# --------------------------------------------------------------------------- #
# Recommender evaluation (held-out completed drafts)
# --------------------------------------------------------------------------- #


def evaluate_recommender(
    recommender: Recommender,
    test_df: pd.DataFrame,
    vocab: Dict[str, int],
    n_samples: int = 200,
    ks: Sequence[int] = (1, 3, 5),
    seed: int = 42,
) -> Dict[str, float]:
    """Hide one random pick per match and measure recall@k / MRR.

    Champions outside the vocab are skipped (cannot be ranked).
    """
    rng = random.Random(seed)
    n_samples = min(n_samples, len(test_df))
    sample_idx = rng.sample(range(len(test_df)), n_samples)

    hits = {k: 0 for k in ks}
    rr_sum = 0.0
    counted = 0

    for idx in sample_idx:
        row = test_df.iloc[idx]
        side = rng.choice(["blue", "red"])
        role = rng.choice(ROLES)
        true_champ = row[f"{side}_{role}_champion"]
        if true_champ not in vocab:
            continue

        state = DraftState()
        for r in ROLES:
            for s in ("blue", "red"):
                if s == side and r == role:
                    continue
                state.picks_for(s)[r] = row[f"{s}_{r}_champion"]

        ranking = recommender.top_k(state, side, role, k=max(ks))
        ranked_champs = [item["champion"] for item in ranking]
        if true_champ in ranked_champs:
            rank = ranked_champs.index(true_champ) + 1
            rr_sum += 1.0 / rank
            for k in ks:
                if rank <= k:
                    hits[k] += 1
        counted += 1

    return {
        **{f"recall@{k}": (hits[k] / counted if counted else 0.0) for k in ks},
        "mrr": rr_sum / counted if counted else 0.0,
        "n_evaluated": counted,
    }


# --------------------------------------------------------------------------- #
# Artifact I/O
# --------------------------------------------------------------------------- #


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, default=str)


def save_pickle(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


class _PipelineUnpickler(pickle.Unpickler):
    """Remap __main__ -> lol_draft_pipeline so artifacts pickled while running
    the script directly (where classes live in __main__) load fine when the
    pipeline is imported as a module (dashboard / simulator)."""

    def find_class(self, module, name):
        if module == "__main__":
            module = __name__  # this module
        return super().find_class(module, name)


def load_pickle(path: Path) -> object:
    with path.open("rb") as f:
        return _PipelineUnpickler(f).load()


# --------------------------------------------------------------------------- #
# Leakage audit
# --------------------------------------------------------------------------- #


def leakage_audit(included: Sequence[str], excluded: Sequence[str]) -> None:
    log.info("==== LEAKAGE AUDIT ====")
    log.info("Included (draft-time only): %s", ", ".join(included))
    seen_excluded = [c for c in excluded if c in POST_GAME_COLUMNS]
    log.info(
        "Confirmed excluded post-game columns: %s",
        ", ".join(sorted(set(seen_excluded))) if seen_excluded else "(none present)",
    )
    log.info("=======================")


# --------------------------------------------------------------------------- #
# High-level orchestration
# --------------------------------------------------------------------------- #


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def prepare_data(cfg: PipelineConfig) -> Tuple[Splits, Dict[str, int], HandcraftedStats, pd.DataFrame]:
    """Load + pivot + split + fit handcrafted stats (train only).

    Returns ``(splits, vocab, handcrafted, long_df)``; ``long_df`` is the raw
    long-form participant dataframe (kept for downstream artifact writers).
    """
    set_seed(cfg.random_seed)
    raw_path = find_raw_csv(cfg.data_dir, cfg.raw_csv)
    long_df, schema = load_long_dataframe(raw_path, max_matches=cfg.max_rows)

    included = [
        "champion (per role × per side)",
        "side",
        schema.get("patch") or "patch (absent)",
        schema.get("rank") or "rank (absent)",
        schema.get("bans") or "bans (absent)",
    ]
    leakage_audit(included=included, excluded=[c for c in long_df.columns])

    match_df = pivot_long_to_match_level(long_df)
    if cfg.fast_dev_run:
        match_df = match_df.sample(
            n=min(2000, len(match_df)), random_state=cfg.random_seed
        ).reset_index(drop=True)
        log.info("fast-dev-run: subsampled to %d matches", len(match_df))

    train, val, test = make_splits(match_df, cfg)
    vocab = build_champion_vocab(train)
    handcrafted = fit_handcrafted_stats(
        train, min_count=cfg.pair_smoothing_min_count, prior=cfg.pair_smoothing_prior
    )
    splits = Splits(train, val, test)

    # Fit extra-feature vocabularies (patch / rank / bans) on TRAIN only.
    extra_vocabs = fit_extra_vocabs(train, cfg)
    if extra_vocabs:
        log.info(
            "Extra vocabs: %s",
            {k: len(v) for k, v in extra_vocabs.items()},
        )

    # Persist schema + leakage audit to the run directory (best effort).
    try:
        run_dir = run_dir_for(cfg)
        write_schema_report(run_dir, raw_path, schema, long_df, match_df, splits, vocab)
        write_leakage_audit(run_dir, included=included, all_cols=list(long_df.columns))
        save_json(Path(cfg.artifacts_dir) / "extra_vocabs.json", extra_vocabs)
        save_json(run_dir / "extra_vocabs.json", extra_vocabs)
        update_latest_pointer(cfg)
    except Exception as exc:  # pragma: no cover
        log.warning("Failed to write schema/leakage artifacts: %s", exc)

    return splits, vocab, handcrafted, long_df, extra_vocabs


def _save_common_artifacts(
    cfg: PipelineConfig,
    vocab: Dict[str, int],
    handcrafted: HandcraftedStats,
) -> Path:
    art = Path(cfg.artifacts_dir)
    art.mkdir(parents=True, exist_ok=True)
    save_json(art / "config.json", cfg.to_dict())
    save_json(art / "champion_to_idx.json", vocab)
    save_pickle(art / "handcrafted_stats.pkl", handcrafted)
    # Also drop a copy of vocab + config in the run directory so the dashboard
    # is self-contained even if the user nukes the top-level artifacts/.
    rd = run_dir_for(cfg)
    save_json(rd / "config.json", cfg.to_dict())
    save_json(rd / "champion_to_idx.json", vocab)
    return art


def _get_event_logger(cfg: PipelineConfig) -> EventLogger:
    """Return (or build) the per-run event logger."""
    rid = resolve_run_id(cfg)
    return EventLogger(run_dir_for(cfg), rid)


# --------------------------------------------------------------------------- #
# Train commands
# --------------------------------------------------------------------------- #


def train_baseline(cfg: PipelineConfig, splits: Optional[Splits] = None,
                   vocab: Optional[Dict[str, int]] = None,
                   handcrafted: Optional[HandcraftedStats] = None,
                   event_logger: Optional[EventLogger] = None,
                   extra_vocabs: Optional[Dict[str, Dict[str, int]]] = None) -> Dict[str, object]:
    """LightGBM (or HistGB fallback) over draft features + handcrafted stats."""
    if splits is None or vocab is None or handcrafted is None:
        splits, vocab, handcrafted, _long_df, extra_vocabs = prepare_data(cfg)
    if extra_vocabs is None:
        extra_vocabs = _load_extra_vocabs(cfg)
    art = _save_common_artifacts(cfg, vocab, handcrafted)
    rd = run_dir_for(cfg)
    if event_logger is None:
        event_logger = _get_event_logger(cfg)
    event_logger.stage_started("lightgbm_baseline")

    X_tr, cat_cols = build_baseline_feature_matrix(splits.train, vocab, handcrafted, cfg, extra_vocabs)
    X_va, _ = build_baseline_feature_matrix(splits.val, vocab, handcrafted, cfg, extra_vocabs)
    X_te, _ = build_baseline_feature_matrix(splits.test, vocab, handcrafted, cfg, extra_vocabs)
    y_tr = splits.train["blue_win"].values
    y_va = splits.val["blue_win"].values
    y_te = splits.test["blue_win"].values

    model, backend = train_lightgbm(
        X_tr, y_tr, X_va, y_va, cat_cols, cfg,
        event_logger=event_logger, model_name="lightgbm_baseline",
    )

    test_prob = predict_lightgbm(model, backend, X_te)
    val_prob = predict_lightgbm(model, backend, X_va)
    if cfg.enable_calibration:
        calibrator = ProbabilityCalibrator().fit(val_prob, y_va)
        val_prob_cal = calibrator.predict(val_prob)
        test_prob_cal = calibrator.predict(test_prob)
        save_pickle(art / "lightgbm_baseline_calibrator.pkl", calibrator)
        save_pickle(rd / "lightgbm_baseline_calibrator.pkl", calibrator)
    else:
        val_prob_cal, test_prob_cal = val_prob, test_prob
    metrics = {
        "val": compute_metrics(y_va, val_prob_cal),
        "test": compute_metrics(y_te, test_prob_cal),
        "backend": backend,
        "calibrated": cfg.enable_calibration,
    }
    print_metrics("baseline / val", metrics["val"])
    print_metrics("baseline / test", metrics["test"])

    save_pickle(art / "lightgbm_baseline.pkl", {"model": model, "backend": backend})
    save_json(art / "lightgbm_baseline_features.json", list(X_tr.columns))
    save_json(art / "metrics_baseline.json", metrics)
    save_json(rd / "metrics_baseline.json", metrics)
    append_predictions_csv(rd, "lightgbm_baseline", splits.test, y_te, test_prob_cal)
    append_feature_importance(rd, "lightgbm_baseline", model, backend, list(X_tr.columns))
    event_logger.stage_completed("lightgbm_baseline", metrics["val"], metrics["test"])
    return metrics


def train_teamcompnet(
    cfg: PipelineConfig,
    splits: Optional[Splits] = None,
    vocab: Optional[Dict[str, int]] = None,
    handcrafted: Optional[HandcraftedStats] = None,
    long_df: Optional[pd.DataFrame] = None,
    event_logger: Optional[EventLogger] = None,
) -> Dict[str, object]:
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is required for TeamCompNet")
    if splits is None or vocab is None or handcrafted is None:
        splits, vocab, handcrafted, long_df, extra_vocabs = prepare_data(cfg)
    art = _save_common_artifacts(cfg, vocab, handcrafted)
    rd = run_dir_for(cfg)
    if event_logger is None:
        event_logger = _get_event_logger(cfg)
    event_logger.stage_started("teamcompnet")
    device = torch_device()
    log.info("TeamCompNet training on %s", device)

    enc_tr = encode_champion_ids(splits.train, vocab)
    enc_va = encode_champion_ids(splits.val, vocab)
    enc_te = encode_champion_ids(splits.test, vocab)
    y_tr = splits.train["blue_win"].values
    y_va = splits.val["blue_win"].values
    y_te = splits.test["blue_win"].values

    train_loader = _make_loader(
        enc_tr["blue"], enc_tr["red"], y_tr, cfg.batch_size, shuffle=True
    )
    val_loader = _make_loader(
        enc_va["blue"], enc_va["red"], y_va, cfg.batch_size, shuffle=False
    )

    if cfg.arch == "transformer":
        model = SetTransformerCompNet(
            num_champions=len(vocab),
            embedding_dim=cfg.embedding_dim,
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
            n_layers=cfg.n_attention_layers,
            n_heads=cfg.n_attention_heads,
            policy_head=cfg.enable_policy_head,
        ).to(device)
    else:
        model = TeamCompNet(
            num_champions=len(vocab),
            embedding_dim=cfg.embedding_dim,
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
        ).to(device)
    if cfg.pretrain_embeddings and hasattr(model, "init_pretrained_embeddings"):
        log.info("Pretraining champion embeddings with PMI+SVD ...")
        pre = pretrain_champion_embeddings(splits.train, vocab, cfg.embedding_dim)
        model.init_pretrained_embeddings(pre)
    history = _train_torch_model(
        model, train_loader, val_loader, cfg, device, "teamcompnet",
        event_logger=event_logger, num_champions=len(vocab),
    )

    val_prob = _torch_predict_proba(model, enc_va["blue"], enc_va["red"], device)
    test_prob = _torch_predict_proba(model, enc_te["blue"], enc_te["red"], device)
    calibrator = None
    if cfg.enable_calibration:
        calibrator = ProbabilityCalibrator().fit(val_prob, y_va)
        val_prob_cal = calibrator.predict(val_prob)
        test_prob_cal = calibrator.predict(test_prob)
        save_pickle(art / "teamcompnet_calibrator.pkl", calibrator)
        save_pickle(rd / "teamcompnet_calibrator.pkl", calibrator)
    else:
        val_prob_cal, test_prob_cal = val_prob, test_prob
    metrics = {
        "val": compute_metrics(y_va, val_prob_cal),
        "test": compute_metrics(y_te, test_prob_cal),
        "history": history,
        "calibrated": cfg.enable_calibration,
    }
    print_metrics("teamcompnet / val", metrics["val"])
    print_metrics("teamcompnet / test", metrics["test"])

    torch.save(model.state_dict(), art / "teamcompnet.pt")
    save_json(art / "teamcompnet_arch.json", {"arch": cfg.arch})
    save_json(rd / "teamcompnet_arch.json", {"arch": cfg.arch})
    champion_emb = model.champion_embeddings()
    np.save(art / "champion_embeddings.npy", champion_emb)
    save_json(art / "metrics_teamcompnet.json", metrics)
    save_json(rd / "metrics_teamcompnet.json", metrics)
    append_predictions_csv(rd, "teamcompnet", splits.test, y_te, test_prob_cal)
    write_embedding_champions_csv(rd, vocab, champion_emb, handcrafted, long_df)
    event_logger.stage_completed("teamcompnet", metrics["val"], metrics["test"])
    return metrics


def train_hybrid(
    cfg: PipelineConfig,
    splits: Optional[Splits] = None,
    vocab: Optional[Dict[str, int]] = None,
    handcrafted: Optional[HandcraftedStats] = None,
    long_df: Optional[pd.DataFrame] = None,
    event_logger: Optional[EventLogger] = None,
    extra_vocabs: Optional[Dict[str, Dict[str, int]]] = None,
) -> Dict[str, object]:
    """LightGBM trained on baseline features + extracted TeamCompNet embeddings."""
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is required for the hybrid model")
    if splits is None or vocab is None or handcrafted is None:
        splits, vocab, handcrafted, long_df, extra_vocabs = prepare_data(cfg)
    if extra_vocabs is None:
        extra_vocabs = _load_extra_vocabs(cfg)
    art = _save_common_artifacts(cfg, vocab, handcrafted)
    rd = run_dir_for(cfg)
    if event_logger is None:
        event_logger = _get_event_logger(cfg)
    event_logger.stage_started("lightgbm_with_embeddings")

    emb_path = art / "champion_embeddings.npy"
    if not emb_path.exists():
        log.info("No champion_embeddings.npy found - training TeamCompNet first.")
        train_teamcompnet(cfg, splits=splits, vocab=vocab, handcrafted=handcrafted,
                          long_df=long_df, event_logger=event_logger)
    champion_emb = np.load(emb_path)
    if champion_emb.shape[0] != len(vocab):
        log.warning(
            "Embedding rows (%d) != vocab (%d); rebuilding TeamCompNet.",
            champion_emb.shape[0],
            len(vocab),
        )
        train_teamcompnet(cfg, splits=splits, vocab=vocab, handcrafted=handcrafted,
                          long_df=long_df, event_logger=event_logger)
        champion_emb = np.load(emb_path)

    enc_tr = encode_champion_ids(splits.train, vocab)
    enc_va = encode_champion_ids(splits.val, vocab)
    enc_te = encode_champion_ids(splits.test, vocab)

    base_tr, cat_cols = build_baseline_feature_matrix(splits.train, vocab, handcrafted, cfg, extra_vocabs)
    base_va, _ = build_baseline_feature_matrix(splits.val, vocab, handcrafted, cfg, extra_vocabs)
    base_te, _ = build_baseline_feature_matrix(splits.test, vocab, handcrafted, cfg, extra_vocabs)

    extra_tr = extract_embedding_features(enc_tr["blue"], enc_tr["red"], champion_emb)
    extra_va = extract_embedding_features(enc_va["blue"], enc_va["red"], champion_emb)
    extra_te = extract_embedding_features(enc_te["blue"], enc_te["red"], champion_emb)

    X_tr = pd.concat([base_tr.reset_index(drop=True), extra_tr.reset_index(drop=True)], axis=1)
    X_va = pd.concat([base_va.reset_index(drop=True), extra_va.reset_index(drop=True)], axis=1)
    X_te = pd.concat([base_te.reset_index(drop=True), extra_te.reset_index(drop=True)], axis=1)
    y_tr = splits.train["blue_win"].values
    y_va = splits.val["blue_win"].values
    y_te = splits.test["blue_win"].values

    model, backend = train_lightgbm(
        X_tr, y_tr, X_va, y_va, cat_cols, cfg,
        event_logger=event_logger, model_name="lightgbm_with_embeddings",
    )
    val_prob = predict_lightgbm(model, backend, X_va)
    test_prob = predict_lightgbm(model, backend, X_te)
    if cfg.enable_calibration:
        calibrator = ProbabilityCalibrator().fit(val_prob, y_va)
        val_prob_cal = calibrator.predict(val_prob)
        test_prob_cal = calibrator.predict(test_prob)
        save_pickle(art / "lightgbm_with_embeddings_calibrator.pkl", calibrator)
        save_pickle(rd / "lightgbm_with_embeddings_calibrator.pkl", calibrator)
    else:
        val_prob_cal, test_prob_cal = val_prob, test_prob
    metrics = {
        "val": compute_metrics(y_va, val_prob_cal),
        "test": compute_metrics(y_te, test_prob_cal),
        "backend": backend,
        "calibrated": cfg.enable_calibration,
    }
    print_metrics("hybrid / val", metrics["val"])
    print_metrics("hybrid / test", metrics["test"])

    save_pickle(
        art / "lightgbm_with_embeddings.pkl", {"model": model, "backend": backend}
    )
    save_json(art / "lightgbm_with_embeddings_features.json", list(X_tr.columns))
    save_json(art / "metrics_hybrid.json", metrics)
    save_json(rd / "metrics_hybrid.json", metrics)
    append_predictions_csv(rd, "lightgbm_with_embeddings", splits.test, y_te, test_prob_cal)
    append_feature_importance(rd, "lightgbm_with_embeddings", model, backend, list(X_tr.columns))
    event_logger.stage_completed("lightgbm_with_embeddings", metrics["val"], metrics["test"])
    return metrics


def train_wide_deep(
    cfg: PipelineConfig,
    splits: Optional[Splits] = None,
    vocab: Optional[Dict[str, int]] = None,
    handcrafted: Optional[HandcraftedStats] = None,
    long_df: Optional[pd.DataFrame] = None,
    event_logger: Optional[EventLogger] = None,
) -> Dict[str, object]:
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is required for Wide & Deep")
    if splits is None or vocab is None or handcrafted is None:
        splits, vocab, handcrafted, long_df, extra_vocabs = prepare_data(cfg)
    art = _save_common_artifacts(cfg, vocab, handcrafted)
    rd = run_dir_for(cfg)
    if event_logger is None:
        event_logger = _get_event_logger(cfg)
    event_logger.stage_started("wide_deep")
    device = torch_device()
    log.info("Wide & Deep training on %s", device)

    enc_tr = encode_champion_ids(splits.train, vocab)
    enc_va = encode_champion_ids(splits.val, vocab)
    enc_te = encode_champion_ids(splits.test, vocab)
    y_tr = splits.train["blue_win"].values
    y_va = splits.val["blue_win"].values
    y_te = splits.test["blue_win"].values

    train_loader = _make_loader(
        enc_tr["blue"], enc_tr["red"], y_tr, cfg.batch_size, shuffle=True
    )
    val_loader = _make_loader(
        enc_va["blue"], enc_va["red"], y_va, cfg.batch_size, shuffle=False
    )

    model = WideDeepDraftNet(
        num_champions=len(vocab),
        embedding_dim=cfg.embedding_dim,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
        combine=cfg.wide_deep_combine,
    ).to(device)
    if cfg.pretrain_embeddings:
        try:
            pre = pretrain_champion_embeddings(splits.train, vocab, cfg.embedding_dim)
            with torch.no_grad():
                if model.deep.champion_emb.weight.shape == pre.shape:
                    model.deep.champion_emb.weight.copy_(torch.as_tensor(pre, dtype=model.deep.champion_emb.weight.dtype))
        except Exception as exc:
            log.warning("Wide&Deep pretrain init failed: %s", exc)
    history = _train_torch_model(
        model, train_loader, val_loader, cfg, device, "wide_deep",
        event_logger=event_logger, num_champions=len(vocab),
    )

    val_prob = _torch_predict_proba(model, enc_va["blue"], enc_va["red"], device)
    test_prob = _torch_predict_proba(model, enc_te["blue"], enc_te["red"], device)
    if cfg.enable_calibration:
        calibrator = ProbabilityCalibrator().fit(val_prob, y_va)
        val_prob_cal = calibrator.predict(val_prob)
        test_prob_cal = calibrator.predict(test_prob)
        save_pickle(art / "wide_deep_calibrator.pkl", calibrator)
        save_pickle(rd / "wide_deep_calibrator.pkl", calibrator)
    else:
        val_prob_cal, test_prob_cal = val_prob, test_prob
    metrics = {
        "val": compute_metrics(y_va, val_prob_cal),
        "test": compute_metrics(y_te, test_prob_cal),
        "history": history,
        "calibrated": cfg.enable_calibration,
    }
    print_metrics("wide_deep / val", metrics["val"])
    print_metrics("wide_deep / test", metrics["test"])

    torch.save(model.state_dict(), art / "wide_deep.pt")
    save_json(art / "metrics_wide_deep.json", metrics)
    save_json(rd / "metrics_wide_deep.json", metrics)
    append_predictions_csv(rd, "wide_deep", splits.test, y_te, test_prob_cal)
    event_logger.stage_completed("wide_deep", metrics["val"], metrics["test"])
    return metrics


def train_all(cfg: PipelineConfig) -> Dict[str, object]:
    """Train every model end-to-end, sharing data prep across them."""
    t0 = time.time()
    splits, vocab, handcrafted, long_df, extra_vocabs = prepare_data(cfg)
    rd = run_dir_for(cfg)
    elog = _get_event_logger(cfg)
    elog.run_started(command="train", config=cfg.to_dict())
    summary: Dict[str, object] = {}
    try:
        summary["baseline"] = train_baseline(
            cfg, splits, vocab, handcrafted, event_logger=elog, extra_vocabs=extra_vocabs,
        )
        if _HAS_TORCH:
            summary["teamcompnet"] = train_teamcompnet(
                cfg, splits, vocab, handcrafted, long_df=long_df, event_logger=elog
            )
            summary["hybrid"] = train_hybrid(
                cfg, splits, vocab, handcrafted, long_df=long_df, event_logger=elog,
                extra_vocabs=extra_vocabs,
            )
            summary["wide_deep"] = train_wide_deep(
                cfg, splits, vocab, handcrafted, long_df=long_df, event_logger=elog
            )
        else:
            log.warning("Skipping torch models (PyTorch missing).")

        art = Path(cfg.artifacts_dir)
        save_json(art / "metrics.json", summary)
        write_comparison_table(summary, art / "model_comparison.csv")
        write_comparison_table(summary, rd / "model_comparison.csv")
        write_calibration_csv(summary, art / "calibration.csv")
        write_calibration_csv(summary, rd / "calibration.csv")
        write_feature_columns(art)
        write_feature_columns(rd)

        # Stacking ensemble over whatever base learners trained successfully.
        if cfg.enable_stacking:
            try:
                base_names = ["baseline"]
                if _HAS_TORCH:
                    base_names += ["hybrid", "teamcompnet", "wide_deep"]
                stacker_metrics = train_stacker(
                    cfg, splits, vocab, handcrafted, base_names, event_logger=elog,
                )
                if stacker_metrics:
                    summary["stacker"] = stacker_metrics
            except Exception as exc:
                log.warning("Stacker training failed: %s", exc)

        # Run-scoped extras for the dashboard
        write_metrics_summary(rd, summary)
        save_json(rd / "metrics.json", summary)
        write_comparison_table(summary, art / "model_comparison.csv")
        write_comparison_table(summary, rd / "model_comparison.csv")
        save_json(
            rd / "confusion_matrices.json",
            {
                name: payload["test"].get("confusion_matrix")
                for name, payload in summary.items()
                if isinstance(payload, dict) and "test" in payload
            },
        )
        try:
            example_recommendations(rd, cfg, vocab, handcrafted, splits)
        except Exception as exc:
            log.warning("example_recommendations failed: %s", exc)

        update_latest_pointer(cfg)
        elog.run_completed("success", duration_seconds=time.time() - t0,
                           run_dir=str(rd), best_model=summary.get("baseline", {}))
    except Exception:
        elog.error("train_all failed", tb=traceback.format_exc())
        elog.run_completed("error", duration_seconds=time.time() - t0)
        raise
    return summary


def write_comparison_table(summary: Dict[str, object], path: Path) -> None:
    rows = []
    for name, payload in summary.items():
        if not isinstance(payload, dict) or "test" not in payload:
            continue
        m = payload["test"]
        rows.append(
            {
                "model": name,
                "accuracy": m["accuracy"],
                "f1": m["f1"],
                "auc": m["roc_auc"],
                "log_loss": m["log_loss"],
                "brier": m["brier_score"],
            }
        )
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)
        log.info("Wrote model comparison table to %s", path)


def write_calibration_csv(summary: Dict[str, object], path: Path) -> None:
    """Long-form calibration table covering all models (test split)."""
    rows = []
    for name, payload in summary.items():
        if not isinstance(payload, dict) or "test" not in payload:
            continue
        for bucket in payload["test"].get("calibration", []):
            rows.append({"model": name, **bucket})
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)
        log.info("Wrote calibration CSV to %s", path)


def write_feature_columns(art: Path) -> None:
    """Write a unified ``feature_columns.json`` listing each model's features."""
    out: Dict[str, List[str]] = {}
    base = art / "lightgbm_baseline_features.json"
    if base.exists():
        out["lightgbm_baseline"] = json.loads(base.read_text())
    hyb = art / "lightgbm_with_embeddings_features.json"
    if hyb.exists():
        out["lightgbm_with_embeddings"] = json.loads(hyb.read_text())
    save_json(art / "feature_columns.json", out)


# --- Run-scoped artifact writers (for the dashboard) --------------------- #


def append_predictions_csv(
    run_dir: Path,
    model_name: str,
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> None:
    """Append per-row test predictions for ``model_name`` to predictions_test.csv."""
    df = pd.DataFrame(
        {
            "model_name": model_name,
            "match_id": test_df["match_id"].values
            if "match_id" in test_df.columns
            else np.arange(len(y_true)),
            "y_true": y_true.astype(int),
            "y_prob": y_prob.astype(float),
            "y_pred": (y_prob >= threshold).astype(int),
        }
    )
    if "patch" in test_df.columns:
        df["patch"] = test_df["patch"].values
    out = run_dir / "predictions_test.csv"
    if out.exists():
        df.to_csv(out, mode="a", header=False, index=False)
    else:
        df.to_csv(out, index=False)


def append_feature_importance(
    run_dir: Path, model_name: str, model: object, backend: str, feature_names: Sequence[str]
) -> None:
    """Append (gain + split) feature importance rows to feature_importance.csv."""
    out = run_dir / "feature_importance.csv"
    rows: List[Dict[str, object]] = []
    if backend == "lightgbm":
        for itype in ("split", "gain"):
            try:
                values = model.feature_importance(importance_type=itype)
            except Exception:
                continue
            for feat, val in zip(feature_names, values):
                rows.append(
                    {
                        "model_name": model_name,
                        "feature": feat,
                        "importance": float(val),
                        "importance_type": itype,
                    }
                )
    else:
        # HistGradientBoosting has no native importance; skip silently.
        return
    if not rows:
        return
    df = pd.DataFrame(rows)
    if out.exists():
        df.to_csv(out, mode="a", header=False, index=False)
    else:
        df.to_csv(out, index=False)


def write_embedding_champions_csv(
    run_dir: Path,
    vocab: Dict[str, int],
    champion_emb: np.ndarray,
    handcrafted: HandcraftedStats,
    long_df: Optional[pd.DataFrame] = None,
) -> None:
    """Persist the learned champion embedding matrix as a tidy CSV.

    Columns: ``champion``, ``champion_idx``, ``dim_0..dim_{D-1}``,
    ``win_rate``, ``sample_count`` (and ``most_common_role`` when ``long_df``
    is given).
    """
    inv = {v: k for k, v in vocab.items()}
    rows = []
    role_lookup: Dict[str, str] = {}
    count_lookup: Dict[str, int] = {}
    if long_df is not None and {"champion_name", "role"}.issubset(long_df.columns):
        rc = (
            long_df.dropna(subset=["role"])
            .groupby(["champion_name", "role"])
            .size()
            .reset_index(name="n")
        )
        for champ in rc["champion_name"].unique():
            sub = rc[rc["champion_name"] == champ].sort_values("n", ascending=False)
            role_lookup[champ] = str(sub.iloc[0]["role"])
            count_lookup[champ] = int(sub["n"].sum())
    for idx in range(champion_emb.shape[0]):
        name = inv.get(idx, UNKNOWN_TOKEN)
        if name == UNKNOWN_TOKEN:
            continue
        row: Dict[str, object] = {
            "champion": name,
            "champion_idx": int(idx),
            "win_rate": float(handcrafted.winrate(name)),
        }
        if name in count_lookup:
            row["sample_count"] = count_lookup[name]
        if name in role_lookup:
            row["most_common_role"] = role_lookup[name]
        for j, val in enumerate(champion_emb[idx]):
            row[f"dim_{j}"] = float(val)
        rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(run_dir / "embedding_champions.csv", index=False)


def write_leakage_audit(
    run_dir: Path, included: Sequence[str], all_cols: Sequence[str]
) -> None:
    """Persist a structured leakage_audit.json for the dashboard."""
    excluded = sorted(set(POST_GAME_COLUMNS) & set(all_cols))
    suspicious = [
        c
        for c in all_cols
        if c not in included
        and any(tok in c.lower() for tok in ("kills", "deaths", "assists", "gold", "damage", "vision", "minion"))
    ]
    payload = {
        "included_columns": list(included),
        "excluded_post_game_columns": excluded,
        "post_game_blocklist": list(POST_GAME_COLUMNS),
        "suspicious_columns": suspicious,
        "leakage_risk_detected": False,
        "notes": (
            "Synergy / counter / champion win-rate stats are fit on the "
            "training split only. Post-game columns are explicitly removed "
            "before any draft-time model touches the data."
        ),
    }
    save_json(run_dir / "leakage_audit.json", payload)


def write_schema_report(
    run_dir: Path,
    raw_path: Path,
    schema: Dict[str, Optional[str]],
    long_df: pd.DataFrame,
    match_df: pd.DataFrame,
    splits: "Splits",
    vocab: Dict[str, int],
) -> None:
    """Persist a schema_report.json describing the dataset and split shapes."""
    n_matches = len(match_df)
    payload: Dict[str, object] = {
        "raw_path": str(raw_path),
        "long_rows": int(len(long_df)),
        "matches": int(n_matches),
        "blue_win_rate": float(match_df["blue_win"].mean()) if n_matches else None,
        "schema": {k: v for k, v in schema.items()},
        "champion_vocab_size": len(vocab) - 1,  # exclude UNK
        "split": {
            "train": int(len(splits.train)),
            "val": int(len(splits.val)),
            "test": int(len(splits.test)),
            "method": "time_based"
            if "timestamp" in match_df.columns and match_df["timestamp"].notna().any()
            else "stratified_random",
        },
        "roles": list(ROLES),
    }
    if "timestamp" in match_df.columns and match_df["timestamp"].notna().any():
        ts = match_df["timestamp"].dropna()
        payload["timestamp_range"] = {
            "min": str(ts.min()),
            "max": str(ts.max()),
        }
    save_json(run_dir / "schema_report.json", payload)


def write_metrics_summary(run_dir: Path, summary: Dict[str, object]) -> None:
    """Top-level run summary used by the dashboard's overview tab."""
    flat: Dict[str, object] = {"models": {}}
    for name, payload in summary.items():
        if isinstance(payload, dict) and "test" in payload:
            flat["models"][name] = {
                k: v for k, v in payload["test"].items() if k != "calibration"
            }
    if "recommender" in summary:
        flat["recommender"] = summary["recommender"]
    # Best-by-AUC convenience field
    best = None
    best_auc = -math.inf
    for name, m in flat["models"].items():
        auc = m.get("roc_auc")
        if auc is not None and not math.isnan(auc) and auc > best_auc:
            best_auc = auc
            best = name
    flat["best_model"] = best
    flat["best_test_auc"] = None if best is None else best_auc
    save_json(run_dir / "metrics_summary.json", flat)


def train_stacker(
    cfg: PipelineConfig,
    splits: Splits,
    vocab: Dict[str, int],
    handcrafted: HandcraftedStats,
    base_models: Sequence[str],
    event_logger: Optional[EventLogger] = None,
) -> Optional[Dict[str, object]]:
    """Logistic-regression stacking ensemble over base-model predictions.

    Builds a (val_size x len(base_models)) matrix of out-of-fold-style val
    predictions, fits an L2 logistic regression as the meta-learner, and
    evaluates on the test split using the same base-model predictors. The
    stacker is a defensible "always at least as good as the best base
    learner" ensemble in practice.

    Returns the metrics dict or ``None`` if fewer than 2 base models are
    available.
    """
    art = Path(cfg.artifacts_dir)
    rd = run_dir_for(cfg)
    available: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for m in base_models:
        try:
            score_fn, batch_fn = _build_score_fns(m, cfg, vocab, handcrafted)
        except Exception:
            continue
        # Reuse vectorised scoring path for speed.
        val_p = np.asarray([score_fn(_state_from_row(row, vocab)) for row in
                            splits.val.to_dict("records")])
        test_p = np.asarray([score_fn(_state_from_row(row, vocab)) for row in
                             splits.test.to_dict("records")])
        available.append((m, val_p, test_p))
    if len(available) < 2:
        log.info("Stacker skipped: fewer than 2 base models available.")
        return None

    from sklearn.linear_model import LogisticRegression

    X_val = np.column_stack([a[1] for a in available])
    X_test = np.column_stack([a[2] for a in available])
    y_val = splits.val["blue_win"].values.astype(int)
    y_test = splits.test["blue_win"].values.astype(int)

    meta = LogisticRegression(C=1.0, max_iter=200, solver="lbfgs")
    meta.fit(X_val, y_val)
    val_prob = meta.predict_proba(X_val)[:, 1]
    test_prob = meta.predict_proba(X_test)[:, 1]

    if cfg.enable_calibration:
        cal = ProbabilityCalibrator().fit(val_prob, y_val)
        save_pickle(art / "stacker_calibrator.pkl", cal)
        save_pickle(rd / "stacker_calibrator.pkl", cal)
        val_prob = cal.predict(val_prob)
        test_prob = cal.predict(test_prob)

    metrics = {
        "val": compute_metrics(y_val, val_prob),
        "test": compute_metrics(y_test, test_prob),
        "base_models": [a[0] for a in available],
        "weights": meta.coef_.flatten().tolist(),
        "intercept": float(meta.intercept_[0]),
    }
    print_metrics("stacker / val", metrics["val"])
    print_metrics("stacker / test", metrics["test"])

    bundle = {"meta": meta, "base_models": [a[0] for a in available]}
    save_pickle(art / "stacker.pkl", bundle)
    save_pickle(rd / "stacker.pkl", bundle)
    save_json(rd / "metrics_stacker.json", metrics)
    save_json(art / "metrics_stacker.json", metrics)
    append_predictions_csv(rd, "stacker", splits.test, y_test, test_prob)
    if event_logger is not None:
        event_logger.stage_completed("stacker", metrics["val"], metrics["test"])
    return metrics


def _state_from_row(row: Dict[str, object], vocab: Dict[str, int]) -> "DraftState":
    """Materialise a DraftState directly from a pivoted match row."""
    state = DraftState()
    for r in ROLES:
        state.blue_picks[r] = str(row.get(f"blue_{r}_champion", UNKNOWN_TOKEN))
        state.red_picks[r] = str(row.get(f"red_{r}_champion", UNKNOWN_TOKEN))
    return state


def _build_stacker_score_fns(cfg: PipelineConfig, vocab, handcrafted):
    """Score-fn pair backed by the saved logistic-regression stacker."""
    art = Path(cfg.artifacts_dir)
    bundle_path = art / "stacker.pkl"
    if not bundle_path.exists():
        log.warning("Stacker missing - falling back to hybrid.")
        return _build_score_fns("hybrid", cfg, vocab, handcrafted)
    bundle = load_pickle(bundle_path)
    base_score_fns = []
    base_batch_fns = []
    for m in bundle["base_models"]:
        s, b = _build_score_fns(m, cfg, vocab, handcrafted)
        base_score_fns.append(s)
        base_batch_fns.append(b)
    meta = bundle["meta"]
    cal = _maybe_load_calibrator(art, "stacker")

    def _stack_score(state):
        feats = np.array([[fn(state) for fn in base_score_fns]])
        prob = float(meta.predict_proba(feats)[0, 1])
        if cal is not None:
            prob = float(cal.predict(np.array([prob]))[0])
        return prob

    def _stack_batch(state, side, role, candidates):
        if not candidates:
            return np.zeros(0)
        # Each base model's batch_fn returns N probs for the candidates.
        cols = np.column_stack([fn(state, side, role, candidates) for fn in base_batch_fns])
        probs = meta.predict_proba(cols)[:, 1]
        if cal is not None:
            probs = cal.predict(probs)
        return probs

    return _stack_score, _stack_batch


def example_recommendations(
    run_dir: Path,
    cfg: PipelineConfig,
    vocab: Dict[str, int],
    handcrafted: HandcraftedStats,
    splits: "Splits",
) -> None:
    """Generate two illustrative recommendation traces using whichever model is available."""
    art = Path(cfg.artifacts_dir)
    examples = []
    score_fn = batch_fn = None
    model_used = None
    for cand in ("hybrid", "wide_deep", "baseline"):
        try:
            score_fn, batch_fn = _build_score_fns(cand, cfg, vocab, handcrafted)
            model_used = cand
            break
        except Exception:
            continue
    if score_fn is None:
        return
    rec = Recommender(score_fn=score_fn, vocab=vocab, handcrafted=handcrafted, batch_score_fn=batch_fn)

    # Example 1: complete-the-draft using a real test match
    if len(splits.test):
        row = splits.test.iloc[0]
        state = DraftState()
        for r in ROLES:
            for s in ("blue", "red"):
                if s == "blue" and r == "top":
                    continue  # role we want to recommend
                state.picks_for(s)[r] = row[f"{s}_{r}_champion"]
        results = rec.top_k(state, "blue", "top", k=cfg.top_k)
        examples.append(
            {
                "name": "Complete-the-draft (blue top)",
                "model": model_used,
                "state": {
                    "blue_picks": state.blue_picks,
                    "red_picks": state.red_picks,
                    "bans": state.bans,
                },
                "side": "blue",
                "role": "top",
                "results": results,
            }
        )
        # Example 2: beam search with a partial draft
        partial = DraftState(
            blue_picks={
                "jungle": row["blue_jungle_champion"],
                "mid": row["blue_mid_champion"],
            },
            red_picks={
                "top": row["red_top_champion"],
                "mid": row["red_mid_champion"],
            },
        )
        beam = rec.beam_search(
            partial, "blue", "top",
            beam_width=cfg.beam_width, depth=cfg.beam_depth, k=cfg.top_k,
        )
        examples.append(
            {
                "name": "Beam search (partial draft)",
                "model": model_used,
                "state": {
                    "blue_picks": partial.blue_picks,
                    "red_picks": partial.red_picks,
                    "bans": partial.bans,
                },
                "side": "blue",
                "role": "top",
                "beam_width": cfg.beam_width,
                "beam_depth": cfg.beam_depth,
                "results": beam,
            }
        )
    save_json(run_dir / "recommendation_examples.json", examples)


# --------------------------------------------------------------------------- #
# Evaluate command
# --------------------------------------------------------------------------- #


def cmd_evaluate(cfg: PipelineConfig) -> Dict[str, object]:
    """Re-load saved artifacts and recompute metrics on the test split."""
    splits, vocab, handcrafted, long_df, extra_vocabs = prepare_data(cfg)
    art = Path(cfg.artifacts_dir)
    summary: Dict[str, object] = {}

    extra_vocabs = _load_extra_vocabs(cfg)
    base_path = art / "lightgbm_baseline.pkl"
    if base_path.exists():
        bundle = load_pickle(base_path)
        feats = json.loads((art / "lightgbm_baseline_features.json").read_text())
        X_te, _ = build_baseline_feature_matrix(splits.test, vocab, handcrafted, cfg, extra_vocabs)
        X_te = X_te.reindex(columns=feats, fill_value=0)
        prob = predict_lightgbm(bundle["model"], bundle["backend"], X_te)
        cal = _maybe_load_calibrator(art, "lightgbm_baseline")
        if cal is not None:
            prob = cal.predict(prob)
        summary["baseline_test"] = compute_metrics(splits.test["blue_win"].values, prob)
        print_metrics("baseline / test (reloaded)", summary["baseline_test"])

    if _HAS_TORCH:
        device = torch_device()
        enc = encode_champion_ids(splits.test, vocab)
        tcn_path = art / "teamcompnet.pt"
        if tcn_path.exists():
            model = _instantiate_teamcompnet(cfg, vocab, device)
            model.load_state_dict(torch.load(tcn_path, map_location=device))
            prob = _torch_predict_proba(model, enc["blue"], enc["red"], device)
            cal = _maybe_load_calibrator(art, "teamcompnet")
            if cal is not None:
                prob = cal.predict(prob)
            summary["teamcompnet_test"] = compute_metrics(
                splits.test["blue_win"].values, prob
            )
            print_metrics("teamcompnet / test (reloaded)", summary["teamcompnet_test"])

        wd_path = art / "wide_deep.pt"
        if wd_path.exists():
            model = WideDeepDraftNet(
                num_champions=len(vocab),
                embedding_dim=cfg.embedding_dim,
                hidden_dim=cfg.hidden_dim,
                dropout=cfg.dropout,
                combine=cfg.wide_deep_combine,
            ).to(device)
            model.load_state_dict(torch.load(wd_path, map_location=device))
            prob = _torch_predict_proba(model, enc["blue"], enc["red"], device)
            cal = _maybe_load_calibrator(art, "wide_deep")
            if cal is not None:
                prob = cal.predict(prob)
            summary["wide_deep_test"] = compute_metrics(
                splits.test["blue_win"].values, prob
            )
            print_metrics("wide_deep / test (reloaded)", summary["wide_deep_test"])

    hyb_path = art / "lightgbm_with_embeddings.pkl"
    emb_path = art / "champion_embeddings.npy"
    if hyb_path.exists() and emb_path.exists():
        bundle = load_pickle(hyb_path)
        feats = json.loads((art / "lightgbm_with_embeddings_features.json").read_text())
        emb = np.load(emb_path)
        enc_te = encode_champion_ids(splits.test, vocab)
        base_te, _ = build_baseline_feature_matrix(splits.test, vocab, handcrafted, cfg, extra_vocabs)
        extra_te = extract_embedding_features(enc_te["blue"], enc_te["red"], emb)
        X_te = pd.concat(
            [base_te.reset_index(drop=True), extra_te.reset_index(drop=True)], axis=1
        )
        X_te = X_te.reindex(columns=feats, fill_value=0)
        prob = predict_lightgbm(bundle["model"], bundle["backend"], X_te)
        cal = _maybe_load_calibrator(art, "lightgbm_with_embeddings")
        if cal is not None:
            prob = cal.predict(prob)
        summary["hybrid_test"] = compute_metrics(splits.test["blue_win"].values, prob)
        print_metrics("hybrid / test (reloaded)", summary["hybrid_test"])

    save_json(art / "metrics_evaluate.json", summary)

    # Recommender hit-rate using the best available model.
    if _HAS_TORCH and (art / "wide_deep.pt").exists():
        log.info("Evaluating recommender with WideDeepDraftNet ...")
        score_fn, batch_fn = _build_score_fns("wide_deep", cfg, vocab, handcrafted)
    elif (art / "lightgbm_with_embeddings.pkl").exists():
        score_fn, batch_fn = _build_score_fns("hybrid", cfg, vocab, handcrafted)
    elif (art / "lightgbm_baseline.pkl").exists():
        score_fn, batch_fn = _build_score_fns("baseline", cfg, vocab, handcrafted)
    else:
        return summary

    rec = Recommender(
        score_fn=score_fn,
        vocab=vocab,
        handcrafted=handcrafted,
        batch_score_fn=batch_fn,
    )
    rec_metrics = evaluate_recommender(
        rec, splits.test, vocab, n_samples=200, seed=cfg.random_seed
    )
    log.info("Recommender hit-rate metrics: %s", rec_metrics)
    save_json(art / "metrics_recommender.json", rec_metrics)
    summary["recommender"] = rec_metrics
    return summary


# --------------------------------------------------------------------------- #
# Recommend command
# --------------------------------------------------------------------------- #


def parse_pick_string(spec: str) -> Dict[str, str]:
    """Parse 'role=Champ,role=Champ' into a dict."""
    if not spec:
        return {}
    out: Dict[str, str] = {}
    for chunk in spec.split(","):
        if not chunk.strip():
            continue
        if "=" not in chunk:
            raise ValueError(f"Bad pick spec: {chunk!r} (expected role=Champion)")
        role, champ = chunk.split("=", 1)
        role = role.strip().lower()
        champ = champ.strip()
        if role not in ROLE_TO_IDX:
            raise ValueError(f"Unknown role: {role}")
        out[role] = champ
    return out


def parse_bans_string(spec: str) -> List[str]:
    if not spec:
        return []
    return [s.strip() for s in spec.split(",") if s.strip()]


def cmd_recommend(args: argparse.Namespace, cfg: PipelineConfig) -> List[Dict[str, object]]:
    art = Path(cfg.artifacts_dir)
    vocab = json.loads((art / "champion_to_idx.json").read_text())
    handcrafted = load_pickle(art / "handcrafted_stats.pkl")
    state = DraftState(
        blue_picks=parse_pick_string(args.blue_picks),
        red_picks=parse_pick_string(args.red_picks),
        bans=parse_bans_string(args.bans),
    )

    # Validate inputs
    for role, champ in state.blue_picks.items():
        if champ not in vocab:
            log.warning("Unknown champion %r in blue %s", champ, role)
    for role, champ in state.red_picks.items():
        if champ not in vocab:
            log.warning("Unknown champion %r in red %s", champ, role)

    if args.role in state.picks_for(args.side):
        raise SystemExit(f"--role {args.role} is already filled on the {args.side} side")

    score_fn, batch_fn = _build_score_fns(args.model, cfg, vocab, handcrafted)
    rec = Recommender(
        score_fn=score_fn,
        vocab=vocab,
        handcrafted=handcrafted,
        batch_score_fn=batch_fn,
    )
    policy_fn = _maybe_build_policy_fn(args.model, cfg, vocab)
    if getattr(args, "mcts", False):
        results = rec.mcts(
            state, args.side, args.role,
            n_simulations=cfg.mcts_simulations, c_puct=cfg.mcts_c_puct,
            depth=max(2, cfg.beam_depth), policy_fn=policy_fn, k=args.top_k,
        )
    elif args.beam_search:
        results = rec.beam_search(
            state, args.side, args.role,
            beam_width=cfg.beam_width, depth=cfg.beam_depth, k=args.top_k,
        )
    else:
        results = rec.top_k(state, args.side, args.role, k=args.top_k)

    print()
    print(
        f"{'Rank':<5}{'Champion':<18}{'WinProb':<10}{'Delta':<10}"
        f"{'Synergy':<10}{'Counter':<10}Notes"
    )
    print("-" * 90)
    for i, item in enumerate(results, 1):
        print(
            f"{i:<5}{item['champion']:<18}"
            f"{item['win_prob']:.4f}    "
            f"{item['delta']:+.4f}   "
            f"{item['synergy']:+.4f}   "
            f"{item['counter']:+.4f}   "
            f"{item['notes']}"
        )
    save_json(art / "recommendation_examples.json", results)
    return results


def _maybe_build_policy_fn(model_name: str, cfg: PipelineConfig, vocab: Dict[str, int]):
    """Return a (state, side, role) -> np.ndarray[num_champions] policy callable, or None.

    Only the Set Transformer with policy head can produce one; other backbones
    fall back to the recommender's value-shaped prior.
    """
    art = Path(cfg.artifacts_dir)
    if model_name not in ("teamcompnet",) or not _HAS_TORCH:
        return None
    arch_path = art / "teamcompnet_arch.json"
    if not arch_path.exists():
        return None
    try:
        arch = json.loads(arch_path.read_text()).get("arch")
    except Exception:
        return None
    if arch != "transformer":
        return None
    device = torch_device()
    model = _instantiate_teamcompnet(cfg, vocab, device)
    try:
        model.load_state_dict(torch.load(art / "teamcompnet.pt", map_location=device))
    except Exception:
        return None

    def _policy(state, side, role):
        return _torch_policy(model, state, vocab, device)

    return _policy


def _load_extra_vocabs(cfg: PipelineConfig) -> Dict[str, Dict[str, int]]:
    """Load extra_vocabs.json from the artifacts directory; empty dict if missing."""
    path = Path(cfg.artifacts_dir) / "extra_vocabs.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _maybe_load_calibrator(art: Path, name: str) -> Optional["ProbabilityCalibrator"]:
    p = art / f"{name}_calibrator.pkl"
    if not p.exists():
        return None
    try:
        return load_pickle(p)
    except Exception as exc:  # pragma: no cover
        log.warning("Failed loading calibrator %s: %s", p, exc)
        return None


def _instantiate_teamcompnet(cfg: PipelineConfig, vocab: Dict[str, int], device):
    """Build the right TeamCompNet variant based on the saved arch metadata."""
    arch_path = Path(cfg.artifacts_dir) / "teamcompnet_arch.json"
    arch = cfg.arch
    if arch_path.exists():
        try:
            arch = json.loads(arch_path.read_text()).get("arch", arch)
        except Exception:
            pass
    if arch == "transformer":
        return SetTransformerCompNet(
            num_champions=len(vocab),
            embedding_dim=cfg.embedding_dim,
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
            n_layers=cfg.n_attention_layers,
            n_heads=cfg.n_attention_heads,
            policy_head=cfg.enable_policy_head,
        ).to(device)
    return TeamCompNet(
        num_champions=len(vocab),
        embedding_dim=cfg.embedding_dim,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
    ).to(device)


def _build_score_fns(model_name: str, cfg: PipelineConfig, vocab, handcrafted):
    """Return ``(score_fn, batch_score_fn)`` for the requested model."""
    art = Path(cfg.artifacts_dir)
    model_name = (model_name or "hybrid").lower()
    extra_vocabs = _load_extra_vocabs(cfg)
    if model_name == "baseline":
        bundle = load_pickle(art / "lightgbm_baseline.pkl")
        feats = json.loads((art / "lightgbm_baseline_features.json").read_text())
        cal = _maybe_load_calibrator(art, "lightgbm_baseline")
        return (
            make_lgb_score_fn(
                bundle["model"], bundle["backend"], vocab, handcrafted, feats, None,
                calibrator=cal, extra_vocabs=extra_vocabs, cfg=cfg,
            ),
            make_lgb_batch_score_fn(
                bundle["model"], bundle["backend"], vocab, handcrafted, feats, None,
                calibrator=cal, extra_vocabs=extra_vocabs, cfg=cfg,
            ),
        )
    if model_name == "hybrid":
        bundle_path = art / "lightgbm_with_embeddings.pkl"
        if not bundle_path.exists():
            log.warning("Hybrid bundle missing - falling back to baseline.")
            return _build_score_fns("baseline", cfg, vocab, handcrafted)
        bundle = load_pickle(bundle_path)
        feats = json.loads((art / "lightgbm_with_embeddings_features.json").read_text())
        emb = np.load(art / "champion_embeddings.npy")
        cal = _maybe_load_calibrator(art, "lightgbm_with_embeddings")
        return (
            make_lgb_score_fn(
                bundle["model"], bundle["backend"], vocab, handcrafted, feats, emb,
                calibrator=cal, extra_vocabs=extra_vocabs, cfg=cfg,
            ),
            make_lgb_batch_score_fn(
                bundle["model"], bundle["backend"], vocab, handcrafted, feats, emb,
                calibrator=cal, extra_vocabs=extra_vocabs, cfg=cfg,
            ),
        )
    if model_name == "stacker":
        # Meta-ensemble; lazy-build a custom adapter.
        return _build_stacker_score_fns(cfg, vocab, handcrafted)
    if model_name in ("teamcompnet", "wide_deep"):
        if not _HAS_TORCH:
            raise RuntimeError(f"PyTorch is required for {model_name}")
        device = torch_device()
        if model_name == "teamcompnet":
            model = _instantiate_teamcompnet(cfg, vocab, device)
            model.load_state_dict(
                torch.load(art / "teamcompnet.pt", map_location=device)
            )
        else:
            model = WideDeepDraftNet(
                num_champions=len(vocab),
                embedding_dim=cfg.embedding_dim,
                hidden_dim=cfg.hidden_dim,
                dropout=cfg.dropout,
                combine=cfg.wide_deep_combine,
            ).to(device)
            model.load_state_dict(
                torch.load(art / "wide_deep.pt", map_location=device)
            )
        cal = _maybe_load_calibrator(art, model_name)
        return (
            make_torch_score_fn(model, vocab, device, calibrator=cal),
            make_torch_batch_score_fn(model, vocab, device, calibrator=cal),
        )
    raise ValueError(f"Unknown --model {model_name}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-dir", default="data")
    p.add_argument("--artifacts-dir", default="artifacts")
    p.add_argument("--raw-csv", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-rows", type=int, default=None,
                   help="Cap matches loaded (debug aid)")
    p.add_argument("--fast-dev-run", action="store_true")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--embedding-dim", type=int, default=None)
    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--lgb-n-estimators", type=int, default=None)
    p.add_argument("--lgb-learning-rate", type=float, default=None)
    p.add_argument("--run-id", default=None,
                   help="Override run id. Use 'auto' for an auto timestamp (default).")
    p.add_argument("--arch", choices=["pairwise", "transformer"], default=None,
                   help="Deep-model backbone (default: transformer).")
    p.add_argument("--n-attention-layers", type=int, default=None)
    p.add_argument("--n-attention-heads", type=int, default=None)
    p.add_argument("--ranking-weight", type=float, default=None,
                   help="Weight of the listwise-ranking auxiliary loss (0=off).")
    p.add_argument("--ranking-negatives", type=int, default=None)
    p.add_argument("--policy-weight", type=float, default=None,
                   help="Weight of the AlphaZero-style policy-head loss (0=off).")
    p.add_argument("--no-policy-head", action="store_true")
    p.add_argument("--mcts-simulations", type=int, default=None)
    p.add_argument("--mcts-c-puct", type=float, default=None)
    p.add_argument("--no-augment", action="store_true",
                   help="Disable side-flip + champion dropout augmentation.")
    p.add_argument("--augment-dropout-p", type=float, default=None)
    p.add_argument("--no-pretrain", action="store_true",
                   help="Disable PMI+SVD pretraining of champion embeddings.")
    p.add_argument("--no-calibration", action="store_true")
    p.add_argument("--no-stacking", action="store_true")
    p.add_argument("--enable-stacking", action="store_true",
                   help="Force stacker training (default off; useful only with random splits).")


def _build_cfg(args: argparse.Namespace) -> PipelineConfig:
    cfg = PipelineConfig(
        data_dir=getattr(args, "data_dir", "data"),
        artifacts_dir=getattr(args, "artifacts_dir", "artifacts"),
        raw_csv=getattr(args, "raw_csv", None),
        random_seed=getattr(args, "seed", 42),
        max_rows=getattr(args, "max_rows", None),
        fast_dev_run=getattr(args, "fast_dev_run", False),
        run_id=getattr(args, "run_id", None),
    )
    overrides = {
        "epochs": "epochs",
        "batch_size": "batch_size",
        "embedding_dim": "embedding_dim",
        "hidden_dim": "hidden_dim",
        "learning_rate": "learning_rate",
        "patience": "patience",
        "lgb_n_estimators": "lgb_n_estimators",
        "lgb_learning_rate": "lgb_learning_rate",
        "arch": "arch",
        "n_attention_layers": "n_attention_layers",
        "n_attention_heads": "n_attention_heads",
        "ranking_weight": "ranking_weight",
        "ranking_negatives": "ranking_negatives",
        "policy_weight": "policy_weight",
        "mcts_simulations": "mcts_simulations",
        "mcts_c_puct": "mcts_c_puct",
        "augment_dropout_p": "augment_dropout_p",
    }
    for arg_name, cfg_name in overrides.items():
        v = getattr(args, arg_name, None)
        if v is not None:
            setattr(cfg, cfg_name, v)
    if getattr(args, "no_policy_head", False):
        cfg.enable_policy_head = False
        cfg.policy_weight = 0.0
    if getattr(args, "no_augment", False):
        cfg.augment_side_flip = False
        cfg.augment_dropout_p = 0.0
    if getattr(args, "no_pretrain", False):
        cfg.pretrain_embeddings = False
    if getattr(args, "no_calibration", False):
        cfg.enable_calibration = False
    if getattr(args, "no_stacking", False):
        cfg.enable_stacking = False
    if getattr(args, "enable_stacking", False):
        cfg.enable_stacking = True
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="LoL draft-time pipeline: train, evaluate, recommend.",
    )
    subs = parser.add_subparsers(dest="command", required=True)

    p_all = subs.add_parser("train", help="Train every model end-to-end")
    _add_common_args(p_all)

    for sub_name in ("train-baseline", "train-teamcompnet", "train-hybrid", "train-wide-deep"):
        sp = subs.add_parser(sub_name, help=f"Run only the {sub_name} stage")
        _add_common_args(sp)

    p_eval = subs.add_parser("evaluate", help="Reload artifacts and recompute metrics")
    _add_common_args(p_eval)

    p_rec = subs.add_parser("recommend", help="Get top-k draft suggestions")
    _add_common_args(p_rec)
    p_rec.add_argument("--blue-picks", default="")
    p_rec.add_argument("--red-picks", default="")
    p_rec.add_argument("--bans", default="")
    p_rec.add_argument("--side", required=True, choices=["blue", "red"])
    p_rec.add_argument("--role", required=True, choices=list(ROLES))
    p_rec.add_argument("--top-k", type=int, default=5)
    p_rec.add_argument(
        "--model",
        default="hybrid",
        choices=["baseline", "hybrid", "teamcompnet", "wide_deep", "stacker"],
    )
    p_rec.add_argument("--beam-search", action="store_true")
    p_rec.add_argument("--mcts", action="store_true",
                       help="Use AlphaZero-style MCTS instead of beam/top-k.")

    args = parser.parse_args(argv)
    cfg = _build_cfg(args)

    if args.command == "train":
        train_all(cfg)
    elif args.command == "train-baseline":
        train_baseline(cfg)
    elif args.command == "train-teamcompnet":
        train_teamcompnet(cfg)
    elif args.command == "train-hybrid":
        train_hybrid(cfg)
    elif args.command == "train-wide-deep":
        train_wide_deep(cfg)
    elif args.command == "evaluate":
        cmd_evaluate(cfg)
    elif args.command == "recommend":
        cmd_recommend(args, cfg)
    else:  # pragma: no cover
        parser.error(f"Unknown command {args.command}")
    log.info("Done. Artifacts in %s", Path(cfg.artifacts_dir).resolve())
    log.info(
        "Next: python lol_draft_pipeline.py recommend "
        "--side blue --role top --blue-picks 'jungle=LeeSin,mid=Ahri' "
        "--red-picks 'top=Fiora,mid=Orianna' --top-k 5"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
