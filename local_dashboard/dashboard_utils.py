"""
dashboard_utils.py
==================

Helper functions for the local Streamlit dashboard. Every loader is
designed to **fail gracefully** so that a missing artifact never crashes
the page - it simply returns ``None`` / ``[]`` / an empty DataFrame and
the calling tab decides how to render the empty state.

Nothing in this module imports Streamlit so the helpers can also be
unit-tested or reused from a notebook.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Optional dependency: PCA always available via sklearn; UMAP only when
# the user explicitly installed `umap-learn`.
try:
    from sklearn.decomposition import PCA
    _HAS_SKLEARN = True
except Exception:  # pragma: no cover - sklearn is required by main pipeline
    PCA = None  # type: ignore[assignment]
    _HAS_SKLEARN = False

try:
    import umap  # type: ignore[import-not-found]
    _HAS_UMAP = True
except Exception:
    umap = None  # type: ignore[assignment]
    _HAS_UMAP = False


# --------------------------------------------------------------------------- #
# Path discovery
# --------------------------------------------------------------------------- #


@dataclass
class RunRef:
    """Lightweight pointer to a single run directory."""

    run_id: str
    path: Path
    is_latest: bool = False
    started_at: Optional[str] = None
    status: Optional[str] = None
    duration_seconds: Optional[float] = None

    def display(self) -> str:
        suffix = " (latest)" if self.is_latest else ""
        if self.status:
            suffix += f"  [{self.status}]"
        return f"{self.run_id}{suffix}"


def list_runs(artifacts_dir: str | os.PathLike) -> List[RunRef]:
    """Enumerate ``artifacts/runs/<run_id>/`` directories, newest first."""
    runs_root = Path(artifacts_dir) / "runs"
    if not runs_root.is_dir():
        return []
    latest_id = ""
    pointer = runs_root / "_latest_run_id.txt"
    if pointer.is_file():
        latest_id = pointer.read_text().strip()
    refs: List[RunRef] = []
    for p in sorted(runs_root.iterdir()):
        if not p.is_dir():
            continue
        if p.name.startswith("_"):
            continue
        ref = RunRef(run_id=p.name, path=p, is_latest=(p.name == latest_id))
        events = load_events(p)
        if events:
            for ev in events:
                if ev.get("event_type") == "run_started":
                    ref.started_at = ev.get("timestamp")
                if ev.get("event_type") == "run_completed":
                    ref.status = ev.get("status")
                    ref.duration_seconds = ev.get("duration_seconds")
        if ref.status is None:
            ref.status = "running" if events else "unknown"
        refs.append(ref)
    refs.sort(key=lambda r: r.run_id, reverse=True)
    return refs


def latest_run(artifacts_dir: str | os.PathLike) -> Optional[RunRef]:
    runs = list_runs(artifacts_dir)
    return runs[0] if runs else None


# --------------------------------------------------------------------------- #
# Safe IO
# --------------------------------------------------------------------------- #


def safe_read_json(path: str | os.PathLike) -> Optional[Any]:
    """Return parsed JSON or ``None`` if the file is missing / unreadable."""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def safe_read_csv(path: str | os.PathLike) -> pd.DataFrame:
    """Return a DataFrame or an empty one if the file is missing / unreadable."""
    p = Path(path)
    if not p.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


# --------------------------------------------------------------------------- #
# Loaders for run-scoped artifacts
# --------------------------------------------------------------------------- #


def load_events(run_dir: str | os.PathLike) -> List[Dict[str, Any]]:
    """Read ``events.jsonl`` line-by-line; bad lines are skipped."""
    path = Path(run_dir) / "events.jsonl"
    if not path.is_file():
        return []
    out: List[Dict[str, Any]] = []
    try:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return []
    return out


def load_metrics_summary(run_dir: str | os.PathLike) -> Optional[Dict[str, Any]]:
    return safe_read_json(Path(run_dir) / "metrics_summary.json")


def load_model_comparison(run_dir: str | os.PathLike) -> pd.DataFrame:
    return safe_read_csv(Path(run_dir) / "model_comparison.csv")


def load_calibration(run_dir: str | os.PathLike) -> pd.DataFrame:
    return safe_read_csv(Path(run_dir) / "calibration.csv")


def load_predictions(run_dir: str | os.PathLike) -> pd.DataFrame:
    return safe_read_csv(Path(run_dir) / "predictions_test.csv")


def load_feature_importance(run_dir: str | os.PathLike) -> pd.DataFrame:
    return safe_read_csv(Path(run_dir) / "feature_importance.csv")


def load_embeddings(run_dir: str | os.PathLike) -> pd.DataFrame:
    return safe_read_csv(Path(run_dir) / "embedding_champions.csv")


def load_recommendation_examples(run_dir: str | os.PathLike) -> List[Dict[str, Any]]:
    payload = safe_read_json(Path(run_dir) / "recommendation_examples.json")
    return payload if isinstance(payload, list) else []


def load_leakage_audit(run_dir: str | os.PathLike) -> Optional[Dict[str, Any]]:
    return safe_read_json(Path(run_dir) / "leakage_audit.json")


def load_schema_report(run_dir: str | os.PathLike) -> Optional[Dict[str, Any]]:
    return safe_read_json(Path(run_dir) / "schema_report.json")


def load_confusion_matrices(run_dir: str | os.PathLike) -> Optional[Dict[str, Any]]:
    return safe_read_json(Path(run_dir) / "confusion_matrices.json")


def load_run_config(run_dir: str | os.PathLike) -> Optional[Dict[str, Any]]:
    return safe_read_json(Path(run_dir) / "config.json")


def load_champion_vocab(run_dir: str | os.PathLike) -> Dict[str, int]:
    payload = safe_read_json(Path(run_dir) / "champion_to_idx.json")
    if isinstance(payload, dict):
        return payload
    # Fall back to top-level artifacts
    parent = Path(run_dir).parent.parent
    payload = safe_read_json(parent / "champion_to_idx.json")
    return payload or {}


# --------------------------------------------------------------------------- #
# Run status
# --------------------------------------------------------------------------- #


def infer_run_status(events: Sequence[Dict[str, Any]]) -> str:
    """Heuristically derive run status from the event stream."""
    if not events:
        return "no_events"
    last = events[-1]
    if last.get("event_type") == "run_completed":
        return last.get("status", "completed")
    if last.get("event_type") == "error":
        return "error"
    return "running"


def run_duration_seconds(events: Sequence[Dict[str, Any]]) -> Optional[float]:
    started = next((e for e in events if e.get("event_type") == "run_started"), None)
    completed = next(
        (e for e in reversed(events) if e.get("event_type") == "run_completed"), None
    )
    if started and completed:
        return completed.get("duration_seconds")
    return None


# --------------------------------------------------------------------------- #
# Metrics from predictions
# --------------------------------------------------------------------------- #


def _safe_metric(fn, *args, **kwargs):
    try:
        return float(fn(*args, **kwargs))
    except Exception:
        return float("nan")


def compute_metrics_from_predictions(
    df: pd.DataFrame, threshold: float = 0.5
) -> Dict[str, Any]:
    """Recompute model-level metrics from a long predictions table.

    The DataFrame is expected to carry ``model_name``, ``y_true``, ``y_prob``.
    Rows without the expected columns are skipped.
    """
    if df.empty or not {"model_name", "y_true", "y_prob"}.issubset(df.columns):
        return {}
    from sklearn.metrics import (
        accuracy_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        log_loss,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )

    out: Dict[str, Any] = {}
    for name, sub in df.groupby("model_name"):
        y_true = sub["y_true"].values.astype(int)
        y_prob = np.clip(sub["y_prob"].values.astype(float), 1e-6, 1 - 1e-6)
        y_pred = (y_prob >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred).tolist() if len(y_true) else None
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
        except Exception:
            fpr, tpr = [], []
        try:
            prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_prob)
        except Exception:
            prec_curve, rec_curve = [], []
        out[name] = {
            "n": int(len(sub)),
            "accuracy": _safe_metric(accuracy_score, y_true, y_pred),
            "precision": _safe_metric(precision_score, y_true, y_pred, zero_division=0),
            "recall": _safe_metric(recall_score, y_true, y_pred, zero_division=0),
            "f1": _safe_metric(f1_score, y_true, y_pred, zero_division=0),
            "roc_auc": _safe_metric(roc_auc_score, y_true, y_prob)
            if len(np.unique(y_true)) > 1
            else float("nan"),
            "log_loss": _safe_metric(log_loss, y_true, y_prob, labels=[0, 1]),
            "brier_score": _safe_metric(brier_score_loss, y_true, y_prob),
            "confusion_matrix": cm,
            "roc_fpr": [float(v) for v in fpr],
            "roc_tpr": [float(v) for v in tpr],
            "pr_precision": [float(v) for v in prec_curve],
            "pr_recall": [float(v) for v in rec_curve],
        }
    return out


def compute_calibration_from_predictions(
    df: pd.DataFrame, n_bins: int = 10
) -> pd.DataFrame:
    """Re-derive a calibration table when ``calibration.csv`` is missing."""
    if df.empty or not {"model_name", "y_true", "y_prob"}.issubset(df.columns):
        return pd.DataFrame()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for name, sub in df.groupby("model_name"):
        y_prob = sub["y_prob"].values.astype(float)
        y_true = sub["y_true"].values.astype(int)
        for lo, hi in zip(edges[:-1], edges[1:]):
            if hi >= 1.0:
                mask = (y_prob >= lo) & (y_prob <= hi)
            else:
                mask = (y_prob >= lo) & (y_prob < hi)
            rows.append(
                {
                    "model": name,
                    "bucket": f"[{lo:.2f},{hi:.2f})",
                    "n": int(mask.sum()),
                    "mean_pred": float(y_prob[mask].mean()) if mask.any() else float("nan"),
                    "empirical_rate": float(y_true[mask].mean()) if mask.any() else float("nan"),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Embedding helpers
# --------------------------------------------------------------------------- #


def _dim_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.startswith("dim_") or re.fullmatch(r"d\d+", c or "")]
    cols.sort(key=lambda c: int(re.findall(r"\d+", c)[0]))
    return cols


def compute_pca_embeddings(
    df: pd.DataFrame, n_components: int = 2, method: str = "pca"
) -> pd.DataFrame:
    """Project the embedding matrix to 2/3 components for plotting."""
    if df.empty or not _HAS_SKLEARN:
        return pd.DataFrame()
    dims = _dim_columns(df)
    if not dims:
        return pd.DataFrame()
    matrix = df[dims].values.astype(float)
    method = (method or "pca").lower()
    if method == "umap" and _HAS_UMAP and len(matrix) > n_components:
        try:
            reducer = umap.UMAP(n_components=n_components, random_state=42, n_neighbors=15)
            coords = reducer.fit_transform(matrix)
        except Exception:
            method = "pca"
            coords = None
    else:
        coords = None
    if coords is None:
        coords = PCA(n_components=min(n_components, matrix.shape[1])).fit_transform(matrix)
    out = df.copy()
    for j in range(coords.shape[1]):
        out[f"component_{j+1}"] = coords[:, j]
    return out


def find_nearest_champions(
    df: pd.DataFrame, champion: str, top_k: int = 10
) -> pd.DataFrame:
    """Return the ``top_k`` champions most similar (cosine) to ``champion``."""
    if df.empty or "champion" not in df.columns:
        return pd.DataFrame()
    dims = _dim_columns(df)
    if not dims:
        return pd.DataFrame()
    matrix = df[dims].values.astype(float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    matrix = matrix / norms
    if champion not in df["champion"].values:
        return pd.DataFrame()
    target_idx = int(df.index[df["champion"] == champion][0])
    sims = matrix @ matrix[target_idx]
    order = np.argsort(-sims)
    rows = []
    for idx in order:
        if df.iloc[idx]["champion"] == champion:
            continue
        rows.append({"champion": df.iloc[idx]["champion"], "cosine_sim": float(sims[idx])})
        if len(rows) >= top_k:
            break
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# UI helpers (tiny, reused across tabs)
# --------------------------------------------------------------------------- #


def format_warning_box(message: str) -> str:
    return f":warning: **Warning** — {message}"


def format_info_box(message: str) -> str:
    return f":information_source: {message}"


def best_value_in_column(df: pd.DataFrame, column: str, higher_is_better: bool = True) -> Optional[float]:
    """Return the best numeric value of ``column`` (NaN-safe)."""
    if df.empty or column not in df.columns:
        return None
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.max() if higher_is_better else series.min())


def highlight_best(
    df: pd.DataFrame,
    higher_is_better: Sequence[str] = ("accuracy", "f1", "auc", "roc_auc", "precision", "recall"),
    lower_is_better: Sequence[str] = ("log_loss", "brier", "brier_score"),
):
    """Return a Styler that bolds the best value per metric column."""
    higher = {c for c in higher_is_better if c in df.columns}
    lower = {c for c in lower_is_better if c in df.columns}
    if not (higher or lower):
        return df.style

    def _row_style(col):
        if col.name in higher:
            v = pd.to_numeric(col, errors="coerce")
            best = v.max()
            return ["font-weight: bold; background-color: rgba(46, 204, 113, 0.18)"
                    if val == best else "" for val in v]
        if col.name in lower:
            v = pd.to_numeric(col, errors="coerce")
            best = v.min()
            return ["font-weight: bold; background-color: rgba(46, 204, 113, 0.18)"
                    if val == best else "" for val in v]
        return ["" for _ in col]

    return df.style.apply(_row_style)


# --------------------------------------------------------------------------- #
# Roles + champions util
# --------------------------------------------------------------------------- #

ROLES: Tuple[str, ...] = ("top", "jungle", "mid", "adc", "support")


def parse_pick_dict(spec: str) -> Dict[str, str]:
    """Parse 'role=Champion,role=Champion' -> dict; ignore empties."""
    out: Dict[str, str] = {}
    for chunk in (spec or "").split(","):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        role, champ = chunk.split("=", 1)
        out[role.strip().lower()] = champ.strip()
    return out


def parse_bans_list(spec: str) -> List[str]:
    return [s.strip() for s in (spec or "").split(",") if s.strip()]


# --------------------------------------------------------------------------- #
# File browser helpers
# --------------------------------------------------------------------------- #


def list_run_files(run_dir: str | os.PathLike) -> pd.DataFrame:
    """Tabulate every file under ``run_dir`` for the artifacts browser."""
    p = Path(run_dir)
    if not p.is_dir():
        return pd.DataFrame()
    rows = []
    for entry in sorted(p.iterdir()):
        if entry.is_dir():
            continue
        try:
            stat = entry.stat()
        except OSError:
            continue
        rows.append(
            {
                "name": entry.name,
                "size_kb": round(stat.st_size / 1024.0, 2),
                "modified": pd.to_datetime(stat.st_mtime, unit="s").isoformat(timespec="seconds"),
                "extension": entry.suffix.lower().lstrip("."),
            }
        )
    return pd.DataFrame(rows)


def preview_text_file(path: str | os.PathLike, max_chars: int = 8000) -> str:
    """Return a truncated preview of a text-like file."""
    p = Path(path)
    if not p.is_file():
        return ""
    try:
        text = p.read_text(errors="replace")
    except Exception:
        return ""
    if len(text) > max_chars:
        return text[:max_chars] + f"\n\n... ({len(text) - max_chars} chars truncated)"
    return text
