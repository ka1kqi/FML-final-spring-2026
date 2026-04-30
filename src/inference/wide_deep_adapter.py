"""Inference adapter for the Wide & Deep draft win-probability model.

Loads artifacts from a directory:
    wide_deep.pt              — torch state_dict (required)
    wide_deep_vocab.json      — champion <-> id mapping with special tokens (required)
    wide_deep_config.json     — model hyperparams; must match the .pt (required)
    wide_deep_calibrator.pkl  — optional sklearn IsotonicRegression for prob calibration

If any required artifact is missing or corrupt, ``self.available`` is False and
``predict_*`` methods return None. Server code uses this flag to decide whether
to call the model or fall back to the legacy heuristic. A missing or corrupt
calibrator is non-fatal — the adapter falls back to raw sigmoid output.
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from src.models.wide_deep import (
    LEGACY_ARCH,
    PAD_TOKEN,
    UNK_TOKEN,
    ROLE_ORDER,
    V2_ARCH,
    WideDeepDraftNet,
)

logger = logging.getLogger(__name__)

_PROB_CLIP_LO = 1e-6
_PROB_CLIP_HI = 1.0 - 1e-6


class WideDeepDraftAdapter:
    """Wraps a trained Wide & Deep model and provides simple predict methods."""

    def __init__(self, model_dir: Path | str) -> None:
        self.model_dir = Path(model_dir)
        self._available = False
        self._model: WideDeepDraftNet | None = None
        self._calibrator = None  # optional IsotonicRegression
        self._champion_to_id: dict[str, int] = {}
        self._role_order: list[str] = list(ROLE_ORDER)
        self._config: dict = {}

        try:
            self._load()
            self._available = True
        except (FileNotFoundError, json.JSONDecodeError, KeyError, RuntimeError, ValueError) as exc:
            logger.warning("Wide & Deep artifacts unavailable (%s); using fallback.", exc)
            self._available = False

    # ----- public API -----

    @property
    def available(self) -> bool:
        return self._available

    @property
    def model_version(self) -> str:
        return self._config.get("model_name", "unavailable")

    def predict_blue_win_prob(
        self,
        blue_picks: Mapping[str, str | None] | Sequence[str | None],
        red_picks: Mapping[str, str | None] | Sequence[str | None],
    ) -> float | None:
        """Return blue-side win probability in [eps, 1-eps], or None if unavailable."""
        if not self._available or self._model is None:
            return None
        blue_ids = self._encode(blue_picks).unsqueeze(0)
        red_ids = self._encode(red_picks).unsqueeze(0)
        with torch.no_grad():
            logit = self._model(blue_ids, red_ids)
            prob = torch.sigmoid(logit).item()
        return float(self._clip(self._calibrate(prob)))

    def predict_blue_win_prob_batch(
        self,
        blue_picks_list: Sequence[Mapping[str, str | None] | Sequence[str | None]],
        red_picks_list: Sequence[Mapping[str, str | None] | Sequence[str | None]],
    ) -> list[float | None]:
        """Vectorised calibrated-probability version of ``predict_blue_win_prob``.

        Returns a list of probabilities in [eps, 1-eps] (or all None if the
        adapter is unavailable). For ranking, prefer
        ``predict_blue_win_prob_batch_with_raw`` which exposes the pre-calibration
        score so ties from the isotonic plateau don't collapse the order.
        """
        out = self.predict_blue_win_prob_batch_with_raw(blue_picks_list, red_picks_list)
        return [None if pair is None else pair[0] for pair in out]

    def predict_blue_win_prob_batch_with_raw(
        self,
        blue_picks_list: Sequence[Mapping[str, str | None] | Sequence[str | None]],
        red_picks_list: Sequence[Mapping[str, str | None] | Sequence[str | None]],
    ) -> list[tuple[float, float] | None]:
        """Return (calibrated_prob, raw_prob) per pair, or None if unavailable.

        Raw is the pre-calibration sigmoid output; useful as a tie-breaker for
        ranking because IsotonicRegression collapses many input values onto the
        same plateau (and for empty/sparse drafts almost every candidate ends
        up in the same plateau, producing identical calibrated probs).
        """
        if not self._available or self._model is None:
            return [None] * len(blue_picks_list)
        if len(blue_picks_list) != len(red_picks_list):
            raise ValueError("blue_picks_list and red_picks_list must be same length")
        if not blue_picks_list:
            return []
        blue_ids = torch.stack([self._encode(b) for b in blue_picks_list])
        red_ids = torch.stack([self._encode(r) for r in red_picks_list])
        with torch.no_grad():
            logits = self._model(blue_ids, red_ids)
            raw_probs = torch.sigmoid(logits).cpu().numpy()
        cal_probs = raw_probs
        if self._calibrator is not None:
            try:
                cal_probs = np.asarray(self._calibrator.predict(raw_probs), dtype=np.float64)
            except (ValueError, AttributeError) as exc:  # noqa: BLE001
                logger.warning("calibrator.predict failed (%s); using raw probs", exc)
                cal_probs = raw_probs
        cal_probs = np.clip(cal_probs, _PROB_CLIP_LO, _PROB_CLIP_HI)
        raw_probs = np.clip(raw_probs, _PROB_CLIP_LO, _PROB_CLIP_HI)
        return [(float(c), float(r)) for c, r in zip(cal_probs, raw_probs)]

    def predict_side_win_prob(
        self,
        blue_picks: Mapping[str, str | None] | Sequence[str | None],
        red_picks: Mapping[str, str | None] | Sequence[str | None],
        side: str,
    ) -> float | None:
        p = self.predict_blue_win_prob(blue_picks, red_picks)
        if p is None:
            return None
        if side.lower() == "blue":
            return p
        if side.lower() == "red":
            return float(self._clip(1.0 - p))
        raise ValueError(f"side must be 'blue' or 'red', got {side!r}")

    # ----- internals -----

    def _calibrate(self, prob: float) -> float:
        if self._calibrator is None:
            return prob
        try:
            calibrated = float(self._calibrator.predict(np.asarray([prob], dtype=np.float64))[0])
        except (ValueError, AttributeError) as exc:  # noqa: BLE001
            logger.warning("calibrator.predict failed (%s); using raw prob", exc)
            return prob
        return calibrated

    @staticmethod
    def _clip(prob: float) -> float:
        if prob < _PROB_CLIP_LO:
            return _PROB_CLIP_LO
        if prob > _PROB_CLIP_HI:
            return _PROB_CLIP_HI
        return prob

    def _load(self) -> None:
        vocab_path = self.model_dir / "wide_deep_vocab.json"
        config_path = self.model_dir / "wide_deep_config.json"
        weights_path = self.model_dir / "wide_deep.pt"
        calibrator_path = self.model_dir / "wide_deep_calibrator.pkl"

        with open(vocab_path) as f:
            vocab = json.load(f)
        with open(config_path) as f:
            self._config = json.load(f)

        self._champion_to_id = {str(k): int(v) for k, v in vocab["champion_to_id"].items()}
        self._role_order = list(vocab.get("role_order", ROLE_ORDER))

        # Decide architecture: explicit field wins; absence implies legacy
        # (back-compat with v1 artifacts that never wrote this field).
        architecture = self._config.get("architecture", LEGACY_ARCH)
        if architecture not in (LEGACY_ARCH, V2_ARCH):
            raise ValueError(f"unknown architecture in config: {architecture!r}")

        num_champs = max(self._champion_to_id.values()) + 1
        self._model = WideDeepDraftNet(
            num_champions=num_champs,
            embedding_dim=int(self._config.get("embedding_dim", 32)),
            hidden_dims=tuple(self._config.get("hidden_dims", (128, 64))),
            hidden_dim=int(self._config.get("hidden_dim", 128)),
            dropout=float(self._config.get("dropout", 0.2)),
            architecture=architecture,
            combine=str(self._config.get("combine", "sum")),
        )
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        self._model.load_state_dict(state)
        self._model.eval()

        # Calibrator is best-effort — never fail _load over it.
        if calibrator_path.exists():
            try:
                with open(calibrator_path, "rb") as f:
                    self._calibrator = pickle.load(f)
                logger.info("Loaded W&D probability calibrator from %s", calibrator_path.name)
            except (pickle.UnpicklingError, EOFError, AttributeError, ImportError, ModuleNotFoundError) as exc:
                logger.warning(
                    "Calibrator at %s is unreadable (%s); falling back to raw sigmoid.",
                    calibrator_path.name, exc,
                )
                self._calibrator = None

    def _encode(self, picks) -> torch.Tensor:
        names = self._normalize_to_ordered_list(picks)
        pad_id = self._champion_to_id[PAD_TOKEN]
        unk_id = self._champion_to_id[UNK_TOKEN]
        ids = []
        for n in names:
            if n is None:
                ids.append(pad_id)
            else:
                ids.append(self._champion_to_id.get(n, unk_id))
        return torch.tensor(ids, dtype=torch.long)

    def _normalize_to_ordered_list(self, picks) -> list[str | None]:
        if isinstance(picks, Mapping):
            return [picks.get(role) for role in self._role_order]
        if isinstance(picks, Iterable):
            seq = list(picks)
            if len(seq) > 5:
                raise ValueError(f"Expected up to 5 picks, got {len(seq)}")
            seq = seq + [None] * (5 - len(seq))
            return seq
        raise TypeError(f"picks must be dict or list, got {type(picks)}")
