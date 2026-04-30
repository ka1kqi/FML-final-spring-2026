"""Inference adapter for the Wide & Deep draft win-probability model.

Loads artifacts from a directory:
    wide_deep.pt          — torch state_dict
    wide_deep_vocab.json  — champion <-> id mapping with special tokens
    wide_deep_config.json — model hyperparams (must match training)

If any artifact is missing or corrupt, ``self.available`` is False and
``predict_*`` methods return None. Server code uses this flag to decide
whether to call the model or fall back to the legacy heuristic.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch

from src.models.wide_deep import (
    PAD_TOKEN,
    UNK_TOKEN,
    ROLE_ORDER,
    WideDeepDraftNet,
)

logger = logging.getLogger(__name__)


class WideDeepDraftAdapter:
    """Wraps a trained Wide & Deep model and provides simple predict methods."""

    def __init__(self, model_dir: Path | str) -> None:
        self.model_dir = Path(model_dir)
        self._available = False
        self._model: WideDeepDraftNet | None = None
        self._champion_to_id: dict[str, int] = {}
        self._role_order: list[str] = list(ROLE_ORDER)
        self._config: dict = {}

        try:
            self._load()
            self._available = True
        except (FileNotFoundError, json.JSONDecodeError, KeyError, RuntimeError) as exc:
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
        """Return blue-side win probability in [0, 1], or None if unavailable."""
        if not self._available or self._model is None:
            return None
        blue_ids = self._encode(blue_picks)
        red_ids = self._encode(red_picks)
        with torch.no_grad():
            logit = self._model(blue_ids.unsqueeze(0), red_ids.unsqueeze(0))
            prob = torch.sigmoid(logit).item()
        return float(prob)

    def predict_side_win_prob(
        self,
        blue_picks,
        red_picks,
        side: str,
    ) -> float | None:
        p = self.predict_blue_win_prob(blue_picks, red_picks)
        if p is None:
            return None
        if side.lower() == "blue":
            return p
        if side.lower() == "red":
            return 1.0 - p
        raise ValueError(f"side must be 'blue' or 'red', got {side!r}")

    # ----- internals -----

    def _load(self) -> None:
        vocab_path = self.model_dir / "wide_deep_vocab.json"
        config_path = self.model_dir / "wide_deep_config.json"
        weights_path = self.model_dir / "wide_deep.pt"

        with open(vocab_path) as f:
            vocab = json.load(f)
        with open(config_path) as f:
            self._config = json.load(f)

        self._champion_to_id = {str(k): int(v) for k, v in vocab["champion_to_id"].items()}
        self._role_order = list(vocab.get("role_order", ROLE_ORDER))

        num_champs = max(self._champion_to_id.values()) + 1
        self._model = WideDeepDraftNet(
            num_champions=num_champs,
            embedding_dim=int(self._config.get("embedding_dim", 32)),
            hidden_dims=tuple(self._config.get("hidden_dims", (128, 64))),
            dropout=float(self._config.get("dropout", 0.2)),
        )
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        self._model.load_state_dict(state)
        self._model.eval()

    def _encode(self, picks) -> torch.Tensor:
        """Convert dict[role->name] or list[name] into a [5] LongTensor of ids."""
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
        """Accepts dict[role->name] or list[name | None] and returns 5-long list ordered by role_order."""
        if isinstance(picks, Mapping):
            return [picks.get(role) for role in self._role_order]
        if isinstance(picks, Iterable):
            seq = list(picks)
            if len(seq) > 5:
                raise ValueError(f"Expected up to 5 picks, got {len(seq)}")
            seq = seq + [None] * (5 - len(seq))
            return seq
        raise TypeError(f"picks must be dict or list, got {type(picks)}")
