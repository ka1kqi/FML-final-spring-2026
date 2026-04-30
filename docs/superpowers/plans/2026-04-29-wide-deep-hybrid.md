# Wide & Deep Hybrid Win Probability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Wide & Deep neural network that predicts blue-side win probability from draft composition, layered on top of the existing Champion2Vec + HistGradientBoostingRegressor pipeline. Replace the heuristic `win_prob = 0.5 + (score - 50) * 0.01` with the new model when available, while preserving full backward compatibility (graceful fallback if the W&D artifact is missing).

**Architecture:** Two-stage prediction: (1) existing pipeline produces `performance_score` per candidate (kept as-is), (2) W&D produces `blue_win_prob` from full 10-champion composition. Hybrid ranker combines both. Server.py loads both adapters at startup; W&D adapter degrades gracefully when artifacts are missing. All endpoints stay backward-compatible — `score` and `win_prob` fields preserved, new fields additive.

**Tech Stack:** PyTorch (model), Flask (server, unchanged), scikit-learn (existing regressor, unchanged), pytest (new test infra).

**Branch:** `feature/draft-role-swap` at HEAD `96e892b`. Work directly on this branch.

**Repo path:** `/Users/bao/Documents/FML/.claude/worktrees/zen-williams-5fea65/`

---

## Critical Constraints (Read First)

1. **App must boot without `wide_deep.pt`.** All adapter init must wrap loading in try/except; on failure set `self.available = False` and continue.
2. **Existing response fields preserved** — `champion`, `score`, `win_prob` for `/api/recommend`; `blue_win_prob`, `red_win_prob`, `blue_score`, `red_score` for `/api/evaluate`.
3. **Frontend (`app/static/app.js`) only gets non-breaking additions.** Do NOT remove or rename fields it reads.
4. **Server.py stays thin** — model logic lives in `src/`. Routes call adapters and assemble JSON.
5. **Champion2Vec stays.** It is the rubric-required custom algorithm. Do not replace.
6. **Training is optional.** Demo runs on pretrained artifacts. The W&D training script is provided so the user can produce `wide_deep.pt` when training data is available, but the demo must work even with the artifact missing.

---

## File Structure

**New files:**
- `src/models/wide_deep.py` — `WideDeepDraftNet` PyTorch module + helper utilities
- `src/models/train_wide_deep.py` — training pipeline → artifacts
- `src/inference/wide_deep_adapter.py` — `WideDeepDraftAdapter` class with graceful loading
- `tests/__init__.py` — empty
- `tests/test_wide_deep_adapter.py` — adapter unit tests
- `tests/test_app_routes.py` — Flask route smoke tests
- `tests/test_recommend_hybrid.py` — recommender hybrid logic tests
- `data/processed/draft_models/model_manifest.json` — manifest of all artifacts (the file `wide_deep.pt` itself is NOT created in this plan; created later by running the training script)
- `requirements-dev.txt` — moved training/dev deps
- `.gitignore` additions (only if not already covered)

**Modified files:**
- `src/inference/draft_recommender.py` — add `recommend_hybrid()` (don't remove `recommend_at_step`)
- `app/server.py` — load adapter at startup; modify `/api/recommend` and `/api/evaluate` to use adapter with fallback
- `app/static/app.js` — minimal, non-breaking display additions only (warnings + prob_source label)
- `requirements.txt` — slim to demo essentials
- `README.md` — dual-model story, remove overclaim language

**Untouched (do not edit):**
- `src/models/train_embeddings.py`
- `src/models/draft_classifier.py`
- `src/features/synergy_features.py`
- `src/features/team_comp.py`, `encoding.py`, `game_stats.py`

---

## Task 1: Foundation — Requirements Split, Manifest, .gitignore

**Files:**
- Create: `requirements-dev.txt`
- Modify: `requirements.txt`
- Create: `data/processed/draft_models/model_manifest.json`
- Modify (only if needed): `.gitignore`

- [ ] **Step 1.1: Inspect current `.gitignore`**

```bash
cat /Users/bao/Documents/FML/.claude/worktrees/zen-williams-5fea65/.gitignore | grep -E "wide_deep|score_stats|test"
```

Expected: probably empty / no matches. If the file already ignores `data/raw/*.csv` etc., leave it. If `wide_deep.pt` and `tests/__pycache__` aren't covered, add minimal lines below.

- [ ] **Step 1.2: Add (only the missing lines) to `.gitignore`**

```
# python tests
__pycache__/
*.pyc
.pytest_cache/

# trained artifacts (large; produced by training scripts, not committed)
data/processed/draft_models/wide_deep.pt
```

Skip lines that already exist.

- [ ] **Step 1.3: Slim `requirements.txt` to demo essentials**

Replace the entire file content with:

```
flask
numpy
pandas
scikit-learn
joblib
python-dotenv
torch
```

- [ ] **Step 1.4: Create `requirements-dev.txt`**

```
xgboost
lightgbm
streamlit
matplotlib
seaborn
riotwatcher
tqdm
pyyaml
pytest
```

- [ ] **Step 1.5: Create `data/processed/draft_models/model_manifest.json`**

```json
{
  "project_version": "draft_role_swap_hybrid_v1",
  "demo_retraining_required": false,
  "models": {
    "champion2vec": {
      "type": "custom pure-NumPy matrix factorization",
      "purpose": "learn champion embeddings from synergy and matchup matrices",
      "artifact": "champion2vec.npz"
    },
    "performance_regressor": {
      "type": "HistGradientBoostingRegressor",
      "purpose": "predict candidate champion performance score",
      "artifact": "draft_model.joblib"
    },
    "wide_deep": {
      "type": "Wide & Deep neural network",
      "purpose": "predict blue-side win probability from draft composition",
      "artifact": "wide_deep.pt"
    }
  },
  "api_contract": {
    "backward_compatible": true,
    "unchanged_endpoints": [
      "/api/champions",
      "/api/recommend",
      "/api/evaluate"
    ]
  }
}
```

- [ ] **Step 1.6: Verify pip resolves the new requirements (smoke check)**

```bash
cd /Users/bao/Documents/FML/.claude/worktrees/zen-williams-5fea65 && python3 -m pip install --dry-run -r requirements.txt 2>&1 | tail -5
```

Expected: no resolution errors. (Actual install not required at this step.)

- [ ] **Step 1.7: Commit**

```bash
cd /Users/bao/Documents/FML/.claude/worktrees/zen-williams-5fea65 && git add requirements.txt requirements-dev.txt data/processed/draft_models/model_manifest.json .gitignore && git commit -m "chore: split requirements, add model manifest, gitignore tweaks"
```

---

## Task 2: Wide & Deep Model Architecture (TDD)

**Files:**
- Create: `src/models/wide_deep.py`
- Create: `tests/__init__.py` (empty)
- Test: `tests/test_wide_deep_model.py`

**Design:**
- `ROLE_ORDER = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]` (single source of truth, exported)
- `PAD_TOKEN = "__PAD__"`, `UNK_TOKEN = "__UNK__"` (assigned vocab ids 0 and 1)
- `WideDeepDraftNet(num_champions, embedding_dim=32, hidden_dims=(128, 64), dropout=0.2)`
- Forward signature: `forward(blue_ids, red_ids) -> logits` with `blue_ids.shape == red_ids.shape == [B, 5]`
- Inside: lookup embeddings → mean-pool blue, mean-pool red, concat with role-slot wide branch → MLP → 1 logit
- Wide branch: `[B, 10 * num_champions]` one-hot per (slot, side); use `nn.Linear(2*5*num_champions, 1, bias=False)` for the wide path. Implement as sparse-friendly path that only sums embedding rows of present champions to avoid building the full one-hot.

- [ ] **Step 2.1: Create `tests/__init__.py`**

```bash
cd /Users/bao/Documents/FML/.claude/worktrees/zen-williams-5fea65 && mkdir -p tests && touch tests/__init__.py
```

- [ ] **Step 2.2: Write failing test `tests/test_wide_deep_model.py`**

```python
"""Unit tests for the Wide & Deep architecture."""
import torch

from src.models.wide_deep import (
    WideDeepDraftNet,
    ROLE_ORDER,
    PAD_TOKEN,
    UNK_TOKEN,
)


def test_role_order_is_canonical():
    assert ROLE_ORDER == ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]


def test_special_tokens():
    assert PAD_TOKEN == "__PAD__"
    assert UNK_TOKEN == "__UNK__"


def test_forward_shape_full_draft():
    net = WideDeepDraftNet(num_champions=170, embedding_dim=8, hidden_dims=(16, 8))
    blue = torch.randint(2, 170, (4, 5))
    red = torch.randint(2, 170, (4, 5))
    logits = net(blue, red)
    assert logits.shape == (4,), logits.shape


def test_forward_with_pad_does_not_crash():
    net = WideDeepDraftNet(num_champions=170, embedding_dim=8, hidden_dims=(16, 8))
    # PAD id is 0 by convention; place pads in some slots
    blue = torch.tensor([[0, 5, 0, 7, 0]])
    red = torch.tensor([[3, 0, 4, 0, 6]])
    logits = net(blue, red)
    assert logits.shape == (1,)
    assert torch.isfinite(logits).all()


def test_sigmoid_in_range():
    net = WideDeepDraftNet(num_champions=170, embedding_dim=8, hidden_dims=(16, 8))
    blue = torch.randint(0, 170, (3, 5))
    red = torch.randint(0, 170, (3, 5))
    probs = torch.sigmoid(net(blue, red))
    assert (probs >= 0).all() and (probs <= 1).all()
```

- [ ] **Step 2.3: Run tests, confirm they fail**

```bash
cd /Users/bao/Documents/FML/.claude/worktrees/zen-williams-5fea65 && python3 -m pytest tests/test_wide_deep_model.py -v
```

Expected: FAIL — `ModuleNotFoundError: src.models.wide_deep`.

- [ ] **Step 2.4: Implement `src/models/wide_deep.py`**

```python
"""Wide & Deep neural network for draft-level blue-side win probability.

Inputs:
    blue_ids: LongTensor [B, 5] — champion ids in role order (TOP, JG, MID, BOT, SUP)
    red_ids:  LongTensor [B, 5] — same shape, for the red team

Output:
    logits: FloatTensor [B] — pre-sigmoid score; apply torch.sigmoid for blue_win_prob.

Special tokens:
    id 0 -> "__PAD__" (unfilled slot)
    id 1 -> "__UNK__" (unknown champion at inference)
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

ROLE_ORDER: list[str] = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
PAD_TOKEN: str = "__PAD__"
UNK_TOKEN: str = "__UNK__"
PAD_ID: int = 0
UNK_ID: int = 1


class WideDeepDraftNet(nn.Module):
    """Wide & Deep architecture for predicting blue-side win probability."""

    def __init__(
        self,
        num_champions: int,
        embedding_dim: int = 32,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_champions = num_champions
        self.embedding_dim = embedding_dim

        # Shared embedding for the deep branch
        self.embedding = nn.Embedding(num_champions, embedding_dim, padding_idx=PAD_ID)

        # Wide branch: per-slot, per-side linear contribution per champion id.
        # Implemented as 10 slot-specific weight vectors over the full champion vocab.
        # Total params: 10 * num_champions (1 logit per (slot, champion)).
        self.wide_weights = nn.Embedding(num_champions, 10, padding_idx=PAD_ID)
        nn.init.zeros_(self.wide_weights.weight)

        # Deep branch: blue mean + red mean + per-slot blue + per-slot red embeddings
        deep_in = embedding_dim * 2 + embedding_dim * 10
        layers: list[nn.Module] = []
        prev = deep_in
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.deep = nn.Sequential(*layers)

    def forward(self, blue_ids: torch.Tensor, red_ids: torch.Tensor) -> torch.Tensor:
        """Return [B] logits. Apply sigmoid externally for probability."""
        if blue_ids.shape[-1] != 5 or red_ids.shape[-1] != 5:
            raise ValueError(
                f"Expected 5 slots per side, got blue={blue_ids.shape}, red={red_ids.shape}"
            )

        b_emb = self.embedding(blue_ids)  # [B, 5, D]
        r_emb = self.embedding(red_ids)   # [B, 5, D]

        # Mean over non-pad slots (avoid dividing by zero)
        b_mask = (blue_ids != PAD_ID).float().unsqueeze(-1)  # [B, 5, 1]
        r_mask = (red_ids != PAD_ID).float().unsqueeze(-1)
        b_mean = (b_emb * b_mask).sum(dim=1) / b_mask.sum(dim=1).clamp(min=1.0)
        r_mean = (r_emb * r_mask).sum(dim=1) / r_mask.sum(dim=1).clamp(min=1.0)

        # Per-slot embeddings flattened (preserves slot identity for deep branch)
        b_flat = b_emb.reshape(b_emb.shape[0], -1)
        r_flat = r_emb.reshape(r_emb.shape[0], -1)

        deep_in = torch.cat([b_mean, r_mean, b_flat, r_flat], dim=1)
        deep_logit = self.deep(deep_in).squeeze(-1)  # [B]

        # Wide: each champion contributes a learned scalar per (side, slot).
        # Slots 0-4 are blue TOP..UTILITY; slots 5-9 are red TOP..UTILITY.
        # Mask out PAD ids so they contribute zero (padding_idx=0 already returns zeros).
        wide_b = self.wide_weights(blue_ids)  # [B, 5, 10]
        wide_r = self.wide_weights(red_ids)
        # Pick the slot-specific column for each position.
        slot_b = torch.arange(5, device=blue_ids.device).view(1, 5, 1)
        slot_r = torch.arange(5, 10, device=red_ids.device).view(1, 5, 1)
        wide_b = wide_b.gather(2, slot_b.expand(blue_ids.shape[0], 5, 1)).squeeze(-1)
        wide_r = wide_r.gather(2, slot_r.expand(red_ids.shape[0], 5, 1)).squeeze(-1)
        wide_logit = wide_b.sum(dim=1) + wide_r.sum(dim=1)

        return deep_logit + wide_logit
```

- [ ] **Step 2.5: Run tests, confirm they pass**

```bash
cd /Users/bao/Documents/FML/.claude/worktrees/zen-williams-5fea65 && python3 -m pytest tests/test_wide_deep_model.py -v
```

Expected: 4 passed.

- [ ] **Step 2.6: Commit**

```bash
git add src/models/wide_deep.py tests/__init__.py tests/test_wide_deep_model.py && git commit -m "feat(model): add Wide & Deep draft network architecture with tests"
```

---

## Task 3: Wide & Deep Inference Adapter (TDD, Graceful Loading)

**Files:**
- Create: `src/inference/wide_deep_adapter.py`
- Test: `tests/test_wide_deep_adapter.py`

The adapter is the boundary between server.py and the model. It MUST handle missing artifacts gracefully so server.py boots without `wide_deep.pt`.

- [ ] **Step 3.1: Write failing tests `tests/test_wide_deep_adapter.py`**

```python
"""Tests for WideDeepDraftAdapter — must boot gracefully when artifacts are missing."""
import json
from pathlib import Path

import pytest
import torch

from src.inference.wide_deep_adapter import WideDeepDraftAdapter
from src.models.wide_deep import (
    PAD_TOKEN,
    UNK_TOKEN,
    ROLE_ORDER,
    WideDeepDraftNet,
)


def test_missing_artifacts_does_not_raise(tmp_path: Path):
    adapter = WideDeepDraftAdapter(model_dir=tmp_path)
    assert adapter.available is False


def test_missing_artifacts_predict_returns_none(tmp_path: Path):
    adapter = WideDeepDraftAdapter(model_dir=tmp_path)
    assert adapter.predict_blue_win_prob([], []) is None
    assert adapter.predict_side_win_prob([], [], side="blue") is None


def _write_fake_artifacts(tmp_path: Path, champions: list[str]) -> None:
    """Write a minimal valid set of W&D artifacts to a temp dir."""
    vocab = {
        "champion_to_id": {PAD_TOKEN: 0, UNK_TOKEN: 1, **{c: i + 2 for i, c in enumerate(champions)}},
        "id_to_champion": {0: PAD_TOKEN, 1: UNK_TOKEN, **{i + 2: c for i, c in enumerate(champions)}},
        "pad_token": PAD_TOKEN,
        "unk_token": UNK_TOKEN,
        "role_order": ROLE_ORDER,
    }
    (tmp_path / "wide_deep_vocab.json").write_text(json.dumps(vocab))
    config = {
        "model_name": "wide_deep_draft_v1",
        "embedding_dim": 8,
        "hidden_dims": [16, 8],
        "dropout": 0.0,
        "champion_dropout": 0.0,
        "target": "blue_win",
        "output": "blue_win_prob",
    }
    (tmp_path / "wide_deep_config.json").write_text(json.dumps(config))
    net = WideDeepDraftNet(
        num_champions=len(champions) + 2,
        embedding_dim=config["embedding_dim"],
        hidden_dims=tuple(config["hidden_dims"]),
        dropout=config["dropout"],
    )
    torch.save(net.state_dict(), tmp_path / "wide_deep.pt")


def test_loads_when_artifacts_present(tmp_path: Path):
    _write_fake_artifacts(tmp_path, ["Yasuo", "Jinx", "Thresh", "LeeSin", "Garen"])
    adapter = WideDeepDraftAdapter(model_dir=tmp_path)
    assert adapter.available is True


def test_predict_blue_win_prob_with_dict_input(tmp_path: Path):
    _write_fake_artifacts(tmp_path, ["Yasuo", "Jinx", "Thresh", "LeeSin", "Garen"])
    adapter = WideDeepDraftAdapter(model_dir=tmp_path)
    blue = {"TOP": "Garen", "JUNGLE": "LeeSin", "MIDDLE": "Yasuo", "BOTTOM": "Jinx", "UTILITY": "Thresh"}
    red = {"TOP": "Yasuo", "JUNGLE": "LeeSin", "MIDDLE": "Garen", "BOTTOM": "Jinx", "UTILITY": "Thresh"}
    p = adapter.predict_blue_win_prob(blue, red)
    assert p is not None
    assert 0.0 <= p <= 1.0


def test_predict_blue_win_prob_with_partial_list_input(tmp_path: Path):
    _write_fake_artifacts(tmp_path, ["Yasuo", "Jinx", "Thresh", "LeeSin", "Garen"])
    adapter = WideDeepDraftAdapter(model_dir=tmp_path)
    blue = ["Garen", None, "Yasuo", None, None]  # partial — Nones become __PAD__
    red = [None] * 5
    p = adapter.predict_blue_win_prob(blue, red)
    assert p is not None
    assert 0.0 <= p <= 1.0


def test_unknown_champion_maps_to_unk(tmp_path: Path):
    _write_fake_artifacts(tmp_path, ["Yasuo", "Jinx", "Thresh", "LeeSin", "Garen"])
    adapter = WideDeepDraftAdapter(model_dir=tmp_path)
    blue = ["NotAChampion", None, None, None, None]
    red = [None] * 5
    p = adapter.predict_blue_win_prob(blue, red)
    assert p is not None
    assert 0.0 <= p <= 1.0


def test_side_conversion(tmp_path: Path):
    _write_fake_artifacts(tmp_path, ["Yasuo", "Jinx", "Thresh", "LeeSin", "Garen"])
    adapter = WideDeepDraftAdapter(model_dir=tmp_path)
    blue = {"TOP": "Garen", "JUNGLE": "LeeSin", "MIDDLE": "Yasuo", "BOTTOM": "Jinx", "UTILITY": "Thresh"}
    red = {"TOP": "Yasuo", "JUNGLE": "LeeSin", "MIDDLE": "Garen", "BOTTOM": "Jinx", "UTILITY": "Thresh"}
    p_blue = adapter.predict_side_win_prob(blue, red, side="blue")
    p_red = adapter.predict_side_win_prob(blue, red, side="red")
    assert p_blue is not None and p_red is not None
    assert abs((p_blue + p_red) - 1.0) < 1e-5
```

- [ ] **Step 3.2: Run tests, confirm they fail**

```bash
cd /Users/bao/Documents/FML/.claude/worktrees/zen-williams-5fea65 && python3 -m pytest tests/test_wide_deep_adapter.py -v
```

Expected: ImportError on `src.inference.wide_deep_adapter`.

- [ ] **Step 3.3: Implement `src/inference/wide_deep_adapter.py`**

```python
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
```

- [ ] **Step 3.4: Run tests, confirm they pass**

```bash
cd /Users/bao/Documents/FML/.claude/worktrees/zen-williams-5fea65 && python3 -m pytest tests/test_wide_deep_adapter.py -v
```

Expected: 7 passed.

- [ ] **Step 3.5: Commit**

```bash
git add src/inference/wide_deep_adapter.py tests/test_wide_deep_adapter.py && git commit -m "feat(inference): add WideDeepDraftAdapter with graceful fallback"
```

---

## Task 4: Hybrid Recommender Function (TDD)

**Files:**
- Modify: `src/inference/draft_recommender.py` — append new function, do NOT remove existing.
- Test: `tests/test_recommend_hybrid.py`

**Important:** existing `recommend_at_step` keeps the same signature and is still callable. The new `recommend_hybrid` returns a list of dicts (richer schema), and is what server.py will call.

- [ ] **Step 4.1: Write failing tests `tests/test_recommend_hybrid.py`**

```python
"""Tests for the hybrid recommender that combines performance score with W&D win prob."""
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.inference.draft_recommender import recommend_hybrid


class _StubAdapter:
    """Minimal adapter stub for testing. ``available`` is configurable."""

    def __init__(self, available: bool, prob: float = 0.62):
        self.available = available
        self.model_version = "stub_v1" if available else "unavailable"
        self._prob = prob

    def predict_side_win_prob(self, blue, red, side):
        if not self.available:
            return None
        return self._prob if side == "blue" else 1.0 - self._prob


def _stub_model(score_fn):
    m = MagicMock()
    m.predict = MagicMock(side_effect=lambda X: np.array([score_fn(row) for row in X]))
    return m


def _stub_resources():
    """Build the minimal embed_dict + champ_scores needed by build_candidate_features."""
    rng = np.random.default_rng(0)
    champs = ["Yasuo", "Jinx", "Thresh", "LeeSin", "Garen", "Ahri", "Lulu", "Kaisa"]
    embed_dict = {c: rng.standard_normal(64).astype(np.float32) for c in champs}
    champ_scores = {c: 50.0 + rng.standard_normal() * 5 for c in champs}
    return champs, embed_dict, champ_scores


def test_returns_top_k_dicts_with_required_fields():
    champs, embed_dict, champ_scores = _stub_resources()
    model = _stub_model(lambda row: 55.0)
    adapter = _StubAdapter(available=True, prob=0.6)

    out = recommend_hybrid(
        step=0,
        blue_picks=[None] * 5,
        red_picks=[None] * 5,
        model=model,
        embed_dict=embed_dict,
        champ_scores=champ_scores,
        candidate_pool=champs,
        banned=[],
        top_k=3,
        wide_deep_adapter=adapter,
        alpha=0.6,
        rerank_top_n=5,
    )
    assert len(out) == 3
    for r in out:
        assert "champion" in r
        assert "performance_score" in r
        assert "wide_deep_side_win_prob" in r
        assert "wide_deep_blue_win_prob" in r
        assert "final_rank_score" in r
        assert "win_prob" in r
        assert "score" in r
        assert "prob_source" in r
        assert r["prob_source"] == "wide_deep"


def test_fallback_when_adapter_unavailable():
    champs, embed_dict, champ_scores = _stub_resources()
    model = _stub_model(lambda row: 55.0)
    adapter = _StubAdapter(available=False)

    out = recommend_hybrid(
        step=0,
        blue_picks=[None] * 5,
        red_picks=[None] * 5,
        model=model,
        embed_dict=embed_dict,
        champ_scores=champ_scores,
        candidate_pool=champs,
        banned=[],
        top_k=3,
        wide_deep_adapter=adapter,
    )
    for r in out:
        assert r["prob_source"] == "score_heuristic_fallback"
        assert r["wide_deep_side_win_prob"] is None
        assert r["wide_deep_blue_win_prob"] is None
        # final_rank_score == normalized performance score
        assert abs(r["final_rank_score"] - r["performance_score"] / 100.0) < 1e-6


def test_fallback_when_adapter_is_none():
    champs, embed_dict, champ_scores = _stub_resources()
    model = _stub_model(lambda row: 55.0)
    out = recommend_hybrid(
        step=0,
        blue_picks=[None] * 5,
        red_picks=[None] * 5,
        model=model,
        embed_dict=embed_dict,
        champ_scores=champ_scores,
        candidate_pool=champs,
        banned=[],
        top_k=2,
        wide_deep_adapter=None,
    )
    for r in out:
        assert r["prob_source"] == "score_heuristic_fallback"


def test_banned_excluded():
    champs, embed_dict, champ_scores = _stub_resources()
    model = _stub_model(lambda row: 55.0)
    out = recommend_hybrid(
        step=0,
        blue_picks=[None] * 5,
        red_picks=[None] * 5,
        model=model,
        embed_dict=embed_dict,
        champ_scores=champ_scores,
        candidate_pool=champs,
        banned=["Yasuo", "Jinx"],
        top_k=10,
        wide_deep_adapter=None,
    )
    names = [r["champion"] for r in out]
    assert "Yasuo" not in names
    assert "Jinx" not in names
```

- [ ] **Step 4.2: Run tests, confirm they fail**

```bash
cd /Users/bao/Documents/FML/.claude/worktrees/zen-williams-5fea65 && python3 -m pytest tests/test_recommend_hybrid.py -v
```

Expected: ImportError — `recommend_hybrid` not defined.

- [ ] **Step 4.3: Append `recommend_hybrid` to `src/inference/draft_recommender.py`**

Open the file. Add at the bottom (do not modify any existing function):

```python
# ============================================================================
# Hybrid recommender — combines performance score with Wide & Deep win prob.
# Existing `recommend_at_step` is preserved for backward compatibility.
# ============================================================================

import numpy as np

from src.features.synergy_features import build_candidate_features


def _legacy_win_prob(score: float) -> float:
    """The original heuristic — kept here so it's importable for fallback."""
    return max(0.0, min(1.0, 0.50 + (score - 50.0) * 0.01))


def recommend_hybrid(
    step,
    blue_picks,
    red_picks,
    model,
    embed_dict,
    champ_scores,
    candidate_pool=None,
    banned=None,
    top_k: int = 5,
    wide_deep_adapter=None,
    alpha: float = 0.6,
    rerank_top_n: int = 30,
):
    """Recommend top-k picks using performance score + W&D win prob.

    Pipeline:
      1. Score every legal candidate with the existing HGBR model.
      2. Take the top ``rerank_top_n`` by performance score.
      3. For each, ask the W&D adapter for the side-specific win prob if available.
      4. final_rank_score = alpha * (perf_score / 100) + (1 - alpha) * win_prob.
      5. If W&D unavailable: final_rank_score = perf_score / 100; win_prob falls back
         to the legacy heuristic; prob_source = "score_heuristic_fallback".

    Returns a list of dicts (length top_k or fewer):
        champion, score, win_prob, performance_score,
        wide_deep_blue_win_prob, wide_deep_side_win_prob,
        final_rank_score, prob_source
    """
    from src.inference.draft_recommender import DRAFT_ORDER, _legal_pool

    side, _slot = DRAFT_ORDER[step]

    # Determine ally/enemy lists for the candidate's *own* perspective
    if side == "blue":
        my_picks, opp_picks = blue_picks, red_picks
    else:
        my_picks, opp_picks = red_picks, blue_picks
    allies = [p for p in my_picks if p]
    enemies = [p for p in opp_picks if p]

    # 1. Build legal candidate pool
    pool = _legal_pool(candidate_pool, embed_dict, banned, blue_picks, red_picks)

    # 2. Score them all
    feats = np.stack([
        build_candidate_features(c, allies, enemies, embed_dict, champ_scores)
        for c in pool
    ])
    perf_scores = model.predict(feats)

    # 3. Take top-N for rerank
    order = np.argsort(perf_scores)[::-1][:rerank_top_n]
    candidates = [(pool[i], float(perf_scores[i])) for i in order]

    use_wd = wide_deep_adapter is not None and getattr(wide_deep_adapter, "available", False)

    results = []
    for name, perf in candidates:
        norm_perf = max(0.0, min(1.0, perf / 100.0))
        wd_blue = None
        wd_side = None
        if use_wd:
            # Insert candidate into the empty draft slot for the current side
            trial_blue = list(blue_picks)
            trial_red = list(red_picks)
            (trial_blue if side == "blue" else trial_red)[_slot] = name
            wd_blue = wide_deep_adapter.predict_blue_win_prob(trial_blue, trial_red)
            if wd_blue is not None:
                wd_side = wd_blue if side == "blue" else 1.0 - wd_blue

        if wd_side is not None:
            final = alpha * norm_perf + (1 - alpha) * wd_side
            win_prob = wd_side
            source = "wide_deep"
        else:
            final = norm_perf
            win_prob = _legacy_win_prob(perf)
            source = "score_heuristic_fallback"

        results.append({
            "champion": name,
            "score": round(perf, 1),
            "win_prob": round(win_prob, 4),
            "performance_score": round(perf, 2),
            "wide_deep_blue_win_prob": None if wd_blue is None else round(wd_blue, 4),
            "wide_deep_side_win_prob": None if wd_side is None else round(wd_side, 4),
            "final_rank_score": round(final, 4),
            "prob_source": source,
        })

    results.sort(key=lambda r: r["final_rank_score"], reverse=True)
    return results[:top_k]
```

You will also need a small helper `_legal_pool` if it doesn't exist. Check first:

```bash
grep -n "_legal_pool\|def recommend_at_step" /Users/bao/Documents/FML/.claude/worktrees/zen-williams-5fea65/src/inference/draft_recommender.py
```

If `_legal_pool` is not present, add this helper above `recommend_hybrid`:

```python
def _legal_pool(candidate_pool, embed_dict, banned, blue_picks, red_picks):
    if candidate_pool is None:
        candidate_pool = list(embed_dict.keys())
    banned = set(banned or [])
    picked = {p for p in (list(blue_picks) + list(red_picks)) if p}
    return [c for c in candidate_pool if c not in banned and c not in picked]
```

- [ ] **Step 4.4: Run tests, confirm they pass**

```bash
cd /Users/bao/Documents/FML/.claude/worktrees/zen-williams-5fea65 && python3 -m pytest tests/test_recommend_hybrid.py -v
```

Expected: 4 passed.

- [ ] **Step 4.5: Commit**

```bash
git add src/inference/draft_recommender.py tests/test_recommend_hybrid.py && git commit -m "feat(inference): add recommend_hybrid combining perf-score and W&D"
```

---

## Task 5: Server Integration (Routes Test First)

**Files:**
- Modify: `app/server.py`
- Test: `tests/test_app_routes.py`

- [ ] **Step 5.1: Write failing route tests `tests/test_app_routes.py`**

```python
"""End-to-end Flask route tests using the real server module.

These tests run against the *actual* loaded artifacts. They verify the API contract
without depending on whether wide_deep.pt is present (fallback path is exercised).
"""
import json

import pytest

from app.server import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_get_champions_ok(client):
    resp = client.get("/api/champions")
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "name" in data[0]


def test_recommend_returns_required_fields(client):
    resp = client.post(
        "/api/recommend",
        data=json.dumps({
            "blue_picks": [None] * 5,
            "red_picks": [None] * 5,
            "blue_bans": [],
            "red_bans": [],
            "step": 0,
        }),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert "recommendations" in data
    assert len(data["recommendations"]) > 0
    rec = data["recommendations"][0]
    # backward-compat fields
    assert "champion" in rec
    assert "score" in rec
    assert "win_prob" in rec
    # new fields
    assert "performance_score" in rec
    assert "wide_deep_side_win_prob" in rec
    assert "wide_deep_blue_win_prob" in rec
    assert "final_rank_score" in rec
    assert "prob_source" in rec
    assert rec["prob_source"] in {"wide_deep", "score_heuristic_fallback"}


def test_evaluate_returns_prob_source(client):
    # Take any 10 champions from the loaded vocab
    from app.server import champion_list
    blue = champion_list[:5]
    red = champion_list[5:10]
    resp = client.post(
        "/api/evaluate",
        data=json.dumps({"blue_picks": blue, "red_picks": red}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    # backward-compat fields
    assert "blue_win_prob" in data
    assert "red_win_prob" in data
    assert "blue_score" in data
    assert "red_score" in data
    # new fields
    assert "prob_source" in data
    assert data["prob_source"] in {"wide_deep", "score_heuristic_fallback"}
    assert "model_version" in data
```

- [ ] **Step 5.2: Run tests, confirm they fail (or pass partially due to existing routes)**

```bash
cd /Users/bao/Documents/FML/.claude/worktrees/zen-williams-5fea65 && python3 -m pytest tests/test_app_routes.py -v
```

Expected: `test_get_champions_ok` likely passes; the others fail because new fields don't exist yet.

- [ ] **Step 5.3: Modify `app/server.py` — load adapter at startup**

After the existing model-loading block (around line 40), add:

```python
from src.inference.wide_deep_adapter import WideDeepDraftAdapter
from src.inference.draft_recommender import recommend_hybrid

wide_deep_adapter = WideDeepDraftAdapter(model_dir=DRAFT_MODELS_DIR)
print(f"Wide & Deep adapter available: {wide_deep_adapter.available}")
```

- [ ] **Step 5.4: Modify `/api/recommend` handler**

Replace the body of `api_recommend()` (lines roughly 113-172) with a call to `recommend_hybrid`:

```python
@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    """Top-K recommendations with hybrid (perf + W&D) ranking."""
    body = request.get_json(force=True)
    blue_picks = body.get("blue_picks", [None] * 5)
    red_picks = body.get("red_picks", [None] * 5)
    blue_bans = body.get("blue_bans", [])
    red_bans = body.get("red_bans", [])
    step = body.get("step", None)
    role_filter = body.get("role", None)
    top_k = int(body.get("top_k", 5))

    if step is None:
        step = get_current_step(blue_picks, red_picks)

    if step >= len(DRAFT_ORDER):
        return jsonify({
            "side": None, "slot": None, "recommendations": [],
            "prob_source": "wide_deep" if wide_deep_adapter.available else "score_heuristic_fallback",
            "model_version": wide_deep_adapter.model_version,
            "warnings": [],
        })

    side, slot = DRAFT_ORDER[step]

    candidate_pool = None
    if role_filter and role_filter in role_options:
        candidate_pool = role_options[role_filter]

    banned = [b for b in (blue_bans + red_bans) if b]

    recs = recommend_hybrid(
        step=step,
        blue_picks=blue_picks,
        red_picks=red_picks,
        model=draft_model,
        embed_dict=embed_dict,
        champ_scores=champ_scores,
        candidate_pool=candidate_pool,
        banned=banned,
        top_k=top_k,
        wide_deep_adapter=wide_deep_adapter,
        alpha=0.6,
        rerank_top_n=30,
    )

    warnings = []
    if not wide_deep_adapter.available:
        warnings.append("Wide & Deep artifact missing — using score-derived heuristic for win_prob.")

    return jsonify({
        "step": step,
        "side": side,
        "slot": slot,
        "recommendations": recs,
        "prob_source": "wide_deep" if wide_deep_adapter.available else "score_heuristic_fallback",
        "model_version": wide_deep_adapter.model_version,
        "warnings": warnings,
    })
```

- [ ] **Step 5.5: Modify `/api/evaluate` handler**

Replace its body with:

```python
@app.route("/api/evaluate", methods=["POST"])
def api_evaluate():
    """Final composition win probability — uses W&D when available."""
    body = request.get_json(force=True)
    blue_picks = body.get("blue_picks", [])
    red_picks = body.get("red_picks", [])

    warnings = []

    if (
        len(blue_picks) != 5
        or len(red_picks) != 5
        or any(p is None for p in blue_picks + red_picks)
    ):
        return jsonify({
            "blue_score": 50.0,
            "blue_win_prob": 0.5,
            "red_score": 50.0,
            "red_win_prob": 0.5,
            "prob_source": "score_heuristic_fallback",
            "model_version": wide_deep_adapter.model_version,
            "warnings": ["Incomplete draft — returning neutral win prob."],
        })

    # Always compute the legacy avg-score figures (used for the "score" display fields)
    blue_scores = []
    for i, candidate in enumerate(blue_picks):
        allies = [p for j, p in enumerate(blue_picks) if j != i]
        enemies = red_picks
        feats = build_candidate_features(candidate, allies, enemies, embed_dict, champ_scores)
        blue_scores.append(float(draft_model.predict(feats.reshape(1, -1))[0]))
    avg_blue_score = sum(blue_scores) / len(blue_scores)
    avg_red_score = 100.0 - avg_blue_score

    if wide_deep_adapter.available:
        blue_win_prob = wide_deep_adapter.predict_blue_win_prob(blue_picks, red_picks)
        red_win_prob = 1.0 - blue_win_prob
        source = "wide_deep"
    else:
        blue_win_prob = max(0.0, min(1.0, 0.50 + (avg_blue_score - 50.0) * 0.01))
        red_win_prob = 1.0 - blue_win_prob
        source = "score_heuristic_fallback"
        warnings.append("Wide & Deep artifact missing — using score-derived heuristic for win_prob.")

    return jsonify({
        "blue_score": round(avg_blue_score, 1),
        "blue_win_prob": round(blue_win_prob, 4),
        "red_score": round(avg_red_score, 1),
        "red_win_prob": round(red_win_prob, 4),
        "prob_source": source,
        "model_version": wide_deep_adapter.model_version,
        "warnings": warnings,
    })
```

- [ ] **Step 5.6: Run route tests**

```bash
cd /Users/bao/Documents/FML/.claude/worktrees/zen-williams-5fea65 && python3 -m pytest tests/test_app_routes.py -v
```

Expected: 3 passed.

- [ ] **Step 5.7: Run full test suite**

```bash
cd /Users/bao/Documents/FML/.claude/worktrees/zen-williams-5fea65 && python3 -m pytest -v
```

Expected: all tests pass.

- [ ] **Step 5.8: Smoke-test the running server**

```bash
cd /Users/bao/Documents/FML/.claude/worktrees/zen-williams-5fea65 && python3 app/server.py &
sleep 3
curl -s http://localhost:8080/api/champions | python3 -c "import sys,json; print('count:', len(json.load(sys.stdin)))"
curl -s -X POST http://localhost:8080/api/recommend -H "Content-Type: application/json" -d '{"blue_picks":[null,null,null,null,null],"red_picks":[null,null,null,null,null],"step":0}' | python3 -m json.tool | head -25
kill %1 2>/dev/null || true
```

Expected: champions count > 100; recommendations include `prob_source` field.

- [ ] **Step 5.9: Commit**

```bash
git add app/server.py tests/test_app_routes.py && git commit -m "feat(server): wire W&D adapter into /api/recommend and /api/evaluate with fallback"
```

---

## Task 6: Wide & Deep Training Script (Optional Pipeline)

**File:**
- Create: `src/models/train_wide_deep.py`

The training script reads `data/raw/compositions_s16.csv`, pivots to match-level rows, splits, trains, and writes artifacts. It is NOT required for demo to run.

- [ ] **Step 6.1: Implement `src/models/train_wide_deep.py`**

```python
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
```

- [ ] **Step 6.2: Verify the script imports cleanly (do not run training — data may be missing)**

```bash
cd /Users/bao/Documents/FML/.claude/worktrees/zen-williams-5fea65 && python3 -c "from src.models.train_wide_deep import build_match_table, build_vocab, CONFIG; print('imports OK')"
```

Expected: `imports OK`.

- [ ] **Step 6.3: Commit**

```bash
git add src/models/train_wide_deep.py && git commit -m "feat(model): training script for Wide & Deep with patch-aware split"
```

---

## Task 7: README & Frontend Polish

**Files:**
- Modify: `README.md`
- Modify: `app/static/app.js` (minimal additions only)

- [ ] **Step 7.1: Update `README.md` — replace overclaim phrasing**

Read the current README, then locate and rewrite any of these phrases:

```
mathematically optimal recommendation
guaranteed win probability
strictly predicts match outcome only from score model
```

Replace the section that describes prediction with this paragraph (place it after the "How It Works" / pipeline section):

```markdown
## Two-Model Architecture

This project uses two complementary models:

1. **Champion2Vec + HistGradientBoostingRegressor** predicts a candidate champion's
   *performance score* (0–100) for the current draft state. The Champion2Vec embeddings
   are trained from scratch using a custom NumPy SGD matrix factorization over
   synergy and matchup matrices.

2. **Wide & Deep** predicts the *draft-level blue-side win probability* from the
   full 10-champion composition.

The web demo uses pretrained artifacts that ship in
`data/processed/draft_models/`. Retraining is optional and is not required to run
the demo. The displayed win probability comes from the Wide & Deep model when
available; if the `wide_deep.pt` artifact is missing, the app falls back to the
original score-derived display probability and labels it as a fallback in the
response (`prob_source: "score_heuristic_fallback"`).
```

- [ ] **Step 7.2: Update README — running the demo section**

Make sure the "How to run" section reads:

```markdown
## Running the demo

```bash
pip install -r requirements.txt
python app/server.py
```

The demo starts on `http://localhost:8080`. No Riot API key, raw data download,
or retraining is needed — pretrained artifacts under
`data/processed/draft_models/` are sufficient.

To retrain (optional):

```bash
pip install -r requirements-dev.txt
python -m src.models.train_draft_models   # Champion2Vec + HGBR
python -m src.models.train_wide_deep      # Wide & Deep (requires compositions_s16.csv)
```
```

- [ ] **Step 7.3: Inspect frontend usage of new fields**

```bash
grep -n "prob_source\|warnings\|wide_deep" /Users/bao/Documents/FML/.claude/worktrees/zen-williams-5fea65/app/static/app.js
```

Expected: no matches — these are new fields.

- [ ] **Step 7.4: Add a non-breaking display for `prob_source` and `warnings` in `app/static/app.js`**

Find where the recommendations are rendered (around the existing `.score`/`.win_prob` rendering in the recommend response handler — the agent reported lines 458, 479-485). Add an unobtrusive line-end label inside the same render block. Pseudocode for the edit:

```javascript
// Inside the response handler for /api/recommend, after the existing render:
if (data.prob_source === 'score_heuristic_fallback' && data.warnings && data.warnings.length) {
    const warnEl = document.createElement('div');
    warnEl.className = 'recommend-warning';
    warnEl.textContent = '⚠ ' + data.warnings.join(' ');
    // Insert at the top of the recommendations container — adjust the selector to match the existing UI:
    const container = document.querySelector('#recommendations') || document.querySelector('.recommendations');
    if (container) container.prepend(warnEl);
}
```

**Important:** the exact selector and insertion point depends on the existing `app.js`. Before editing, read the surrounding lines (around line 458) and adapt. Do **not** change any existing field names.

Add a CSS rule at the bottom of `app/static/style.css`:

```css
.recommend-warning {
    color: #b08800;
    background: #fff8d8;
    border: 1px solid #e6c800;
    padding: 6px 10px;
    border-radius: 6px;
    margin-bottom: 8px;
    font-size: 0.9em;
}
```

- [ ] **Step 7.5: Manual UI verification**

```bash
cd /Users/bao/Documents/FML/.claude/worktrees/zen-williams-5fea65 && python3 app/server.py
```

Open `http://localhost:8080`, run a draft step, confirm:
- Recommendations still show champion / score / win prob (no regressions).
- A small yellow warning banner appears IF `wide_deep.pt` is missing.
- No console errors.

Stop the server (Ctrl-C) when satisfied.

- [ ] **Step 7.6: Commit**

```bash
git add README.md app/static/app.js app/static/style.css && git commit -m "docs: dual-model README; ui: non-breaking W&D fallback warning"
```

---

## Task 8: Final Verification & Manifest Check

- [ ] **Step 8.1: Run the entire test suite**

```bash
cd /Users/bao/Documents/FML/.claude/worktrees/zen-williams-5fea65 && python3 -m pytest -v
```

Expected: all tests pass.

- [ ] **Step 8.2: End-to-end smoke test**

```bash
cd /Users/bao/Documents/FML/.claude/worktrees/zen-williams-5fea65 && python3 app/server.py &
sleep 3
curl -s http://localhost:8080/api/champions > /tmp/champs.json && echo "champions OK ($(python3 -c 'import json; print(len(json.load(open("/tmp/champs.json"))))') champions)"
curl -s -X POST http://localhost:8080/api/recommend -H "Content-Type: application/json" -d '{"blue_picks":[null,null,null,null,null],"red_picks":[null,null,null,null,null],"step":0}' > /tmp/rec.json && python3 -c "import json; d=json.load(open('/tmp/rec.json')); r=d['recommendations'][0]; assert all(k in r for k in ['champion','score','win_prob','performance_score','final_rank_score','prob_source']); print('recommend OK, prob_source=', d['prob_source'])"
kill %1 2>/dev/null || true
```

Expected: both lines print OK.

- [ ] **Step 8.3: Verify acceptance criteria are met**

Check by reading code/output:
- App boots without `wide_deep.pt` ✓ (Task 5.7)
- `/api/champions` returns 200 ✓ (Task 8.2)
- `/api/recommend` returns 200 with new fields ✓ (Task 8.2)
- `/api/evaluate` returns `prob_source` ✓ (Task 5.6)
- Backward-compat: `champion`, `score`, `win_prob`, `blue_win_prob`, `red_win_prob`, `blue_score`, `red_score` all still present ✓
- Manifest at `data/processed/draft_models/model_manifest.json` lists three models ✓
- README mentions dual-model story without overclaim ✓
- `requirements.txt` slim, `requirements-dev.txt` separate ✓
- Frontend not broken; warnings shown on fallback ✓ (Task 7.5)

- [ ] **Step 8.4: Final commit (only if anything new was changed)**

```bash
cd /Users/bao/Documents/FML/.claude/worktrees/zen-williams-5fea65 && git status
```

If nothing pending, skip. Otherwise:

```bash
git add . && git commit -m "chore: finalize wide-deep hybrid integration"
```

---

## Summary

**Modified files:**
- `src/inference/draft_recommender.py`
- `app/server.py`
- `app/static/app.js`
- `app/static/style.css`
- `requirements.txt`
- `README.md`

**New files:**
- `src/models/wide_deep.py`
- `src/models/train_wide_deep.py`
- `src/inference/wide_deep_adapter.py`
- `tests/__init__.py`
- `tests/test_wide_deep_model.py`
- `tests/test_wide_deep_adapter.py`
- `tests/test_recommend_hybrid.py`
- `tests/test_app_routes.py`
- `requirements-dev.txt`
- `data/processed/draft_models/model_manifest.json`

**API backward compatibility:** Yes. All existing fields preserved. New fields are additive.

**How to run the demo:**

```bash
pip install -r requirements.txt
python app/server.py
```

**Wide & Deep fallback behavior when `wide_deep.pt` is missing:**
- Adapter sets `available = False`; no exception raised.
- `/api/recommend` and `/api/evaluate` use the legacy heuristic `win_prob = 0.5 + (score - 50) * 0.01`.
- Response includes `prob_source: "score_heuristic_fallback"` and a `warnings` entry.
- Frontend shows a small banner alerting the user.
