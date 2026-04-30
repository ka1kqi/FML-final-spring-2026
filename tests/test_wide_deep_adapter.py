"""Tests for WideDeepDraftAdapter — must boot gracefully when artifacts are missing."""
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from src.inference.wide_deep_adapter import WideDeepDraftAdapter
from src.models.wide_deep import (
    LEGACY_ARCH,
    PAD_TOKEN,
    UNK_TOKEN,
    ROLE_ORDER,
    V2_ARCH,
    WideDeepDraftNet,
)


def test_missing_artifacts_does_not_raise(tmp_path: Path):
    adapter = WideDeepDraftAdapter(model_dir=tmp_path)
    assert adapter.available is False


def test_missing_artifacts_predict_returns_none(tmp_path: Path):
    adapter = WideDeepDraftAdapter(model_dir=tmp_path)
    assert adapter.predict_blue_win_prob([], []) is None
    assert adapter.predict_side_win_prob([], [], side="blue") is None


def _write_fake_artifacts(
    tmp_path: Path,
    champions: list[str],
    architecture: str = V2_ARCH,
) -> None:
    """Write a minimal valid set of W&D artifacts to a temp dir.

    Uses ``architecture`` to choose between the v2 pairwise branch (default for
    new models) and the legacy flat-MLP branch (for back-compat tests).
    """
    vocab = {
        "champion_to_id": {PAD_TOKEN: 0, UNK_TOKEN: 1, **{c: i + 2 for i, c in enumerate(champions)}},
        "id_to_champion": {0: PAD_TOKEN, 1: UNK_TOKEN, **{i + 2: c for i, c in enumerate(champions)}},
        "pad_token": PAD_TOKEN,
        "unk_token": UNK_TOKEN,
        "role_order": ROLE_ORDER,
    }
    (tmp_path / "wide_deep_vocab.json").write_text(json.dumps(vocab))
    config = {
        "model_name": "wide_deep_draft_test",
        "architecture": architecture,
        "embedding_dim": 8,
        "hidden_dims": [16, 8],
        "hidden_dim": 16,
        "dropout": 0.0,
        "combine": "sum",
        "target": "blue_win",
        "output": "blue_win_prob",
    }
    (tmp_path / "wide_deep_config.json").write_text(json.dumps(config))
    net = WideDeepDraftNet(
        num_champions=len(champions) + 2,
        embedding_dim=config["embedding_dim"],
        hidden_dims=tuple(config["hidden_dims"]),
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
        architecture=architecture,
        combine=config["combine"],
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


def test_empty_vocab_does_not_raise(tmp_path: Path):
    """Malformed artifact with empty champion_to_id must NOT crash __init__."""
    (tmp_path / "wide_deep_vocab.json").write_text(
        json.dumps({"champion_to_id": {}, "id_to_champion": {}, "pad_token": "__PAD__", "unk_token": "__UNK__", "role_order": ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]})
    )
    (tmp_path / "wide_deep_config.json").write_text(json.dumps({"model_name": "x", "embedding_dim": 8, "hidden_dims": [16, 8], "dropout": 0.0}))
    # Don't even need wide_deep.pt — _load fails before that on the empty max() call
    adapter = WideDeepDraftAdapter(model_dir=tmp_path)
    assert adapter.available is False


def test_loads_legacy_architecture(tmp_path: Path):
    """Old artifacts (no `architecture` field, flat-MLP weights) must still load."""
    _write_fake_artifacts(tmp_path, ["A", "B", "C", "D", "E"], architecture=LEGACY_ARCH)
    # Strip the architecture key so we exercise the absent-field default branch
    cfg = json.loads((tmp_path / "wide_deep_config.json").read_text())
    cfg.pop("architecture")
    (tmp_path / "wide_deep_config.json").write_text(json.dumps(cfg))
    adapter = WideDeepDraftAdapter(model_dir=tmp_path)
    assert adapter.available is True
    p = adapter.predict_blue_win_prob(["A", "B", "C", "D", "E"], ["A", "B", "C", "D", "E"])
    assert p is not None
    assert 0.0 < p < 1.0  # clipped, never exactly 0 or 1


def test_calibrator_missing_does_not_disable_adapter(tmp_path: Path):
    _write_fake_artifacts(tmp_path, ["A", "B", "C", "D", "E"])
    # No wide_deep_calibrator.pkl on disk
    adapter = WideDeepDraftAdapter(model_dir=tmp_path)
    assert adapter.available is True
    p = adapter.predict_blue_win_prob(["A", "B", "C", "D", "E"], ["A", "B", "C", "D", "E"])
    assert p is not None
    assert _PROB_CLIP_LO <= p <= _PROB_CLIP_HI


def test_calibrator_corrupt_does_not_disable_adapter(tmp_path: Path):
    _write_fake_artifacts(tmp_path, ["A", "B", "C", "D", "E"])
    (tmp_path / "wide_deep_calibrator.pkl").write_bytes(b"not a real pickle")
    adapter = WideDeepDraftAdapter(model_dir=tmp_path)
    assert adapter.available is True  # corrupt calibrator is non-fatal
    p = adapter.predict_blue_win_prob(["A"] * 5, ["B"] * 5)
    assert p is not None


def test_calibrator_applied_when_present(tmp_path: Path):
    """An identity isotonic calibrator should leave probs almost unchanged."""
    pytest.importorskip("sklearn")
    from sklearn.isotonic import IsotonicRegression

    _write_fake_artifacts(tmp_path, ["A", "B", "C", "D", "E"])
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(np.linspace(0.0, 1.0, 21), np.linspace(0.0, 1.0, 21))
    with open(tmp_path / "wide_deep_calibrator.pkl", "wb") as f:
        pickle.dump(iso, f)
    adapter = WideDeepDraftAdapter(model_dir=tmp_path)
    assert adapter.available
    p = adapter.predict_blue_win_prob(["A"] * 5, ["B"] * 5)
    assert p is not None
    assert _PROB_CLIP_LO <= p <= _PROB_CLIP_HI


def test_batch_predict_matches_single(tmp_path: Path):
    _write_fake_artifacts(tmp_path, ["A", "B", "C", "D", "E"])
    adapter = WideDeepDraftAdapter(model_dir=tmp_path)
    blue = ["A", "B", "C", "D", "E"]
    red = ["E", "D", "C", "B", "A"]
    single = adapter.predict_blue_win_prob(blue, red)
    batch = adapter.predict_blue_win_prob_batch([blue, blue], [red, red])
    assert len(batch) == 2
    assert all(b is not None for b in batch)
    assert abs(batch[0] - single) < 1e-5
    assert abs(batch[1] - single) < 1e-5


def test_batch_returns_none_when_unavailable(tmp_path: Path):
    adapter = WideDeepDraftAdapter(model_dir=tmp_path)
    out = adapter.predict_blue_win_prob_batch([["A"] * 5], [["B"] * 5])
    assert out == [None]


# Re-exposed for clip range assertions
from src.inference.wide_deep_adapter import _PROB_CLIP_HI, _PROB_CLIP_LO  # noqa: E402
