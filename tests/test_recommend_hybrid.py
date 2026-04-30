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
