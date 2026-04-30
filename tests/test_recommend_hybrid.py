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
    """Stub the classifier interface used by recommend_hybrid.

    Returns a MagicMock whose predict_proba(X) yields [[1-p, p], ...] where p is
    score_fn(row) / 100 — so multiplying back by 100 (as the recommender does)
    recovers the original score, keeping the existing 0-100-scale tests working.
    """
    m = MagicMock()
    m.predict_proba = MagicMock(
        side_effect=lambda X: np.array([[1.0 - score_fn(row) / 100.0,
                                          score_fn(row) / 100.0] for row in X])
    )
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


# ---------- W&D-only path (alpha=0, adapter available) ----------

class _WDPerCandidateStub:
    """Adapter stub that returns a different W&D prob for every candidate name.

    Used to confirm that recommend_hybrid sorts by side win prob in the W&D-only
    path, and that the HGBR ``predict_proba`` is never invoked.
    """

    def __init__(self, name_to_prob):
        self.available = True
        self.model_version = "stub_wd"
        self._name_to_prob = name_to_prob
        self.batch_calls = 0
        self.single_calls = 0

    def predict_blue_win_prob_batch_with_raw(self, blue_picks_list, red_picks_list):
        self.batch_calls += 1
        out = []
        for blue, red in zip(blue_picks_list, red_picks_list):
            inserted = next((c for c in blue + red if c and c in self._name_to_prob), None)
            p = self._name_to_prob.get(inserted, 0.5)
            out.append((p, p))  # (calibrated, raw)
        return out

    def predict_blue_win_prob_batch(self, blue_picks_list, red_picks_list):
        return [pair[0] for pair in self.predict_blue_win_prob_batch_with_raw(
            blue_picks_list, red_picks_list)]

    def predict_side_win_prob(self, blue, red, side):
        self.single_calls += 1
        inserted = next((c for c in blue + red if c and c in self._name_to_prob), None)
        p = self._name_to_prob.get(inserted, 0.5)
        return p if side == "blue" else 1.0 - p


def test_wide_deep_only_path_bypasses_hgbr():
    champs, embed_dict, champ_scores = _stub_resources()
    # Predict_proba must never be called in this path.
    failing_model = MagicMock()
    failing_model.predict_proba = MagicMock(side_effect=AssertionError(
        "HGBR predict_proba must NOT be called when alpha<=eps and adapter is available"
    ))

    name_to_prob = {c: 0.40 + 0.05 * i for i, c in enumerate(champs)}  # ascending
    adapter = _WDPerCandidateStub(name_to_prob)

    out = recommend_hybrid(
        step=0,
        blue_picks=[None] * 5,
        red_picks=[None] * 5,
        model=failing_model,
        embed_dict=embed_dict,
        champ_scores=champ_scores,
        candidate_pool=champs,
        banned=[],
        top_k=3,
        wide_deep_adapter=adapter,
        alpha=0.0,                # → W&D-only path
        rerank_top_n=999,         # irrelevant when alpha=0
    )

    failing_model.predict_proba.assert_not_called()
    assert adapter.batch_calls == 1   # batched once, not per-candidate
    assert len(out) == 3
    # Results sorted by side win prob descending (side is blue at step 0).
    probs = [r["win_prob"] for r in out]
    assert probs == sorted(probs, reverse=True)
    # All 8 fields present + prob_source = wide_deep
    for r in out:
        for field in ("champion", "score", "performance_score", "win_prob",
                      "wide_deep_blue_win_prob", "wide_deep_side_win_prob",
                      "win_prob_wide_deep", "win_prob_heuristic",
                      "final_rank_score", "prob_source"):
            assert field in r, f"missing {field}"
        assert r["prob_source"] == "wide_deep"


def test_wide_deep_only_path_falls_back_when_adapter_unavailable():
    """alpha=0 but adapter unavailable → must NOT crash; goes to legacy fallback."""
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
        top_k=3,
        wide_deep_adapter=_StubAdapter(available=False),
        alpha=0.0,
    )
    assert len(out) == 3
    for r in out:
        assert r["prob_source"] == "score_heuristic_fallback"
