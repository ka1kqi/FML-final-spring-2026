"""
Draft-order-aware champion recommender.

At each of the 10 pick steps (B1->R1,R2->B2,B3->R3,R4->B4,B5->R5),
scores all available champions for the current picking side and returns
the top-k with predicted win probabilities.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import numpy as np

from src.features.synergy_features import build_candidate_features


# Standard ranked draft order
DRAFT_ORDER = [
    ("Blue", 0), ("Red", 0), ("Red", 1), ("Blue", 1), ("Blue", 2),
    ("Red", 2), ("Red", 3), ("Blue", 3), ("Blue", 4), ("Red", 4),
]


def recommend_at_step(
    step: int,
    blue_picks: List[Optional[str]],
    red_picks: List[Optional[str]],
    model,
    embed_dict: Dict[str, np.ndarray],
    champ_scores: Dict[str, float],
    candidate_pool: Optional[List[str]] = None,
    banned: Optional[List[str]] = None,
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """
    Recommend top-k champions for the current draft step.

    Args:
        step: current step index (0-9)
        blue_picks: list of 5 slots, None for unfilled
        red_picks: list of 5 slots, None for unfilled
        model: trained HistGradientBoostingClassifier
        embed_dict: champion name -> embedding vector
        champ_scores: champion name -> historical avg comp score (feature)
        candidate_pool: list of valid champion names for this slot
                       (e.g. filtered by role). If None, uses all champions.
        banned: list of banned champion names to exclude
        top_k: number of recommendations to return

    Returns:
        List of (champion_name, win_prob in [0,1]) sorted descending.
    """
    if step < 0 or step >= len(DRAFT_ORDER):
        return []

    side, slot = DRAFT_ORDER[step]
    banned = banned or []

    # Determine current allies and enemies for the picking side
    if side == "Blue":
        allies = [p for p in blue_picks if p is not None]
        enemies = [p for p in red_picks if p is not None]
    else:
        allies = [p for p in red_picks if p is not None]
        enemies = [p for p in blue_picks if p is not None]

    # All already-picked + banned champions
    used = set(p for p in blue_picks + red_picks if p is not None)
    used.update(banned)

    # Candidate pool
    if candidate_pool is None:
        candidate_pool = list(embed_dict.keys())

    scored = []
    for champ in candidate_pool:
        if champ in used or champ not in embed_dict:
            continue

        features = build_candidate_features(
            champ, allies, enemies, embed_dict, champ_scores
        )
        win_prob = model.predict_proba(features.reshape(1, -1))[0, 1]

        scored.append((champ, float(win_prob)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def get_current_step(blue_picks: List[Optional[str]],
                     red_picks: List[Optional[str]]) -> int:
    """
    Determine the current draft step based on which slots are filled.

    Returns the index of the next unfilled step (0-9), or 10 if draft is complete.
    """
    for step_idx, (side, slot) in enumerate(DRAFT_ORDER):
        if side == "Blue" and blue_picks[slot] is None:
            return step_idx
        if side == "Red" and red_picks[slot] is None:
            return step_idx
    return len(DRAFT_ORDER)


def simulate_full_draft_recommendations(
    blue_picks: List[Optional[str]],
    red_picks: List[Optional[str]],
    model,
    embed_dict: Dict[str, np.ndarray],
    champ_scores: Dict[str, float],
    role_options: Optional[Dict[str, List[str]]] = None,
    blue_roles: Optional[List[str]] = None,
    red_roles: Optional[List[str]] = None,
    banned: Optional[List[str]] = None,
    top_k: int = 5,
) -> List[dict]:
    """
    Generate recommendations for all unfilled draft steps.

    Args:
        blue_picks: list of 5 slots (None for unfilled)
        red_picks: list of 5 slots (None for unfilled)
        model: trained HistGradientBoostingRegressor
        embed_dict: champion embeddings
        champ_scores: champion average comp scores
        role_options: dict mapping role name -> list of viable champions
        blue_roles: list of 5 role names assigned to blue slots
        red_roles: list of 5 role names assigned to red slots
        banned: list of banned champions
        top_k: number of recommendations per step

    Returns:
        List of dicts with keys: step, side, slot, role, recommendations
    """
    results = []
    ROLES = ["Top", "Jungle", "Mid", "ADC", "Support"]
    blue_roles = blue_roles or ROLES
    red_roles = red_roles or ROLES

    for step_idx, (side, slot) in enumerate(DRAFT_ORDER):
        # Skip already-filled slots
        if side == "Blue" and blue_picks[slot] is not None:
            continue
        if side == "Red" and red_picks[slot] is not None:
            continue

        # Determine role and candidate pool for this slot
        if side == "Blue":
            role = blue_roles[slot]
        else:
            role = red_roles[slot]

        candidate_pool = None
        if role_options and role in role_options:
            candidate_pool = role_options[role]

        recs = recommend_at_step(
            step=step_idx,
            blue_picks=blue_picks,
            red_picks=red_picks,
            model=model,
            embed_dict=embed_dict,
            champ_scores=champ_scores,
            candidate_pool=candidate_pool,
            banned=banned,
            top_k=top_k,
        )

        results.append({
            "step": step_idx,
            "side": side,
            "slot": slot,
            "role": role,
            "recommendations": recs,
        })

    return results


def load_draft_resources(models_dir: Path):
    """
    Load all resources needed for draft recommendations.

    Returns:
        (model, embed_dict, champ_scores, biases) tuple
        `biases` is a dict with keys
            mu_syn, b_syn_u, b_syn_v, mu_match, b_match_u, b_match_v
        each indexed by champion name. Empty dict if the artifact was
        produced by an older training run that doesn't ship biases.
    """
    from src.models.draft_classifier import load_draft_model

    model = load_draft_model(models_dir / "draft_model.joblib")

    data = np.load(str(models_dir / "champion2vec.npz"), allow_pickle=True)
    embed_weights = data["weights"]
    vocab = data["vocab"].tolist()
    embed_dict = {name: embed_weights[i] for i, name in enumerate(vocab)}

    biases: dict = {}
    if "b_syn_u" in data.files:
        biases = {
            "mu_syn": float(data["mu_syn"]),
            "mu_match": float(data["mu_match"]),
            "b_syn_u": {n: float(data["b_syn_u"][i]) for i, n in enumerate(vocab)},
            "b_syn_v": {n: float(data["b_syn_v"][i]) for i, n in enumerate(vocab)},
            "b_match_u": {n: float(data["b_match_u"][i]) for i, n in enumerate(vocab)},
            "b_match_v": {n: float(data["b_match_v"][i]) for i, n in enumerate(vocab)},
        }

    with open(models_dir / "champ_scores.json", "r") as f:
        champ_scores = json.load(f)

    return model, embed_dict, champ_scores, biases


# ============================================================================
# Hybrid recommender — combines performance score with Wide & Deep win prob.
# Existing `recommend_at_step` is preserved for backward compatibility.
# ============================================================================


def _legacy_win_prob(score: float) -> float:
    """The original heuristic — kept here so it's importable for fallback."""
    return max(0.0, min(1.0, 0.50 + (score - 50.0) * 0.01))


def _legal_pool(candidate_pool, embed_dict, banned, blue_picks, red_picks):
    """Filter candidates against bans, already-picked, and unknown champions.

    Mirrors the embed_dict-membership check in ``recommend_at_step`` so a champion
    present in the role pool but missing from the trained embeddings (e.g. a new
    champion added to the CSV before retraining) doesn't crash
    ``build_candidate_features``.
    """
    if candidate_pool is None:
        candidate_pool = list(embed_dict.keys())
    banned_set = set(banned or [])
    picked = {p for p in (list(blue_picks) + list(red_picks)) if p}
    return [
        c for c in candidate_pool
        if c not in banned_set and c not in picked and c in embed_dict
    ]


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
    """Recommend top-k picks; ranking strategy depends on alpha and W&D availability.

    Three modes:
      * **W&D-only** (alpha ≤ 1e-9 and wide_deep_adapter.available):
        - Enumerate ALL legal candidates (no HGBR pre-filter).
        - Score each with W&D side win probability.
        - Sort by side win prob descending.
        - score / performance_score / win_prob / final_rank_score all reflect
          W&D's side win prob × 100 (or in [0,1] for win_prob/final_rank_score).
        This is the path used by the "wide_deep" toggle in the UI.

      * **Hybrid** (0 < alpha < 1 and W&D available):
        - HGBR scores every legal candidate, top-rerank_top_n by perf score
          go to W&D for win-prob lookup.
        - final_rank_score = alpha * (perf/100) + (1-alpha) * wd_side.

      * **Heuristic / fallback** (alpha ≥ 1 or W&D unavailable):
        - HGBR's predict_proba is the score, win_prob comes from the legacy
          formula 0.5 + (perf-50)*0.01.

    Note: DRAFT_ORDER uses capitalized side strings ("Blue"/"Red"); side comparisons
    use .lower() to be case-insensitive. slot is an integer index (0-4).
    """
    side_raw, slot = DRAFT_ORDER[step]
    side = side_raw.lower()

    pool = _legal_pool(candidate_pool, embed_dict, banned, blue_picks, red_picks)
    if not pool:
        return []

    use_wd = wide_deep_adapter is not None and getattr(wide_deep_adapter, "available", False)

    # ---- W&D-only path ---------------------------------------------------
    if use_wd and alpha <= 1e-9:
        return _wide_deep_only_recommend(
            pool=pool,
            blue_picks=blue_picks,
            red_picks=red_picks,
            side=side,
            slot=slot,
            wide_deep_adapter=wide_deep_adapter,
            top_k=top_k,
        )

    # ---- Hybrid / heuristic path: keep historical HGBR-based scoring ----
    if side == "blue":
        my_picks, opp_picks = blue_picks, red_picks
    else:
        my_picks, opp_picks = red_picks, blue_picks
    allies = [p for p in my_picks if p]
    enemies = [p for p in opp_picks if p]

    feats = np.stack([
        build_candidate_features(c, allies, enemies, embed_dict, champ_scores)
        for c in pool
    ])
    perf_scores = model.predict_proba(feats)[:, 1] * 100.0

    order = np.argsort(perf_scores)[::-1][:rerank_top_n]
    candidates = [(pool[i], float(perf_scores[i])) for i in order]

    results = []
    for name, perf in candidates:
        norm_perf = max(0.0, min(1.0, perf / 100.0))
        wd_blue = None
        wd_side = None
        if use_wd:
            trial_blue = list(blue_picks)
            trial_red = list(red_picks)
            (trial_blue if side == "blue" else trial_red)[slot] = name
            wd_blue = wide_deep_adapter.predict_side_win_prob(trial_blue, trial_red, "blue")
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

        win_prob_heuristic = _legacy_win_prob(perf)
        results.append({
            "champion": name,
            "score": round(perf, 1),
            "win_prob": round(win_prob, 4),
            "performance_score": round(perf, 2),
            "wide_deep_blue_win_prob": None if wd_blue is None else round(wd_blue, 4),
            "wide_deep_side_win_prob": None if wd_side is None else round(wd_side, 4),
            "win_prob_wide_deep": None if wd_side is None else round(wd_side, 4),
            "win_prob_heuristic": round(win_prob_heuristic, 4),
            "final_rank_score": round(final, 4),
            "prob_source": source,
        })

    results.sort(key=lambda r: r["final_rank_score"], reverse=True)
    return results[:top_k]


def _wide_deep_only_recommend(
    pool,
    blue_picks,
    red_picks,
    side: str,
    slot: int,
    wide_deep_adapter,
    top_k: int,
):
    """Enumerate every legal candidate and rank purely by W&D side win prob.

    Uses the adapter's batch helper when available so 150+ candidates are
    scored in a single forward pass. Falls back to per-candidate calls if the
    adapter is a stub without ``predict_blue_win_prob_batch`` (e.g. tests).
    """
    blue_lists, red_lists, names = [], [], []
    for name in pool:
        trial_blue = list(blue_picks)
        trial_red = list(red_picks)
        (trial_blue if side == "blue" else trial_red)[slot] = name
        blue_lists.append(trial_blue)
        red_lists.append(trial_red)
        names.append(name)

    batch_fn = getattr(wide_deep_adapter, "predict_blue_win_prob_batch", None)
    if callable(batch_fn):
        wd_blues = batch_fn(blue_lists, red_lists)
    else:
        wd_blues = [
            wide_deep_adapter.predict_side_win_prob(b, r, "blue")
            for b, r in zip(blue_lists, red_lists)
        ]

    results = []
    for name, wd_blue in zip(names, wd_blues):
        if wd_blue is None:
            continue
        wd_side = wd_blue if side == "blue" else 1.0 - wd_blue
        # Heuristic stays available for diagnostics; perf maps from the same
        # win prob so the legacy 0–100 display remains consistent.
        perf_pct = wd_side * 100.0
        win_prob_heuristic = _legacy_win_prob(perf_pct)
        results.append({
            "champion": name,
            "score": round(perf_pct, 1),
            "performance_score": round(perf_pct, 2),
            "win_prob": round(wd_side, 4),
            "wide_deep_blue_win_prob": round(wd_blue, 4),
            "wide_deep_side_win_prob": round(wd_side, 4),
            "win_prob_wide_deep": round(wd_side, 4),
            "win_prob_heuristic": round(win_prob_heuristic, 4),
            "final_rank_score": round(wd_side, 4),
            "prob_source": "wide_deep",
        })

    results.sort(key=lambda r: r["final_rank_score"], reverse=True)
    return results[:top_k]
