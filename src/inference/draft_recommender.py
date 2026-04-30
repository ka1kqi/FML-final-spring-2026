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
        model: trained HistGradientBoostingRegressor
        embed_dict: champion name -> embedding vector
        champ_scores: champion name -> average comp score
        candidate_pool: list of valid champion names for this slot
                       (e.g. filtered by role). If None, uses all champions.
        banned: list of banned champion names to exclude
        top_k: number of recommendations to return

    Returns:
        List of (champion_name, win_probability) sorted descending
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
        score = model.predict(features.reshape(1, -1))[0]

        scored.append((champ, float(score)))

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

    Args:
        models_dir: path to data/processed/draft_models/

    Returns:
        (model, embed_dict, champ_scores) tuple
    """
    from src.models.draft_classifier import load_draft_model

    model = load_draft_model(models_dir / "draft_model.joblib")

    # Load embeddings
    data = np.load(str(models_dir / "champion2vec.npz"), allow_pickle=True)
    embed_weights = data["weights"]
    vocab = data["vocab"].tolist()  # list of champion names
    embed_dict = {name: embed_weights[i] for i, name in enumerate(vocab)}

    # Load champ scores
    with open(models_dir / "champ_scores.json", "r") as f:
        champ_scores = json.load(f)

    return model, embed_dict, champ_scores


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

    Note: DRAFT_ORDER uses capitalized side strings ("Blue"/"Red"); side comparisons
    use .lower() to be case-insensitive. slot is an integer index (0-4).
    """
    side_raw, slot = DRAFT_ORDER[step]
    side = side_raw.lower()  # normalise to lowercase for comparisons

    # Determine ally/enemy lists for the candidate's own perspective
    if side == "blue":
        my_picks, opp_picks = blue_picks, red_picks
    else:
        my_picks, opp_picks = red_picks, blue_picks
    allies = [p for p in my_picks if p]
    enemies = [p for p in opp_picks if p]

    # 1. Build legal candidate pool
    pool = _legal_pool(candidate_pool, embed_dict, banned, blue_picks, red_picks)
    if not pool:
        return []

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
            (trial_blue if side == "blue" else trial_red)[slot] = name
            # Use predict_side_win_prob so both the real adapter and test stubs work
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

        # Always-present score-heuristic prob (frontend uses for "Heuristic" toggle)
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
