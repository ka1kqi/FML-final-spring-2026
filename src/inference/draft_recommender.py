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
