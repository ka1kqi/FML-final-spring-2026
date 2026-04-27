"""
Synergy and counter features computed from Champion2Vec embeddings.

These features capture how well a candidate champion fits with the
current draft state:
  - ally_synergy: cosine similarity to teammates (high = good fit)
  - enemy_counter: cosine similarity to enemies (high = similar playstyle)
  - Embedding aggregations for team and enemy compositions

All cosine similarity computations are hand-written (no scipy).
"""

import numpy as np
import pandas as pd


def cosine_similarity(vec_a, vec_b):
    """
    Hand-written cosine similarity between two vectors.

    cos(a, b) = (a · b) / (||a|| * ||b||)

    Returns a float in [-1, 1].
    """
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(dot / (norm_a * norm_b))


def ally_synergy_score(candidate_vec, ally_vecs):
    """
    Mean cosine similarity between a candidate champion and current allies.

    Higher score = candidate fits well with the existing team composition.
    Returns 0.0 if there are no allies yet.

    Args:
        candidate_vec: np.ndarray of shape (embed_dim,)
        ally_vecs: list of np.ndarray, one per current ally

    Returns:
        float — average cosine similarity to allies
    """
    if not ally_vecs:
        return 0.0
    sims = [cosine_similarity(candidate_vec, av) for av in ally_vecs]
    return float(np.mean(sims))


def enemy_counter_score(candidate_vec, enemy_vecs):
    """
    Mean cosine similarity between a candidate champion and current enemies.

    In our embedding space, similar champions have similar playstyles.
    A high similarity to enemies could mean the candidate is easily
    countered (same weaknesses). We return raw similarity; the model
    decides how to weight it.

    Returns 0.0 if there are no enemies yet.

    Args:
        candidate_vec: np.ndarray of shape (embed_dim,)
        enemy_vecs: list of np.ndarray, one per current enemy

    Returns:
        float — average cosine similarity to enemies
    """
    if not enemy_vecs:
        return 0.0
    sims = [cosine_similarity(candidate_vec, ev) for ev in enemy_vecs]
    return float(np.mean(sims))


def compute_champion_scores(comp_df):
    """
    Compute per-champion average composition scores from historical match data.

    Args:
        comp_df: DataFrame with columns [champion_name, comp_score, ...]

    Returns:
        dict[str, float] — champion_name -> average comp_score
    """
    grouped = comp_df.groupby("champion_name")["comp_score"].agg(["sum", "count"])
    champ_scores = {}
    for name, row in grouped.iterrows():
        champ_scores[name] = float(row["sum"] / row["count"]) if row["count"] > 0 else 50.0
    return champ_scores


def build_candidate_features(candidate_name, ally_names, enemy_names,
                             embed_dict, champ_scores, embed_dim=64):
    """
    Build a feature vector for a candidate champion given the current draft state.

    Feature layout (196 dimensions total):
      [0:64]    candidate's own embedding vector
      [64:128]  mean embedding of current allies (zeros if none)
      [128:192] mean embedding of current enemies (zeros if none)
      [192]     ally synergy score (avg cosine sim to allies)
      [193]     enemy counter score (avg cosine sim to enemies)
      [194]     candidate's historical comp score (0-100)
      [195]     number of allies already picked (0-4)
      [196]     number of enemies already picked (0-4)

    Args:
        candidate_name: name of the candidate champion
        ally_names: list of champion names already on the same team
        enemy_names: list of champion names on the opposing team
        embed_dict: dict mapping champion names to embedding vectors
        champ_scores: dict mapping champion names to average comp scores
        embed_dim: dimensionality of embeddings (default 64)

    Returns:
        np.ndarray of shape (197,) — the feature vector
    """
    candidate_vec = embed_dict[candidate_name]

    # Ally embeddings
    ally_vecs = [embed_dict[n] for n in ally_names if n in embed_dict]
    if ally_vecs:
        ally_mean = np.mean(ally_vecs, axis=0)
    else:
        ally_mean = np.zeros(embed_dim, dtype=np.float32)

    # Enemy embeddings
    enemy_vecs = [embed_dict[n] for n in enemy_names if n in embed_dict]
    if enemy_vecs:
        enemy_mean = np.mean(enemy_vecs, axis=0)
    else:
        enemy_mean = np.zeros(embed_dim, dtype=np.float32)

    # Synergy and counter scores
    synergy = ally_synergy_score(candidate_vec, ally_vecs)
    counter = enemy_counter_score(candidate_vec, enemy_vecs)

    # Average Comp Score
    wr = champ_scores.get(candidate_name, 50.0)

    # Draft state
    num_allies = float(len(ally_names))
    num_enemies = float(len(enemy_names))

    # Concatenate all features
    features = np.concatenate([
        candidate_vec,          # 64
        ally_mean,              # 64
        enemy_mean,             # 64
        np.array([synergy, counter, wr, num_allies, num_enemies], dtype=np.float64),
    ])

    return features.astype(np.float32)
