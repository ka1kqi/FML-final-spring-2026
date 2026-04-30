"""
Synergy and counter features computed from Champion2Vec embeddings.

The 64-d champion embedding is the concatenation of four 16-d blocks
produced by matrix factorization:

    [ U_syn (16) | V_syn (16) | U_match (16) | V_match (16) ]

with the trained property

    S[i, j] (centered ally score) ≈ U_syn[i] · V_syn[j]
    M[i, j] (centered matchup score) ≈ U_match[i] · V_match[j]

So predicted synergy of candidate c with ally a is the *cross-block*
dot product U_syn[c] · V_syn[a]. Cosine similarity of the full
embeddings does NOT recover this — it mixes same-block terms that
encode archetype similarity, not synergy.
"""

import numpy as np
import pandas as pd


def _split_blocks(vec, embed_dim=64):
    """Slice a flat embedding into (U_syn, V_syn, U_match, V_match)."""
    q = embed_dim // 4
    return vec[0:q], vec[q:2*q], vec[2*q:3*q], vec[3*q:4*q]


def predicted_synergy(candidate_vec, ally_vec, embed_dim=64):
    """
    Asymmetric MF-predicted synergy: candidate's centered score boost
    when `ally_vec`'s champion is on the same team.

        synergy(c, a) = U_syn[c] · V_syn[a]
    """
    u_syn_c, _, _, _ = _split_blocks(candidate_vec, embed_dim)
    _, v_syn_a, _, _ = _split_blocks(ally_vec, embed_dim)
    return float(np.dot(u_syn_c, v_syn_a))


def predicted_counter(candidate_vec, enemy_vec, embed_dim=64):
    """
    Asymmetric MF-predicted matchup: candidate's centered score
    when facing `enemy_vec`'s champion. Positive = candidate counters.

        counter(c, e) = U_match[c] · V_match[e]
    """
    _, _, u_match_c, _ = _split_blocks(candidate_vec, embed_dim)
    _, _, _, v_match_e = _split_blocks(enemy_vec, embed_dim)
    return float(np.dot(u_match_c, v_match_e))


def ally_synergy_score(candidate_vec, ally_vecs, embed_dim=64):
    """
    Mean MF-predicted synergy across current allies. Returns 0.0 when
    no allies are picked yet (matches the centered-matrix prior).
    """
    if not ally_vecs:
        return 0.0
    sims = [predicted_synergy(candidate_vec, av, embed_dim) for av in ally_vecs]
    return float(np.mean(sims))


def enemy_counter_score(candidate_vec, enemy_vecs, embed_dim=64):
    """
    Mean MF-predicted matchup score across current enemies.
    Positive = candidate is favored vs. the enemy team on average.
    """
    if not enemy_vecs:
        return 0.0
    sims = [predicted_counter(candidate_vec, ev, embed_dim) for ev in enemy_vecs]
    return float(np.mean(sims))


def compute_champion_scores(comp_df):
    """
    Compute per-champion average composition scores from historical match data.

    Args:
        comp_df: DataFrame with columns [champion_name, champ_score, ...]

    Returns:
        dict[str, float] — champion_name -> average champ_score
    """
    grouped = comp_df.groupby("champion_name")["champ_score"].agg(["sum", "count"])
    champ_scores = {}
    for name, row in grouped.iterrows():
        champ_scores[name] = float(row["sum"] / row["count"]) if row["count"] > 0 else 50.0
    return champ_scores


def build_candidate_features(candidate_name, ally_names, enemy_names,
                             embed_dict, champ_scores, embed_dim=64):
    """
    Build a feature vector for a candidate champion given the current draft state.

    Feature layout (197 dimensions total):
      [0:64]    candidate's own embedding vector
      [64:128]  mean embedding of current allies (zeros if none)
      [128:192] mean embedding of current enemies (zeros if none)
      [192]     ally synergy score = mean(U_syn[c] · V_syn[a]) over allies
      [193]     enemy counter score = mean(U_match[c] · V_match[e]) over enemies
      [194]     candidate's historical comp score (0-100)
      [195]     number of allies already picked (0-4)
      [196]     number of enemies already picked (0-4)

    The synergy/counter scalars are the MF model's own predictions of
    centered score contribution — not cosine similarity, which would
    measure archetype overlap rather than interaction quality.
    """
    candidate_vec = embed_dict[candidate_name]

    ally_vecs = [embed_dict[n] for n in ally_names if n in embed_dict]
    if ally_vecs:
        ally_mean = np.mean(ally_vecs, axis=0)
    else:
        ally_mean = np.zeros(embed_dim, dtype=np.float32)

    enemy_vecs = [embed_dict[n] for n in enemy_names if n in embed_dict]
    if enemy_vecs:
        enemy_mean = np.mean(enemy_vecs, axis=0)
    else:
        enemy_mean = np.zeros(embed_dim, dtype=np.float32)

    synergy = ally_synergy_score(candidate_vec, ally_vecs, embed_dim)
    counter = enemy_counter_score(candidate_vec, enemy_vecs, embed_dim)

    wr = champ_scores.get(candidate_name, 50.0)

    num_allies = float(len(ally_names))
    num_enemies = float(len(enemy_names))

    features = np.concatenate([
        candidate_vec,          # 64
        ally_mean,              # 64
        enemy_mean,             # 64
        np.array([synergy, counter, wr, num_allies, num_enemies], dtype=np.float64),
    ])

    return features.astype(np.float32)
