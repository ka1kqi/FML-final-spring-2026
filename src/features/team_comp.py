"""
Team-level composition features: per-champion win rates, role-specific stats,
pairwise synergy/counter matrices, and ban information.
"""

import numpy as np
import pandas as pd


def champion_win_rates(df: pd.DataFrame) -> dict[int, float]:
    """Compute win rate for each champion from the training set only."""
    win_counts = {}
    game_counts = {}

    for i in range(1, 11):
        champ_col = f"p{i}_champion"
        win_col = f"p{i}_win"
        if champ_col not in df.columns or win_col not in df.columns:
            continue
        for champ, win in zip(df[champ_col], df[win_col]):
            if pd.isna(champ):
                continue
            champ = int(champ)
            win_counts[champ] = win_counts.get(champ, 0) + int(win)
            game_counts[champ] = game_counts.get(champ, 0) + 1

    return {
        champ: win_counts[champ] / game_counts[champ]
        for champ in game_counts
        if game_counts[champ] > 0
    }


def synergy_matrix(df: pd.DataFrame, num_champions: int = 170) -> np.ndarray:
    """Build an NxN matrix of ally-ally pairwise win rates."""
    win_counts = np.zeros((num_champions, num_champions))
    game_counts = np.zeros((num_champions, num_champions))

    for _, row in df.iterrows():
        # Get blue and red team champions separately
        for team_id, player_range in [(100, range(1, 6)), (200, range(6, 11))]:
            champs = []
            win = None
            for i in player_range:
                champ = row.get(f"p{i}_champion")
                w = row.get(f"p{i}_win")
                if pd.isna(champ):
                    continue
                champ = int(champ)
                if champ < num_champions:
                    champs.append(champ)
                if win is None and w is not None:
                    win = int(w)

            if win is None:
                continue

            # Update pairwise counts for all ally pairs
            for j in range(len(champs)):
                for k in range(j + 1, len(champs)):
                    a, b = champs[j], champs[k]
                    win_counts[a][b] += win
                    win_counts[b][a] += win
                    game_counts[a][b] += 1
                    game_counts[b][a] += 1

    # Avoid division by zero
    with np.errstate(invalid="ignore", divide="ignore"):
        matrix = np.where(game_counts > 0, win_counts / game_counts, 0.5)

    return matrix


def counter_matrix(df: pd.DataFrame, num_champions: int = 170) -> np.ndarray:
    """Build an NxN matrix of ally-vs-enemy pairwise win rates."""
    win_counts = np.zeros((num_champions, num_champions))
    game_counts = np.zeros((num_champions, num_champions))

    for _, row in df.iterrows():
        blue_champs, blue_win = [], None
        red_champs, red_win = [], None

        for i in range(1, 11):
            champ = row.get(f"p{i}_champion")
            team = row.get(f"p{i}_team")
            win = row.get(f"p{i}_win")
            if pd.isna(champ):
                continue
            champ = int(champ)
            if champ >= num_champions:
                continue
            if team == 100:
                blue_champs.append(champ)
                if blue_win is None and win is not None:
                    blue_win = int(win)
            else:
                red_champs.append(champ)
                if red_win is None and win is not None:
                    red_win = int(win)

        if blue_win is None or red_win is None:
            continue

        # Blue vs Red matchups
        for a in blue_champs:
            for b in red_champs:
                win_counts[a][b] += blue_win
                game_counts[a][b] += 1
                win_counts[b][a] += red_win
                game_counts[b][a] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        matrix = np.where(game_counts > 0, win_counts / game_counts, 0.5)

    return matrix


def ban_features(bans: list[int], num_champions: int = 170) -> np.ndarray:
    """Encode team bans as a binary vector."""
    vec = np.zeros(num_champions)
    for champ_id in bans:
        if champ_id is not None and not (isinstance(champ_id, float) and np.isnan(champ_id)):
            idx = int(champ_id)
            if 0 <= idx < num_champions:
                vec[idx] = 1.0
    return vec


def build_comp_features(
    row: pd.Series,
    win_rates: dict,
    synergy: np.ndarray,
    counter: np.ndarray,
    num_champions: int = 170,
) -> np.ndarray:
    """Combine all team-comp features for a single match row."""
    features = []

    for team_id, player_range, ban_prefix in [
        (100, range(1, 6), "blue"),
        (200, range(6, 11), "red"),
    ]:
        champs = []
        for i in player_range:
            champ = row.get(f"p{i}_champion")
            if champ is not None and not (isinstance(champ, float) and np.isnan(champ)):
                champ = int(champ)
                if champ < num_champions:
                    champs.append(champ)

        # Per-champion win rates (mean for the team)
        wr_feats = [win_rates.get(c, 0.5) for c in champs]
        features.append(np.mean(wr_feats) if wr_feats else 0.5)

        # Mean pairwise synergy
        syn_vals = []
        for j in range(len(champs)):
            for k in range(j + 1, len(champs)):
                syn_vals.append(synergy[champs[j]][champs[k]])
        features.append(np.mean(syn_vals) if syn_vals else 0.5)

        # Bans
        bans = [row.get(f"{ban_prefix}_ban{i}") for i in range(1, 6)]
        features.append(ban_features(bans, num_champions))

    # Counter matchup scores (blue vs red)
    blue_champs = []
    red_champs = []
    for i in range(1, 11):
        champ = row.get(f"p{i}_champion")
        team = row.get(f"p{i}_team")
        if champ is None or (isinstance(champ, float) and np.isnan(champ)):
            continue
        champ = int(champ)
        if champ >= num_champions:
            continue
        if team == 100:
            blue_champs.append(champ)
        else:
            red_champs.append(champ)

    counter_vals = []
    for a in blue_champs:
        for b in red_champs:
            counter_vals.append(counter[a][b])
    features.append(np.mean(counter_vals) if counter_vals else 0.5)

    # Flatten everything into a single vector
    flat = []
    for f in features:
        if isinstance(f, np.ndarray):
            flat.extend(f.tolist())
        else:
            flat.append(f)

    return np.array(flat, dtype=np.float32)