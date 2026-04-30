"""
In-game player stat features: gold earned, total damage dealt,
damage taken, vision score, CS, and per-team aggregates.
"""

import numpy as np
import pandas as pd

STAT_COLUMNS = [
    "gold",
    "damage",
    "vision",
    "cs",
]


def extract_player_stats(participant: dict) -> dict:
    """Pull stat fields from a single participant dict."""
    return {col: participant.get(col, 0) or 0 for col in STAT_COLUMNS}


def aggregate_team_stats(team_stats: list[dict]) -> dict:
    """Sum/average player stats across a 5-player team."""
    aggregated = {}
    for col in STAT_COLUMNS:
        values = [p.get(col, 0) or 0 for p in team_stats]
        aggregated[f"{col}_sum"] = sum(values)
        aggregated[f"{col}_mean"] = sum(values) / len(values) if values else 0.0
    return aggregated


def build_stat_features(row: pd.Series) -> np.ndarray:
    """Return a feature vector of per-team aggregated stats for a match."""
    features = []

    for team_id, player_range in [(100, range(1, 6)), (200, range(6, 11))]:
        team_stats = []
        for i in player_range:
            participant = {
                col: row.get(f"p{i}_{col}") for col in STAT_COLUMNS
            }
            team_stats.append(extract_player_stats(participant))
        agg = aggregate_team_stats(team_stats)
        features.extend(agg.values())

    return np.array(features, dtype=np.float32)