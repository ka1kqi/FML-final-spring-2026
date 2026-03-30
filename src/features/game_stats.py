"""
In-game player stat features: gold earned, total damage dealt,
damage taken, vision score, CS, and per-team aggregates.
"""

import numpy as np
import pandas as pd

STAT_COLUMNS = [
    "goldEarned",
    "totalDamageDealtToChampions",
    "totalDamageTaken",
    "visionScore",
    "totalMinionsKilled",
    "neutralMinionsKilled",
]


def extract_player_stats(participant: dict) -> dict:
    """Pull stat fields from a single participant dict."""
    raise NotImplementedError


def aggregate_team_stats(team_stats: list[dict]) -> dict:
    """Sum/average player stats across a 5-player team."""
    raise NotImplementedError


def build_stat_features(row: pd.Series) -> np.ndarray:
    """Return a feature vector of per-team aggregated stats for a match."""
    raise NotImplementedError
