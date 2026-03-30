"""
Team-level composition features: per-champion win rates, role-specific stats,
pairwise synergy/counter matrices, and ban information.
"""

import numpy as np
import pandas as pd


def champion_win_rates(df: pd.DataFrame) -> dict[int, float]:
    """Compute win rate for each champion from the training set only."""
    raise NotImplementedError


def synergy_matrix(df: pd.DataFrame, num_champions: int = 170) -> np.ndarray:
    """Build an NxN matrix of ally-ally pairwise win rates."""
    raise NotImplementedError


def counter_matrix(df: pd.DataFrame, num_champions: int = 170) -> np.ndarray:
    """Build an NxN matrix of ally-vs-enemy pairwise win rates."""
    raise NotImplementedError


def ban_features(bans: list[int], num_champions: int = 170) -> np.ndarray:
    """Encode team bans as a binary vector."""
    raise NotImplementedError


def build_comp_features(row: pd.Series, win_rates: dict, synergy: np.ndarray, counter: np.ndarray) -> np.ndarray:
    """Combine all team-comp features for a single match row."""
    raise NotImplementedError
