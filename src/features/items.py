"""
Item build features: end-game item binary vectors per player,
and team-level item aggregation.
"""

import numpy as np
import pandas as pd

NUM_ITEMS = 250  # approximate number of purchasable items


def player_item_vector(item_ids: list[int], num_items: int = NUM_ITEMS) -> np.ndarray:
    """Binary vector with 1 at each item's index (up to 6 items per player)."""
    raise NotImplementedError


def team_item_vector(team_item_ids: list[list[int]], num_items: int = NUM_ITEMS) -> np.ndarray:
    """Concatenate item vectors for all 5 players on a team."""
    raise NotImplementedError


def build_item_features(row: pd.Series, num_items: int = NUM_ITEMS) -> np.ndarray:
    """Return combined item feature vector for both teams in a match."""
    raise NotImplementedError
