"""
Data quality filters: removes remakes, early surrenders, non-ranked queues,
and rows with invalid positions.
"""

import pandas as pd


def remove_remakes(df: pd.DataFrame, min_duration: int = 180) -> pd.DataFrame:
    """Drop games shorter than min_duration seconds (remakes)."""
    raise NotImplementedError


def remove_early_surrenders(df: pd.DataFrame, min_duration: int = 900) -> pd.DataFrame:
    """Drop games shorter than min_duration seconds (15-min surrenders)."""
    raise NotImplementedError


def filter_ranked_solo(df: pd.DataFrame, queue_id: int = 420) -> pd.DataFrame:
    """Keep only Ranked Solo/Duo games."""
    raise NotImplementedError


def validate_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where teamPosition is missing or invalid."""
    raise NotImplementedError


def apply_all_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Run all filters in sequence and return the cleaned DataFrame."""
    raise NotImplementedError
