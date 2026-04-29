"""
Data quality filters: removes remakes, early surrenders, non-ranked queues,
and rows with invalid positions.
"""

import pandas as pd

VALID_POSITIONS = {"TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"}


def remove_remakes(df: pd.DataFrame, min_duration: int = 180) -> pd.DataFrame:
    """Drop games shorter than min_duration seconds (remakes)."""
    return df[df["game_duration"] >= min_duration].reset_index(drop=True)


def remove_early_surrenders(df: pd.DataFrame, min_duration: int = 900) -> pd.DataFrame:
    """Drop games shorter than min_duration seconds (15-min surrenders)."""
    return df[df["game_duration"] >= min_duration].reset_index(drop=True)


def filter_ranked_solo(df: pd.DataFrame, queue_id: int = 420) -> pd.DataFrame:
    """Keep only Ranked Solo/Duo games."""
    return df[df["queue_id"] == queue_id].reset_index(drop=True)


def validate_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where any participant's teamPosition is missing or invalid."""
    position_cols = [f"p{i}_position" for i in range(1, 11) if f"p{i}_position" in df.columns]
    mask = pd.Series(True, index=df.index)
    for col in position_cols:
        mask &= df[col].isin(VALID_POSITIONS)
    return df[mask].reset_index(drop=True)


def apply_all_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Run all filters in sequence and return the cleaned DataFrame."""
    df = remove_remakes(df)
    df = remove_early_surrenders(df)
    df = filter_ranked_solo(df)
    df = validate_positions(df)
    return df
