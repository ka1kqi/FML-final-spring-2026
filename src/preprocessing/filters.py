"""
Data quality filters: removes remakes, early surrenders, non-ranked queues,
and rows with invalid positions.
"""

import pandas as pd


VALID_POSITIONS = {"TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"}


def remove_remakes(df: pd.DataFrame, min_duration: int = 180) -> pd.DataFrame:
    """Drop games shorter than min_duration seconds (remakes)."""
    if "game_duration" not in df.columns:
        return df.copy()
    durations = pd.to_numeric(df["game_duration"], errors="coerce")
    return df[durations >= int(min_duration)].copy()


def remove_early_surrenders(df: pd.DataFrame, min_duration: int = 900) -> pd.DataFrame:
    """Drop games shorter than min_duration seconds (15-min surrenders)."""
    filtered = df
    if "game_duration" in filtered.columns:
        durations = pd.to_numeric(filtered["game_duration"], errors="coerce")
        filtered = filtered[durations >= int(min_duration)]

    if "game_ended_in_early_surrender" in filtered.columns:
        raw = filtered["game_ended_in_early_surrender"].fillna(False)
        if pd.api.types.is_bool_dtype(raw):
            early_surrender = raw.astype(bool)
        else:
            early_surrender = (
                raw.astype(str)
                .str.strip()
                .str.lower()
                .isin({"1", "true", "yes", "y", "on"})
            )
        filtered = filtered[~early_surrender]

    return filtered.copy()


def filter_ranked_solo(df: pd.DataFrame, queue_id: int = 420) -> pd.DataFrame:
    """Keep only Ranked Solo/Duo games."""
    if "queue_id" not in df.columns:
        return df.copy()
    queues = pd.to_numeric(df["queue_id"], errors="coerce")
    return df[queues == int(queue_id)].copy()


def validate_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where teamPosition is missing or invalid."""
    position_col = "position" if "position" in df.columns else "team_position"
    if position_col not in df.columns:
        return df.copy()

    filtered = df.copy()
    filtered[position_col] = filtered[position_col].fillna("UNKNOWN").astype(str).str.upper()
    return filtered[filtered[position_col].isin(VALID_POSITIONS)].copy()


def apply_all_filters(
    df: pd.DataFrame,
    queue_id: int = 420,
    min_game_duration: int = 900,
    min_remake_duration: int = 180,
    require_valid_position: bool = True,
) -> pd.DataFrame:
    """Run all filters in sequence and return the cleaned DataFrame."""
    filtered = remove_remakes(df, min_duration=min_remake_duration)
    filtered = remove_early_surrenders(filtered, min_duration=min_game_duration)
    filtered = filter_ranked_solo(filtered, queue_id=queue_id)
    if require_valid_position:
        filtered = validate_positions(filtered)
    return filtered.reset_index(drop=True)
