"""
Parses raw match JSON into a flat DataFrame (one row per match).
Extracts match_id, team_id, champions, positions, win, bans,
gold, damage, vision score, items per participant.
"""

import pandas as pd


def parse_match(match_json: dict) -> dict:
    """Extract a flat dict of features from a single match-v5 DTO."""
    raise NotImplementedError


def parse_all(raw_dir: str = "data/raw") -> pd.DataFrame:
    """Load all raw JSON files from raw_dir and return a combined DataFrame."""
    raise NotImplementedError


def save_processed(df: pd.DataFrame, output_path: str = "data/processed/matches.csv") -> None:
    """Write the processed DataFrame to CSV."""
    raise NotImplementedError
