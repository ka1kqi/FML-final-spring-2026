"""
Seeds ranked players from league-v4, crawls their match histories,
deduplicates match IDs, fetches full match data, and writes raw JSON to disk.
"""

import pandas as pd


def seed_players(region: str, tier: str, divisions: list[str]) -> list[str]:
    """Collect PUUIDs from ranked ladder entries across divisions."""
    raise NotImplementedError


def crawl_matches(puuids: list[str], max_per_player: int = 50) -> set[str]:
    """Fetch match IDs for each PUUID and return a deduplicated set."""
    raise NotImplementedError


def fetch_and_store(match_ids: set[str], output_dir: str = "data/raw") -> int:
    """Download full match JSON for each ID and save to output_dir. Returns count saved."""
    raise NotImplementedError


def run_pipeline(config_path: str = "configs/config.yaml") -> None:
    """End-to-end: seed → crawl → fetch → store."""
    raise NotImplementedError
