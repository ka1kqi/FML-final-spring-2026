"""
Thin wrapper around the Riot match-v5 / league-v4 / summoner-v4 endpoints.
"""

import os
from riotwatcher import LolWatcher


def get_client() -> LolWatcher:
    """Return an authenticated LolWatcher client using RIOT_API_KEY from env."""
    raise NotImplementedError


def get_ranked_entries(client: LolWatcher, region: str, tier: str, division: str) -> list[dict]:
    """Fetch all summoner entries for a given ranked tier/division."""
    raise NotImplementedError


def get_puuid(client: LolWatcher, region: str, summoner_id: str) -> str:
    """Convert an encrypted summoner ID to a PUUID."""
    raise NotImplementedError


def get_match_ids(client: LolWatcher, region: str, puuid: str, count: int = 50, queue: int = 420) -> list[str]:
    """Return recent ranked match IDs for a given PUUID."""
    raise NotImplementedError


def get_match_data(client: LolWatcher, region: str, match_id: str) -> dict:
    """Fetch the full match-v5 DTO for a single match."""
    raise NotImplementedError
