"""
Thin wrapper around the Riot match-v5 / league-v4 / summoner-v4 endpoints.
"""

import os
from riotwatcher import LolWatcher

# Match-v5 uses regional routing, not platform routing
PLATFORM_TO_REGION = {
    "na1": "americas",
    "br1": "americas",
    "la1": "americas",
    "la2": "americas",
    "euw1": "europe",
    "eun1": "europe",
    "tr1": "europe",
    "ru": "europe",
    "kr": "asia",
    "jp1": "asia",
    "oc1": "sea",
    "ph2": "sea",
    "sg2": "sea",
    "th2": "sea",
    "tw2": "sea",
    "vn2": "sea",
}


def get_client() -> LolWatcher:
    """Return an authenticated LolWatcher client using RIOT_API_KEY from env."""
    api_key = os.environ.get("RIOT_API_KEY")
    if not api_key:
        raise ValueError("RIOT_API_KEY environment variable is not set")
    return LolWatcher(api_key)


def get_ranked_entries(client: LolWatcher, region: str, tier: str, division: str) -> list[dict]:
    """Fetch all summoner entries for a given ranked tier/division."""
    entries = []
    page = 1
    while True:
        page_entries = client.league.entries(region, "RANKED_SOLO_5x5", tier, division, page=page)
        if not page_entries:
            break
        entries.extend(page_entries)
        page += 1
    return entries


def get_puuid(client: LolWatcher, region: str, summoner_id: str) -> str:
    """Convert an encrypted summoner ID to a PUUID."""
    summoner = client.summoner.by_id(region, summoner_id)
    return summoner["puuid"]


def get_match_ids(client: LolWatcher, region: str, puuid: str, count: int = 50, queue: int = 420) -> list[str]:
    """Return recent ranked match IDs for a given PUUID."""
    regional = PLATFORM_TO_REGION.get(region, "americas")
    return client.match.matchlist_by_puuid(regional, puuid, queue=queue, count=count)


def get_match_data(client: LolWatcher, region: str, match_id: str) -> dict:
    """Fetch the full match-v5 DTO for a single match."""
    regional = PLATFORM_TO_REGION.get(region, "americas")
    return client.match.by_id(regional, match_id)
