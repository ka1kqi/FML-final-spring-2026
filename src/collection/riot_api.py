"""
Thin wrapper around the Riot match-v5 / league-v4 / summoner-v4 endpoints.
"""

import os
import time
from typing import Callable, Optional, TypeVar

from riotwatcher import ApiError
from riotwatcher import LolWatcher


T = TypeVar("T")

# Match-v5 uses regional routes, while league/summoner endpoints use platform routes.
_PLATFORM_TO_REGIONAL = {
    "br1": "americas",
    "la1": "americas",
    "la2": "americas",
    "na1": "americas",
    "eun1": "europe",
    "euw1": "europe",
    "tr1": "europe",
    "ru": "europe",
    "jp1": "asia",
    "kr": "asia",
    "oc1": "sea",
    "ph2": "sea",
    "sg2": "sea",
    "th2": "sea",
    "tw2": "sea",
    "vn2": "sea",
}


def _load_key_from_dotenv(dotenv_path: str = ".env") -> Optional[str]:
    """Read RIOT_API_KEY from a local .env file without extra dependencies."""
    if not os.path.exists(dotenv_path):
        return None

    with open(dotenv_path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() != "RIOT_API_KEY":
                continue
            return value.strip().strip('"').strip("'")

    return None


def _to_regional_route(region: str) -> str:
    """Convert platform route (na1/euw1/...) to regional route for match-v5."""
    normalized = region.lower()
    if normalized in {"americas", "europe", "asia", "sea"}:
        return normalized
    return _PLATFORM_TO_REGIONAL.get(normalized, normalized)


def _call_with_rate_limit_retry(call: Callable[[], T], max_attempts: int = 5) -> T:
    """Retry Riot API calls on HTTP 429 using Retry-After when present."""
    for attempt in range(max_attempts):
        try:
            return call()
        except ApiError as exc:
            response = exc.response
            if response is None or response.status_code != 429 or attempt == max_attempts - 1:
                raise

            retry_after = response.headers.get("Retry-After", "1")
            try:
                sleep_seconds = float(retry_after)
            except (TypeError, ValueError):
                sleep_seconds = 1.0
            time.sleep(max(sleep_seconds, 1.0))

    raise RuntimeError("Unreachable retry loop termination.")


def get_client() -> LolWatcher:
    """Return an authenticated LolWatcher client using RIOT_API_KEY from env."""
    api_key = os.getenv("RIOT_API_KEY")
    if not api_key:
        api_key = _load_key_from_dotenv()
        if api_key:
            os.environ["RIOT_API_KEY"] = api_key

    if not api_key:
        raise EnvironmentError("RIOT_API_KEY is not set. Export it or place it in .env.")

    return LolWatcher(api_key)


def get_ranked_entries(client: LolWatcher, region: str, tier: str, division: str) -> list[dict]:
    """Fetch all summoner entries for a given ranked tier/division."""
    platform_route = region.lower()
    queue = "RANKED_SOLO_5x5"

    entries: list[dict] = []
    page = 1
    while True:
        batch = _call_with_rate_limit_retry(
            lambda: client.league.entries(
                platform_route,
                queue,
                tier.upper(),
                division.upper(),
                page=page,
            )
        )
        if not batch:
            break
        entries.extend(batch)
        page += 1

    return entries


def get_puuid(client: LolWatcher, region: str, summoner_id: str) -> str:
    """Convert an encrypted summoner ID to a PUUID."""
    platform_route = region.lower()
    data = _call_with_rate_limit_retry(lambda: client.summoner.by_id(platform_route, summoner_id))
    return data["puuid"]


def get_match_ids(client: LolWatcher, region: str, puuid: str, count: int = 50, queue: int = 420) -> list[str]:
    """Return recent ranked match IDs for a given PUUID."""
    if count <= 0:
        return []

    regional_route = _to_regional_route(region)
    match_type = "ranked" if queue == 420 else None

    match_ids: list[str] = []
    start = 0
    remaining = count

    while remaining > 0:
        batch_size = min(remaining, 100)
        batch = _call_with_rate_limit_retry(
            lambda: client.match.matchlist_by_puuid(
                regional_route,
                puuid,
                start=start,
                count=batch_size,
                queue=queue,
                type=match_type,
            )
        )

        if not batch:
            break

        match_ids.extend(batch)
        if len(batch) < batch_size:
            break

        start += len(batch)
        remaining -= len(batch)

    return match_ids


def get_match_data(client: LolWatcher, region: str, match_id: str) -> dict:
    """Fetch the full match-v5 DTO for a single match."""
    regional_route = _to_regional_route(region)
    return _call_with_rate_limit_retry(lambda: client.match.by_id(regional_route, match_id))
