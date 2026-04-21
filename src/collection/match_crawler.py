"""
Seeds ranked players from league-v4, crawls their match histories,
deduplicates match IDs, fetches full match data, and writes raw JSON to disk.
"""

import gzip
import json
import glob
import logging
import os
import time

import yaml
from dotenv import load_dotenv

from src.collection.riot_api import (
    get_client,
    get_ranked_entries,
    get_puuid,
    get_match_ids,
    get_match_data,
    get_match_timeline,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CHECKPOINT_PATH = "data/raw/.crawler_checkpoint.json"


# ── helpers ──────────────────────────────────────────────────────────────────


def _load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _load_checkpoint() -> dict:
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return {
        "phase": "seed",
        "puuids": [],
        "match_ids": [],
        "fetched_ids": [],
        "failures": [],
    }


def _save_checkpoint(ckpt: dict) -> None:
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(ckpt, f)


def _api_call_with_retry(fn, *args, max_retries: int = 3, **kwargs):
    """Call *fn* with retries on 429/5xx. Raises on 403 (expired key) or 404."""
    from requests.exceptions import HTTPError

    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else 0
            if status == 403:
                logger.error("403 Forbidden — API key likely expired. Exiting.")
                raise
            if status == 404:
                logger.warning("404 Not Found — skipping.")
                return None
            if status == 429 or status >= 500:
                wait = 2 ** attempt * 5
                logger.warning("HTTP %d, retrying in %ds (attempt %d/%d)", status, wait, attempt + 1, max_retries)
                time.sleep(wait)
            else:
                raise
        except Exception:
            wait = 2 ** attempt * 5
            logger.warning("Unexpected error, retrying in %ds (attempt %d/%d)", wait, attempt + 1, max_retries)
            time.sleep(wait)
    logger.error("Max retries exceeded for %s", fn.__name__)
    return None


# ── pipeline stages ──────────────────────────────────────────────────────────


def seed_players(region: str, tier: str, divisions: list[str]) -> list[str]:
    """Collect PUUIDs from ranked ladder entries across divisions."""
    client = get_client()
    ckpt = _load_checkpoint()

    if ckpt["phase"] != "seed" or ckpt["puuids"]:
        logger.info("Resuming seed phase with %d existing PUUIDs", len(ckpt["puuids"]))
        existing_puuids = set(ckpt["puuids"])
    else:
        existing_puuids = set()

    puuids = list(existing_puuids)

    try:
        for div in divisions:
            logger.info("Fetching league entries for %s %s …", tier, div)
            entries = _api_call_with_retry(get_ranked_entries, client, region, tier, div)
            if entries is None:
                continue
            logger.info("  Got %d entries for %s %s", len(entries), tier, div)
            if entries:
                logger.info("  Entry keys: %s", list(entries[0].keys()))

            for i, entry in enumerate(entries):
                # Modern league-v4 includes puuid directly; fall back to summoner lookup
                puuid = entry.get("puuid")
                if not puuid:
                    sid = entry.get("summonerId")
                    if not sid:
                        logger.warning("Entry missing both puuid and summonerId, skipping")
                        continue
                    puuid = _api_call_with_retry(get_puuid, client, region, sid)
                    if puuid is None:
                        ckpt["failures"].append({"type": "puuid", "summonerId": sid})
                        continue

                if puuid not in existing_puuids:
                    puuids.append(puuid)
                    existing_puuids.add(puuid)

                if (i + 1) % 100 == 0:
                    ckpt["puuids"] = puuids
                    _save_checkpoint(ckpt)
                    logger.info("  Checkpoint: %d PUUIDs collected", len(puuids))
    except Exception:
        logger.error("seed_players interrupted — saving checkpoint with %d PUUIDs", len(puuids))
        raise
    finally:
        ckpt["puuids"] = puuids
        _save_checkpoint(ckpt)

    ckpt["phase"] = "crawl"
    _save_checkpoint(ckpt)
    logger.info("Seed complete: %d PUUIDs", len(puuids))
    return puuids


def crawl_matches(puuids: list[str], max_per_player: int = 50, max_match_ids: int = 60_000) -> set[str]:
    """Fetch match IDs for each PUUID and return a deduplicated set."""
    client = get_client()
    ckpt = _load_checkpoint()
    config = _load_config()
    region = config["riot_api"]["region"]
    queue_id = config["riot_api"]["queue_id"]

    existing_ids = set(ckpt.get("match_ids", []))

    try:
        for i, puuid in enumerate(puuids):
            if len(existing_ids) >= max_match_ids:
                logger.info("Reached %d match IDs (target: %d) — moving to fetch phase", len(existing_ids), max_match_ids)
                break

            ids = _api_call_with_retry(
                get_match_ids, client, region, puuid,
                count=max_per_player, queue=queue_id,
            )
            if ids:
                existing_ids.update(ids)

            if (i + 1) % 100 == 0:
                ckpt["match_ids"] = list(existing_ids)
                _save_checkpoint(ckpt)
                logger.info("  Checkpoint: %d unique match IDs after %d players", len(existing_ids), i + 1)
    except Exception:
        logger.error("crawl_matches interrupted — saving checkpoint with %d match IDs", len(existing_ids))
        raise
    finally:
        ckpt["match_ids"] = list(existing_ids)
        _save_checkpoint(ckpt)

    ckpt["phase"] = "fetch"
    _save_checkpoint(ckpt)
    logger.info("Crawl complete: %d unique match IDs", len(existing_ids))
    return existing_ids


def _write_json_gz(path: str, data: dict) -> None:
    """Write a dict as gzipped JSON."""
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(data, f)


def _scan_match_ids(directory: str, pattern: str = "*.json*") -> set[str]:
    """Scan directory for match IDs, handling both .json and .json.gz files."""
    ids = set()
    for fp in glob.glob(os.path.join(directory, pattern)):
        basename = os.path.basename(fp).replace(".json.gz", "").replace(".json", "")
        ids.add(basename)
    return ids


def _compress_existing(directory: str) -> int:
    """Compress any uncompressed .json files to .json.gz and remove originals."""
    compressed = 0
    for fp in glob.glob(os.path.join(directory, "*.json")):
        gz_path = fp + ".gz"
        if not os.path.exists(gz_path):
            with open(fp, "rb") as f_in:
                with gzip.open(gz_path, "wb") as f_out:
                    f_out.write(f_in.read())
        os.remove(fp)
        compressed += 1
    return compressed


def fetch_and_store(match_ids: set[str], output_dir: str = "data/raw", max_fetches: int = 50_000) -> int:
    """Download full match JSON + timeline for each ID and save gzipped to output_dir. Returns count saved."""
    client = get_client()
    config = _load_config()
    region = config["riot_api"]["region"]
    min_duration = config["collection"]["min_game_duration"]
    min_remake = config["collection"]["min_remake_duration"]
    fetch_timelines = config["collection"].get("fetch_timelines", True)

    os.makedirs(output_dir, exist_ok=True)
    timeline_dir = os.path.join(output_dir, "timelines")
    if fetch_timelines:
        os.makedirs(timeline_dir, exist_ok=True)

    # Compress any uncompressed files from previous runs to reclaim disk
    n = _compress_existing(output_dir)
    if n:
        logger.info("Compressed %d existing .json files to .json.gz", n)

    # Scan existing files to skip already-downloaded matches (.json and .json.gz)
    existing_files = _scan_match_ids(output_dir)

    to_fetch = [mid for mid in match_ids if mid not in existing_files]
    if len(to_fetch) > max_fetches:
        logger.info("Capping fetch list from %d to %d matches", len(to_fetch), max_fetches)
        to_fetch = to_fetch[:max_fetches]
    logger.info("Fetching %d matches (%d already on disk)", len(to_fetch), len(existing_files))

    ckpt = _load_checkpoint()
    saved = 0
    processed_ids = set()

    try:
        for i, match_id in enumerate(to_fetch):
            data = _api_call_with_retry(get_match_data, client, region, match_id)
            processed_ids.add(match_id)
            if data is None:
                ckpt["failures"].append({"type": "match", "matchId": match_id})
                continue

            # Skip remakes and non-ranked games
            duration = data.get("info", {}).get("gameDuration", 0)
            if duration < min_duration:
                if duration < min_remake:
                    logger.debug("Skipping remake %s (duration=%ds)", match_id, duration)
                else:
                    logger.debug("Skipping short game %s (duration=%ds)", match_id, duration)
                continue

            queue = data.get("info", {}).get("queueId", 0)
            if queue != config["riot_api"]["queue_id"]:
                continue

            _write_json_gz(os.path.join(output_dir, f"{match_id}.json.gz"), data)
            saved += 1

            # Fetch timeline (separate API call)
            if fetch_timelines:
                tl_exists = (
                    os.path.exists(os.path.join(timeline_dir, f"{match_id}.json.gz"))
                    or os.path.exists(os.path.join(timeline_dir, f"{match_id}.json"))
                )
                if not tl_exists:
                    timeline = _api_call_with_retry(get_match_timeline, client, region, match_id)
                    if timeline is not None:
                        _write_json_gz(os.path.join(timeline_dir, f"{match_id}.json.gz"), timeline)
                    else:
                        ckpt["failures"].append({"type": "timeline", "matchId": match_id})

            if (i + 1) % 200 == 0:
                ckpt["fetched_ids"] = list(existing_files | processed_ids)
                _save_checkpoint(ckpt)
                logger.info("  Checkpoint: %d saved, %d/%d processed", saved, i + 1, len(to_fetch))
    except Exception:
        logger.error("fetch_and_store interrupted — %d matches already saved to disk", saved)
        raise
    finally:
        ckpt["fetched_ids"] = list(existing_files | processed_ids)
        _save_checkpoint(ckpt)

    ckpt["phase"] = "done"
    _save_checkpoint(ckpt)
    logger.info("Fetch complete: %d matches saved", saved)
    return saved


def backfill_timelines(output_dir: str = "data/raw", max_fetches: int = 50_000) -> int:
    """Fetch timelines for matches that were collected without them."""
    client = get_client()
    config = _load_config()
    region = config["riot_api"]["region"]

    timeline_dir = os.path.join(output_dir, "timelines")
    os.makedirs(timeline_dir, exist_ok=True)

    # Find matches missing timelines (check both .json and .json.gz)
    match_ids = _scan_match_ids(output_dir)
    existing_timelines = _scan_match_ids(timeline_dir)

    to_backfill = [mid for mid in match_ids if mid not in existing_timelines]

    if len(to_backfill) > max_fetches:
        to_backfill = to_backfill[:max_fetches]

    logger.info("Backfilling timelines: %d matches need timelines (%d already have them)",
                len(to_backfill), len(existing_timelines))

    fetched = 0
    for i, match_id in enumerate(to_backfill):
        timeline = _api_call_with_retry(get_match_timeline, client, region, match_id)
        if timeline is not None:
            _write_json_gz(os.path.join(timeline_dir, f"{match_id}.json.gz"), timeline)
            fetched += 1

        if (i + 1) % 200 == 0:
            logger.info("  Timeline backfill: %d/%d fetched", fetched, i + 1)

    logger.info("Timeline backfill complete: %d fetched", fetched)
    return fetched


# ── orchestrator ─────────────────────────────────────────────────────────────


def run_pipeline(config_path: str = "configs/config.yaml") -> None:
    """End-to-end: seed → crawl → fetch → store → backfill timelines."""
    load_dotenv()
    config = _load_config(config_path)

    riot = config["riot_api"]
    collection = config["collection"]
    region = riot["region"]
    tier = riot["tier"]
    divisions = riot["divisions"]

    ckpt = _load_checkpoint()

    # Phase 1: Seed players
    if ckpt["phase"] in ("seed",):
        logger.info("═══ Phase 1: Seeding players ═══")
        puuids = seed_players(region, tier, divisions)
    else:
        puuids = ckpt["puuids"]
        logger.info("Skipping seed phase (%d PUUIDs from checkpoint)", len(puuids))

    # Phase 2: Crawl match IDs
    if ckpt["phase"] in ("seed", "crawl"):
        ckpt = _load_checkpoint()  # re-read after seed
    if ckpt["phase"] in ("crawl",):
        logger.info("═══ Phase 2: Crawling match IDs ═══")
        match_ids = crawl_matches(puuids, max_per_player=collection["max_matches_per_player"])
    else:
        match_ids = set(ckpt.get("match_ids", []))
        logger.info("Skipping crawl phase (%d match IDs from checkpoint)", len(match_ids))

    # Phase 3: Fetch and store
    if ckpt["phase"] in ("crawl", "fetch"):
        ckpt = _load_checkpoint()  # re-read after crawl
    if ckpt["phase"] in ("fetch",):
        logger.info("═══ Phase 3: Fetching match data + timelines ═══")
        saved = fetch_and_store(match_ids)
        logger.info("Pipeline complete — %d matches saved to data/raw/", saved)
    elif ckpt["phase"] == "done":
        logger.info("Pipeline already complete — running timeline backfill.")
        backfill_timelines()
    else:
        logger.info("═══ Phase 3: Fetching match data + timelines ═══")
        saved = fetch_and_store(match_ids)
        logger.info("Pipeline complete — %d matches saved to data/raw/", saved)

    # Always attempt timeline backfill for previously collected matches
    ckpt = _load_checkpoint()
    if ckpt["phase"] == "done" and collection.get("fetch_timelines", True):
        backfill_timelines()


if __name__ == "__main__":
    run_pipeline()
