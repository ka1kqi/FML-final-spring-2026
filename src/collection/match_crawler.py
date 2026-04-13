"""
Seeds ranked players from league-v4, crawls their match histories,
deduplicates match IDs, fetches full match data, and persists to raw JSON and/or CSV.
"""

import csv
import json
import os
from pathlib import Path

import pandas as pd
import yaml
from riotwatcher import ApiError
from tqdm import tqdm

from src.collection.riot_api import (
    get_client,
    get_match_data,
    get_match_ids,
    get_puuid,
    get_ranked_entries,
)
from src.preprocessing.parser import OUTPUT_COLUMNS, clean_dataframe, load_processed_csv, parse_match, save_processed


def _runtime_region(default: str = "na1") -> str:
    return os.getenv("RIOT_REGION", default)


def _runtime_queue_id(default: int = 420) -> int:
    raw_value = os.getenv("RIOT_QUEUE_ID", str(default))
    try:
        return int(raw_value)
    except ValueError:
        return default


def _as_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


def _existing_csv_match_ids(csv_path: Path) -> set[str]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()

    match_ids: set[str] = set()
    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as infile:
            reader = csv.DictReader(infile)
            if not reader.fieldnames or "match_id" not in reader.fieldnames:
                return set()

            for row in reader:
                match_id = row.get("match_id")
                if match_id:
                    match_ids.add(match_id)
    except OSError:
        return set()

    return match_ids


def _ensure_output_csv_schema(csv_path: Path) -> None:
    """Migrate existing CSV to the current OUTPUT_COLUMNS schema if needed."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return

    with open(csv_path, "r", encoding="utf-8", newline="") as infile:
        reader = csv.reader(infile)
        header = next(reader, [])

    if header == OUTPUT_COLUMNS:
        return

    normalized_df = load_processed_csv(str(csv_path))
    save_processed(normalized_df, output_path=str(csv_path))


def _load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


def seed_players(region: str, tier: str, divisions: list[str], max_players: int | None = None) -> list[str]:
    """Collect PUUIDs from ranked ladder entries across divisions."""
    client = get_client()
    puuids: set[str] = set()

    for division in tqdm(divisions, desc="Seeding ranked players"):
        entries = get_ranked_entries(client, region=region, tier=tier, division=division)
        for entry in entries:
            puuid = entry.get("puuid")
            if puuid:
                puuids.add(puuid)
            else:
                summoner_id = entry.get("summonerId")
                if not summoner_id:
                    continue

                try:
                    puuids.add(get_puuid(client, region=region, summoner_id=summoner_id))
                except ApiError as exc:
                    if exc.response is not None and exc.response.status_code in {403, 404}:
                        continue
                    raise

            if max_players is not None and len(puuids) >= max_players:
                return sorted(puuids)

    return sorted(puuids)


def crawl_matches(puuids: list[str], max_per_player: int = 50) -> set[str]:
    """Fetch match IDs for each PUUID and return a deduplicated set."""
    client = get_client()
    region = _runtime_region()
    queue_id = _runtime_queue_id()

    match_ids: set[str] = set()
    for puuid in tqdm(puuids, desc="Crawling match IDs"):
        try:
            player_match_ids = get_match_ids(
                client,
                region=region,
                puuid=puuid,
                count=max_per_player,
                queue=queue_id,
            )
            match_ids.update(player_match_ids)
        except ApiError as exc:
            if exc.response is not None and exc.response.status_code in {403, 404}:
                continue
            raise

    return match_ids


def fetch_and_store(
    match_ids: set[str],
    output_dir: str = "data/raw",
    store_raw_json: bool = True,
    csv_output_path: str | None = None,
    queue_id: int = 420,
    min_game_duration: int = 900,
    min_remake_duration: int = 180,
    require_valid_position: bool = True,
) -> tuple[int, int, int]:
    """Download matches and persist to raw JSON and/or participant-level CSV.

    Returns (new_json_count, new_csv_match_count, new_csv_row_count).
    """
    client = get_client()
    region = _runtime_region()

    output_path = Path(output_dir)
    if store_raw_json:
        output_path.mkdir(parents=True, exist_ok=True)

    csv_file = None
    csv_writer = None
    csv_existing_match_ids: set[str] = set()
    csv_path = None
    if csv_output_path:
        csv_path = Path(csv_output_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        _ensure_output_csv_schema(csv_path)
        csv_existing_match_ids = _existing_csv_match_ids(csv_path)
        csv_file = open(csv_path, "a", encoding="utf-8", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_COLUMNS)
        if csv_path.stat().st_size == 0:
            csv_writer.writeheader()

    saved_json_count = 0
    saved_csv_match_count = 0
    saved_csv_row_count = 0
    filtered_out_row_count = 0
    try:
        for match_id in tqdm(sorted(match_ids), desc="Downloading matches"):
            match_file = output_path / f"{match_id}.json"
            has_raw_json = store_raw_json and match_file.exists()
            has_csv_match = csv_writer is not None and match_id in csv_existing_match_ids

            if has_raw_json and (csv_writer is None or has_csv_match):
                continue

            if has_csv_match and not store_raw_json:
                continue

            try:
                match_json = get_match_data(client, region=region, match_id=match_id)
            except ApiError as exc:
                if exc.response is not None and exc.response.status_code in {403, 404}:
                    continue
                raise

            if store_raw_json and not has_raw_json:
                with open(match_file, "w", encoding="utf-8") as outfile:
                    json.dump(match_json, outfile)
                saved_json_count += 1

            if csv_writer is not None and not has_csv_match:
                rows = parse_match(match_json)
                row_df = pd.DataFrame(rows)
                cleaned_df = clean_dataframe(
                    row_df,
                    queue_id=queue_id,
                    min_game_duration=min_game_duration,
                    min_remake_duration=min_remake_duration,
                    require_valid_position=require_valid_position,
                    dedupe=False,
                )
                cleaned_rows = cleaned_df.to_dict(orient="records")

                for row in cleaned_rows:
                    csv_writer.writerow({column: row.get(column) for column in OUTPUT_COLUMNS})

                csv_existing_match_ids.add(match_id)
                saved_csv_match_count += 1
                saved_csv_row_count += len(cleaned_rows)
                filtered_out_row_count += max(len(rows) - len(cleaned_rows), 0)
    finally:
        if csv_file is not None:
            csv_file.close()

    if csv_writer is not None and filtered_out_row_count > 0:
        print(f"Filtered out {filtered_out_row_count} participant rows during crawl-time cleaning.")

    return saved_json_count, saved_csv_match_count, saved_csv_row_count


def run_pipeline(config_path: str = "configs/config.yaml") -> None:
    """End-to-end: seed → crawl → fetch → persist."""
    config = _load_config(config_path)

    riot_cfg = config.get("riot_api", {})
    collection_cfg = config.get("collection", {})

    region = riot_cfg.get("region", "na1")
    tier = riot_cfg.get("tier", "DIAMOND")
    divisions = riot_cfg.get("divisions", ["I", "II", "III", "IV"])
    queue_id = int(riot_cfg.get("queue_id", 420))

    max_matches = int(collection_cfg.get("max_matches_per_player", 50))
    max_players_raw = collection_cfg.get("max_seed_players")
    max_players = int(max_players_raw) if max_players_raw is not None else None
    if max_players is not None and max_players <= 0:
        max_players = None

    store_raw_json = _as_bool(collection_cfg.get("store_raw_json", True), default=True)
    csv_output_path = collection_cfg.get("csv_output_path", "data/processed/compositions_stats.csv")
    if csv_output_path is not None and not str(csv_output_path).strip():
        csv_output_path = None

    min_game_duration = int(collection_cfg.get("min_game_duration", 900))
    min_remake_duration = int(collection_cfg.get("min_remake_duration", 180))
    require_valid_position = _as_bool(collection_cfg.get("require_valid_position", True), default=True)

    output_dir = collection_cfg.get("raw_output_dir", "data/raw")

    # Persist runtime defaults for helper functions that do not accept region/queue args.
    os.environ["RIOT_REGION"] = str(region)
    os.environ["RIOT_QUEUE_ID"] = str(queue_id)

    puuids = seed_players(
        region=region,
        tier=tier,
        divisions=list(divisions),
        max_players=max_players,
    )
    match_ids = crawl_matches(puuids=puuids, max_per_player=max_matches)
    saved_json_count, saved_csv_match_count, saved_csv_row_count = fetch_and_store(
        match_ids=match_ids,
        output_dir=output_dir,
        store_raw_json=store_raw_json,
        csv_output_path=csv_output_path,
        queue_id=queue_id,
        min_game_duration=min_game_duration,
        min_remake_duration=min_remake_duration,
        require_valid_position=require_valid_position,
    )

    print(f"Seeded {len(puuids)} players.")
    print(f"Collected {len(match_ids)} unique matches.")
    if store_raw_json:
        print(f"Saved {saved_json_count} new match JSON files to {output_dir}.")
    if csv_output_path:
        print(
            f"Appended {saved_csv_row_count} participant rows "
            f"from {saved_csv_match_count} new matches to {csv_output_path}."
        )
