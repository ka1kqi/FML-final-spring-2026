"""
Parses raw Riot match JSON into a compact, player-level DataFrame.

Output shape: one row per participant (10 rows per standard match).
Core columns include match_id, player_id (puuid/summonerId), champion,
team, role/position, result, KDA, gold, damage, vision, and lane stats.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from src.preprocessing.filters import apply_all_filters


OUTPUT_COLUMNS = [
    "match_id",
    "game_creation",
    "game_start_timestamp",
    "game_end_timestamp",
    "game_duration",
    "game_version",
    "platform_id",
    "queue_id",
    "team_id",
    "participant_id",
    "player_id",
    "champion_id",
    "champion_name",
    "position",
    "win",
    "kills",
    "deaths",
    "assists",
    "kda",
    "kill_participation",
    "gold_earned",
    "gold_spent",
    "gold_per_minute",
    "total_damage_to_champions",
    "physical_damage_to_champions",
    "magic_damage_to_champions",
    "true_damage_to_champions",
    "damage_per_minute",
    "total_damage_taken",
    "damage_self_mitigated",
    "total_heal",
    "total_heals_on_teammates",
    "vision_score",
    "vision_score_per_minute",
    "wards_placed",
    "wards_killed",
    "vision_wards_bought",
    "total_minions_killed",
    "neutral_minions_killed",
    "total_cs",
    "cs_per_minute",
    "champ_level",
    "time_ccing_others",
    "total_time_spent_dead",
    "damage_to_objectives",
    "damage_to_turrets",
    "turret_kills",
    "turret_takedowns",
    "first_blood_kill",
    "first_blood_assist",
    "double_kills",
    "triple_kills",
    "quadra_kills",
    "penta_kills",
    "game_ended_in_surrender",
    "game_ended_in_early_surrender",
    "team_early_surrendered",
]


FILTER_COLUMNS = ["individual_position", "lane", "role"]
LEGACY_OUTPUT_COLUMNS = [column for column in OUTPUT_COLUMNS if column != "position"]


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float(numerator)
    return float(numerator) / float(denominator)


def _participant_position(participant: dict) -> str:
    position = participant.get("teamPosition") or participant.get("individualPosition")
    if position:
        return str(position)
    return "UNKNOWN"


def _parse_participant(match_info: dict, participant: dict) -> dict:
    kills = int(participant.get("kills", 0))
    deaths = int(participant.get("deaths", 0))
    assists = int(participant.get("assists", 0))

    challenges = participant.get("challenges", {}) or {}
    kda = challenges.get("kda")
    if kda is None:
        kda = _safe_div(kills + assists, max(1, deaths))

    game_duration = int(match_info.get("gameDuration", 0))
    game_minutes = game_duration / 60.0 if game_duration else 0.0

    total_minions_killed = int(participant.get("totalMinionsKilled", 0))
    neutral_minions_killed = int(participant.get("neutralMinionsKilled", 0))
    total_cs = total_minions_killed + neutral_minions_killed

    puuid = participant.get("puuid")
    summoner_id = participant.get("summonerId")
    player_id = puuid or summoner_id

    return {
        "match_id": match_info.get("matchId"),
        "game_creation": match_info.get("gameCreation"),
        "game_start_timestamp": match_info.get("gameStartTimestamp"),
        "game_end_timestamp": match_info.get("gameEndTimestamp"),
        "game_duration": game_duration,
        "game_version": match_info.get("gameVersion"),
        "platform_id": match_info.get("platformId"),
        "queue_id": match_info.get("queueId"),
        "team_id": participant.get("teamId"),
        "participant_id": participant.get("participantId"),
        "player_id": player_id,
        "champion_id": participant.get("championId"),
        "champion_name": participant.get("championName"),
        "position": _participant_position(participant),
        "individual_position": participant.get("individualPosition"),
        "lane": participant.get("lane"),
        "role": participant.get("role"),
        "win": bool(participant.get("win", False)),
        "kills": kills,
        "deaths": deaths,
        "assists": assists,
        "kda": float(kda),
        "kill_participation": float(challenges.get("killParticipation", 0.0)),
        "gold_earned": int(participant.get("goldEarned", 0)),
        "gold_spent": int(participant.get("goldSpent", 0)),
        "gold_per_minute": float(challenges.get("goldPerMinute", 0.0)),
        "total_damage_to_champions": int(participant.get("totalDamageDealtToChampions", 0)),
        "physical_damage_to_champions": int(participant.get("physicalDamageDealtToChampions", 0)),
        "magic_damage_to_champions": int(participant.get("magicDamageDealtToChampions", 0)),
        "true_damage_to_champions": int(participant.get("trueDamageDealtToChampions", 0)),
        "damage_per_minute": float(challenges.get("damagePerMinute", 0.0)),
        "total_damage_taken": int(participant.get("totalDamageTaken", 0)),
        "damage_self_mitigated": int(participant.get("damageSelfMitigated", 0)),
        "total_heal": int(participant.get("totalHeal", 0)),
        "total_heals_on_teammates": int(participant.get("totalHealsOnTeammates", 0)),
        "vision_score": int(participant.get("visionScore", 0)),
        "vision_score_per_minute": float(challenges.get("visionScorePerMinute", 0.0)),
        "wards_placed": int(participant.get("wardsPlaced", 0)),
        "wards_killed": int(participant.get("wardsKilled", 0)),
        "vision_wards_bought": int(participant.get("visionWardsBoughtInGame", 0)),
        "total_minions_killed": total_minions_killed,
        "neutral_minions_killed": neutral_minions_killed,
        "total_cs": total_cs,
        "cs_per_minute": _safe_div(total_cs, game_minutes) if game_minutes else 0.0,
        "champ_level": int(participant.get("champLevel", 0)),
        "time_ccing_others": int(participant.get("timeCCingOthers", 0)),
        "total_time_spent_dead": int(participant.get("totalTimeSpentDead", 0)),
        "damage_to_objectives": int(participant.get("damageDealtToObjectives", 0)),
        "damage_to_turrets": int(participant.get("damageDealtToTurrets", 0)),
        "turret_kills": int(participant.get("turretKills", 0)),
        "turret_takedowns": int(participant.get("turretTakedowns", 0)),
        "first_blood_kill": bool(participant.get("firstBloodKill", False)),
        "first_blood_assist": bool(participant.get("firstBloodAssist", False)),
        "double_kills": int(participant.get("doubleKills", 0)),
        "triple_kills": int(participant.get("tripleKills", 0)),
        "quadra_kills": int(participant.get("quadraKills", 0)),
        "penta_kills": int(participant.get("pentaKills", 0)),
        "game_ended_in_surrender": bool(participant.get("gameEndedInSurrender", False)),
        "game_ended_in_early_surrender": bool(participant.get("gameEndedInEarlySurrender", False)),
        "team_early_surrendered": bool(participant.get("teamEarlySurrendered", False)),
    }


def parse_match(match_json: dict) -> list[dict]:
    """Extract participant-level rows from a single match-v5 DTO."""
    metadata = match_json.get("metadata", {}) or {}
    info = match_json.get("info", {}) or {}

    participants = info.get("participants", []) or []

    match_info = {
        "matchId": metadata.get("matchId"),
        "gameCreation": info.get("gameCreation"),
        "gameStartTimestamp": info.get("gameStartTimestamp"),
        "gameEndTimestamp": info.get("gameEndTimestamp"),
        "gameDuration": info.get("gameDuration"),
        "gameVersion": info.get("gameVersion"),
        "platformId": info.get("platformId"),
        "queueId": info.get("queueId"),
    }

    return [_parse_participant(match_info, participant) for participant in participants]


def parse_all(raw_dir: str = "data/raw") -> pd.DataFrame:
    """Load all raw JSON files from raw_dir and return a participant-level DataFrame."""
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        return pd.DataFrame(columns=OUTPUT_COLUMNS + FILTER_COLUMNS)

    rows: list[dict] = []
    for json_path in sorted(raw_path.glob("*.json")):
        try:
            with open(json_path, "r", encoding="utf-8") as infile:
                match_json = json.load(infile)
        except (OSError, json.JSONDecodeError):
            continue

        rows.extend(parse_match(match_json))

    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS + FILTER_COLUMNS)

    df = pd.DataFrame(rows)
    for column in OUTPUT_COLUMNS + FILTER_COLUMNS:
        if column not in df:
            df[column] = pd.NA

    return df[OUTPUT_COLUMNS + FILTER_COLUMNS]


def load_processed_csv(csv_path: str) -> pd.DataFrame:
    """Load an existing processed CSV and normalize it to the current schema."""
    path = Path(csv_path)
    if not path.exists():
        return pd.DataFrame(columns=OUTPUT_COLUMNS + FILTER_COLUMNS)

    rows: list[dict] = []
    with open(path, "r", encoding="utf-8", newline="") as infile:
        reader = csv.reader(infile)
        header = next(reader, [])
        if not header:
            return pd.DataFrame(columns=OUTPUT_COLUMNS + FILTER_COLUMNS)

        for values in reader:
            if not values:
                continue

            row: dict = {}
            if header == OUTPUT_COLUMNS:
                if len(values) >= len(OUTPUT_COLUMNS):
                    row.update(dict(zip(OUTPUT_COLUMNS, values[: len(OUTPUT_COLUMNS)])))
                elif len(values) == len(LEGACY_OUTPUT_COLUMNS):
                    row.update(dict(zip(LEGACY_OUTPUT_COLUMNS, values)))
                else:
                    continue
            elif header == LEGACY_OUTPUT_COLUMNS:
                if len(values) == len(LEGACY_OUTPUT_COLUMNS):
                    row.update(dict(zip(LEGACY_OUTPUT_COLUMNS, values)))
                elif len(values) >= len(OUTPUT_COLUMNS):
                    row.update(dict(zip(OUTPUT_COLUMNS, values[: len(OUTPUT_COLUMNS)])))
                else:
                    continue
            else:
                mapped = dict(zip(header, values[: len(header)]))
                for column in OUTPUT_COLUMNS + FILTER_COLUMNS:
                    if column in mapped:
                        row[column] = mapped[column]

            if "position" not in row or not row["position"]:
                row["position"] = "UNKNOWN"

            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS + FILTER_COLUMNS)

    df = pd.DataFrame(rows)
    for column in OUTPUT_COLUMNS + FILTER_COLUMNS:
        if column not in df:
            df[column] = pd.NA

    return df[OUTPUT_COLUMNS + FILTER_COLUMNS]


def clean_dataframe(
    df: pd.DataFrame,
    queue_id: int = 420,
    min_game_duration: int = 900,
    min_remake_duration: int = 180,
    require_valid_position: bool = True,
    dedupe: bool = True,
) -> pd.DataFrame:
    """Apply filters and optional dedupe for participant-level rows."""
    cleaned = apply_all_filters(
        df,
        queue_id=queue_id,
        min_game_duration=min_game_duration,
        min_remake_duration=min_remake_duration,
        require_valid_position=require_valid_position,
    )

    if dedupe:
        dedupe_cols = [column for column in ["match_id", "participant_id"] if column in cleaned.columns]
        if dedupe_cols:
            cleaned = cleaned.drop_duplicates(subset=dedupe_cols, keep="first")

    return cleaned.reset_index(drop=True)


def save_processed(df: pd.DataFrame, output_path: str = "data/processed/compositions_stats.csv") -> None:
    """Write the processed DataFrame to CSV."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_columns = [column for column in OUTPUT_COLUMNS if column in df.columns]
    df[write_columns].to_csv(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse raw Riot JSON into compact player-level CSV")
    parser.add_argument("--input-csv", default=None, help="Optional existing CSV to clean/migrate")
    parser.add_argument("--raw-dir", default="data/raw", help="Directory containing raw match JSON files")
    parser.add_argument(
        "--output",
        default="data/processed/compositions_stats.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--no-filters",
        action="store_true",
        help="Skip ranked/position/duration filtering",
    )
    parser.add_argument("--queue-id", type=int, default=420, help="Queue ID to keep (default: 420)")
    parser.add_argument("--min-game-duration", type=int, default=900, help="Minimum game duration in seconds")
    parser.add_argument("--min-remake-duration", type=int, default=180, help="Minimum remake duration in seconds")
    parser.add_argument(
        "--no-position-filter",
        action="store_true",
        help="Do not drop rows with invalid or missing position",
    )
    parser.add_argument("--no-dedupe", action="store_true", help="Do not deduplicate match_id+participant_id")
    args = parser.parse_args()

    if args.input_csv:
        df = load_processed_csv(args.input_csv)
    else:
        df = parse_all(raw_dir=args.raw_dir)

    if not args.no_filters:
        df = clean_dataframe(
            df,
            queue_id=args.queue_id,
            min_game_duration=args.min_game_duration,
            min_remake_duration=args.min_remake_duration,
            require_valid_position=not args.no_position_filter,
            dedupe=not args.no_dedupe,
        )
    elif not args.no_dedupe:
        dedupe_cols = [column for column in ["match_id", "participant_id"] if column in df.columns]
        if dedupe_cols:
            df = df.drop_duplicates(subset=dedupe_cols, keep="first").reset_index(drop=True)

    save_processed(df, output_path=args.output)
    print(f"Parsed {len(df)} participant rows to {args.output}")


if __name__ == "__main__":
    main()
