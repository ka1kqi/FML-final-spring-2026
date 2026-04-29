"""
Parses raw match JSON into a flat DataFrame (one row per match).
Extracts match_id, team_id, champions, positions, win, bans,
gold, damage, vision score, items per participant.
"""

import glob
import json
import os

import pandas as pd


def parse_match(match_json: dict) -> dict:
    """Extract a flat dict of features from a single match-v5 DTO."""
    info = match_json.get("info", {})
    metadata = match_json.get("metadata", {})

    row = {
        "match_id": metadata.get("matchId"),
        "game_version": info.get("gameVersion"),
        "game_creation": info.get("gameCreation"),
        "game_duration": info.get("gameDuration"),
        "queue_id": info.get("queueId"),
    }

    # Team-level: bans and win
    for team in info.get("teams", []):
        tid = team["teamId"]  # 100 or 200
        prefix = "blue" if tid == 100 else "red"
        row[f"{prefix}_win"] = team["win"]
        for i, ban in enumerate(team.get("bans", [])):
            row[f"{prefix}_ban{i+1}"] = ban.get("championId")

    # Participant-level
    for p in info.get("participants", []):
        prefix = f"p{p['participantId']}"
        row[f"{prefix}_champion"] = p.get("championId")
        row[f"{prefix}_position"] = p.get("teamPosition")
        row[f"{prefix}_team"] = p.get("teamId")
        row[f"{prefix}_win"] = p.get("win")
        row[f"{prefix}_kills"] = p.get("kills")
        row[f"{prefix}_deaths"] = p.get("deaths")
        row[f"{prefix}_assists"] = p.get("assists")
        row[f"{prefix}_gold"] = p.get("goldEarned")
        row[f"{prefix}_damage"] = p.get("totalDamageDealtToChampions")
        row[f"{prefix}_vision"] = p.get("visionScore")
        row[f"{prefix}_cs"] = p.get("totalMinionsKilled")
        for i in range(7):
            row[f"{prefix}_item{i}"] = p.get(f"item{i}")

    return row


def parse_pro_match(game_df: pd.DataFrame) -> dict:
    """
    Convert a group of Oracle's Elixir rows (one gameid) into the same
    flat dict format as parse_match.
    """
    # Use team summary rows (participantid 100/200) for team-level info
    teams = game_df[game_df["participantid"].isin([100, 200])]
    players = game_df[~game_df["participantid"].isin([100, 200])]

    first = game_df.iloc[0]
    row = {
        "match_id": first["gameid"],
        "game_version": str(first["patch"]),
        "game_creation": first["date"],
        "game_duration": first["gamelength"],
        "queue_id": 420,  # treat as ranked equivalent
    }

    # Team-level bans and win
    for _, team_row in teams.iterrows():
        prefix = "blue" if team_row["side"] == "Blue" else "red"
        row[f"{prefix}_win"] = int(team_row["result"])
        for i in range(1, 6):
            row[f"{prefix}_ban{i}"] = team_row.get(f"ban{i}")

    # Participant-level
    for i, (_, p) in enumerate(players.iterrows(), start=1):
        prefix = f"p{i}"
        team_id = 100 if p["side"] == "Blue" else 200
        row[f"{prefix}_champion"] = p.get("champion")
        row[f"{prefix}_position"] = p.get("position")
        row[f"{prefix}_team"] = team_id
        row[f"{prefix}_win"] = int(p.get("result", 0))
        row[f"{prefix}_kills"] = p.get("kills")
        row[f"{prefix}_deaths"] = p.get("deaths")
        row[f"{prefix}_assists"] = p.get("assists")
        row[f"{prefix}_gold"] = p.get("totalgold")
        row[f"{prefix}_damage"] = p.get("damagetochampions")
        row[f"{prefix}_vision"] = p.get("visionscore")
        row[f"{prefix}_cs"] = p.get("minionkills")
        # no item columns in OE data, fill with None
        for j in range(7):
            row[f"{prefix}_item{j}"] = None

    return row


def _get_champion_name_to_id() -> dict[str, int]:
    """Fetch champion name -> integer ID mapping from Riot Data Dragon."""
    import requests
    resp = requests.get(
        "https://ddragon.leagueoflegends.com/cdn/14.8.1/data/en_US/champion.json"
    )
    data = resp.json()
    return {v["name"]: int(v["key"]) for v in data["data"].values()}


def parse_pro_csv(csv_path: str) -> pd.DataFrame:
    """Load Oracle's Elixir CSV and return one row per match."""
    name_to_id = _get_champion_name_to_id()
    df = pd.read_csv(csv_path)

    # Normalize champion name to ID
    df["champion"] = df["champion"].map(name_to_id)
    for i in range(1, 6):
        df[f"ban{i}"] = df[f"ban{i}"].map(name_to_id)

    rows = []
    for gameid, group in df.groupby("gameid"):
        try:
            rows.append(parse_pro_match(group))
        except Exception as e:
            print(f"Skipping {gameid}: {e}")
    return pd.DataFrame(rows)


def parse_all(raw_dir: str = "data/raw") -> pd.DataFrame:
    """Load all raw JSON files from raw_dir and return a combined DataFrame."""
    rows = []
    for fp in glob.glob(os.path.join(raw_dir, "*.json")):
        with open(fp) as f:
            try:
                match_json = json.load(f)
                rows.append(parse_match(match_json))
            except Exception as e:
                print(f"Skipping {fp}: {e}")
    return pd.DataFrame(rows)


def save_processed(df: pd.DataFrame, output_path: str = "data/processed/matches.csv") -> None:
    """Write the processed DataFrame to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)