from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def load_champion_vocab(project_root: Path) -> Tuple[Dict[str, int], Dict[int, str], List[str]]:
    data_file = project_root / "data/raw/compositions_s16.csv"

    df = pd.read_csv(data_file)
    champions = sorted(df["champion_name"].dropna().unique().tolist())
    champ_to_id = {name: idx for idx, name in enumerate(champions)}
    id_to_champ = {idx: name for idx, name in enumerate(champions)}
    return champ_to_id, id_to_champ, champions


def load_role_champion_options(
    project_root: Path,
    min_games_for_role: int = 20,
    min_role_share: float = 0.10,
) -> Dict[str, List[str]]:
    """
    Build role-specific champion options from historical position counts.
    Falls back to all champions for any missing role bucket.
    """
    data_file = project_root / "data/raw/compositions_s16.csv"

    df = pd.read_csv(data_file)
    df = df.dropna(subset=["champion_name", "position"])
    df["position"] = df["position"].astype(str).str.upper()
    df["champion_name"] = df["champion_name"].astype(str)

    all_champions = sorted(df["champion_name"].unique().tolist())
    role_map = {
        "TOP": "Top",
        "JUNGLE": "Jungle",
        "MIDDLE": "Mid",
        "BOTTOM": "ADC",
        "UTILITY": "Support",
    }

    counts = (
        df.groupby(["position", "champion_name"])
        .size()
        .reset_index(name="games")
    )
    total_games = (
        df.groupby("champion_name")
        .size()
        .reset_index(name="total_games")
    )
    counts = counts.merge(total_games, on="champion_name", how="left")
    counts["role_share"] = counts["games"] / counts["total_games"]
    counts = counts[
        (counts["games"] >= min_games_for_role)
        & (counts["role_share"] >= min_role_share)
    ]

    role_options: Dict[str, List[str]] = {}
    for raw_role, app_role in role_map.items():
        champs = sorted(
            counts[counts["position"] == raw_role]["champion_name"].unique().tolist()
        )
        role_options[app_role] = champs if champs else all_champions

    return role_options
