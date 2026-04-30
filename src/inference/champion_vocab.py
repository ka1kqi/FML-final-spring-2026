import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MODELS_SUBDIR = Path("data") / "processed" / "draft_models"


def _champions_from_npz(project_root: Path) -> List[str]:
    """Derive champion list from champion2vec.npz vocab when CSV is missing."""
    npz_path = project_root / _MODELS_SUBDIR / "champion2vec.npz"
    data = np.load(str(npz_path), allow_pickle=True)
    return sorted(data["vocab"].tolist())


def load_champion_vocab(project_root: Path) -> Tuple[Dict[str, int], Dict[int, str], List[str]]:
    data_file = project_root / "data/raw/compositions_s16.csv"

    if data_file.exists():
        df = pd.read_csv(data_file)
        champions = sorted(df["champion_name"].dropna().unique().tolist())
    else:
        logger.warning(
            "%s not found; deriving champion vocab from champion2vec.npz.", data_file.name
        )
        champions = _champions_from_npz(project_root)

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

    if not data_file.exists():
        logger.warning(
            "%s not found; role_options will map all roles to full champion list.", data_file.name
        )
        all_champions = _champions_from_npz(project_root)
        role_map = {
            "TOP": "Top",
            "JUNGLE": "Jungle",
            "MIDDLE": "Mid",
            "BOTTOM": "ADC",
            "UTILITY": "Support",
        }
        return {app_role: all_champions for app_role in role_map.values()}

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
