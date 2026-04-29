"""
prepare_riot_v5_data.py
========================

One-shot conversion script that turns the rich Riot match-v5 dump at
``~/Downloads/data_processed/matches.csv`` into the long-form participant
CSV the existing pipeline expects, with these key upgrades:

* Adds ``patch`` (gameVersion), ``timestamp`` (gameCreation),
  ``bans`` (comma-separated champion names from ban1..5_championId).
* Keeps only the draft-time + label columns. Drops post-game KDA /
  damage / vision / items / 80+ challenges fields **on purpose** to
  avoid leakage temptation downstream. The leakage audit will reflect
  the absence of those columns.

The pipeline's auto-detection (`_detect_columns`) recognises
``matchId / championId / championName / teamId / teamPosition / win /
gameVersion / gameCreation / bans`` directly, so no pipeline code change
is required.

Usage::

    python prepare_riot_v5_data.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SRC = Path.home() / "Downloads" / "data_processed" / "matches.csv"
DST = Path("data/processed/matches.csv")

BAN_COLS = [f"ban{i}_championId" for i in range(1, 6)]
KEEP_COLS = [
    "matchId",
    "championId",
    "championName",
    "teamId",
    "teamPosition",
    "win",
    "gameVersion",
    "gameCreation",
    "bans",
]


def main() -> None:
    if not SRC.is_file():
        raise SystemExit(f"Source CSV not found: {SRC}")
    DST.parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] {SRC}")
    df = pd.read_csv(SRC, low_memory=False)
    print(f"  rows={len(df):,}  matches={df['matchId'].nunique():,}")

    id_to_name = (
        df[["championId", "championName"]]
        .drop_duplicates()
        .set_index("championId")["championName"]
        .to_dict()
    )
    print(f"  champion vocab from data: {len(id_to_name)}")

    # Bans are per-team (each team's 5 bans live on each of its participant
    # rows). Aggregate to a per-match string of up to 10 ban names.
    team_bans = df[["matchId", "teamId"] + BAN_COLS].drop_duplicates(["matchId", "teamId"])
    long_bans = team_bans.melt(
        id_vars=["matchId", "teamId"], value_name="ban_id", var_name="slot"
    )
    long_bans = long_bans.dropna(subset=["ban_id"])
    long_bans["ban_id"] = long_bans["ban_id"].astype(int)
    long_bans = long_bans[long_bans["ban_id"] > 0]
    long_bans["ban_name"] = long_bans["ban_id"].map(id_to_name)
    long_bans = long_bans.dropna(subset=["ban_name"])
    per_match_bans = (
        long_bans.groupby("matchId")["ban_name"]
        .apply(lambda s: ",".join(sorted(set(s))))
        .rename("bans")
        .reset_index()
    )
    df = df.merge(per_match_bans, on="matchId", how="left")
    df["bans"] = df["bans"].fillna("")

    out = df[KEEP_COLS].copy()
    out = out[out["teamPosition"].astype(str).str.strip() != ""].copy()
    out.to_csv(DST, index=False)

    print(f"[write] {DST}")
    print(f"  rows={len(out):,}  matches={out['matchId'].nunique():,}")
    print(f"  patches={out['gameVersion'].nunique()}  champions={out['championName'].nunique()}")
    ts = pd.to_datetime(out["gameCreation"], unit="ms")
    print(f"  time range: {ts.min()} -> {ts.max()}")
    print(f"  bans-per-team non-empty rate: {(out['bans'].str.len() > 0).mean():.2%}")


if __name__ == "__main__":
    main()
