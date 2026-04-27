import pandas as pd
from pathlib import Path

df = pd.read_csv("data/raw/compositions_s16.csv")
df = df.dropna(subset=["champion_name", "position"])
df["position"] = df["position"].astype(str).str.upper()

counts = df.groupby(["position", "champion_name"]).size().reset_index(name="games")
total_games = df.groupby("champion_name").size().reset_index(name="total_games")
counts = counts.merge(total_games, on="champion_name", how="left")
counts["role_share"] = counts["games"] / counts["total_games"]

print("Gragas:")
print(counts[counts["champion_name"] == "Gragas"])

print("\nYasuo:")
print(counts[counts["champion_name"] == "Yasuo"])
