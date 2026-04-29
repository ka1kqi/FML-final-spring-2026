import pandas as pd
import numpy as np
from pathlib import Path

def preprocess_matches():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    raw_file = PROJECT_ROOT / "data/raw/matches.csv"
    out_file = PROJECT_ROOT / "data/raw/compositions_s16.csv"
    
    print(f"Loading {raw_file}...")
    df = pd.read_csv(raw_file)
    
    # Filter for season 16
    print("Filtering for Season 16 patches...")
    df['patch'] = df['gameVersion'].str.extract(r'^(\d+\.\d+)')
    s16_df = df[df['patch'].str.startswith('16.')].copy()
    print(f"Found {len(s16_df)} player records from Season 16.")
    
    # Group by match and team
    print("Extracting individual stats...")
    
    # Calculate per-minute and ratio stats for each individual player
    s16_df['gpm'] = s16_df['goldEarned'] / (s16_df['gameDuration'] / 60.0)
    s16_df['dpm'] = s16_df['totalDamageDealtToChampions'] / (s16_df['gameDuration'] / 60.0)
    s16_df['kda'] = (s16_df['kills'] + s16_df['assists']) / s16_df['deaths'].clip(lower=1)
    
    # Z-score normalize the metrics within roles (teamPosition)
    for col in ['gpm', 'dpm', 'kda', 'team_dragon_kills', 'team_tower_kills']:
        s16_df[f'{col}_z'] = s16_df.groupby('teamPosition')[col].transform(lambda x: (x - x.mean()) / x.std())
        
    # Composite score
    # Win brings a flat +3 or 0 base
    s16_df['raw_champ_score'] = (
        s16_df['win'].astype(int) * 3.0 +
        s16_df['kda_z'] * 1.5 +
        s16_df['gpm_z'] * 0.6 +
        s16_df['dpm_z'] * 0.6 +
        s16_df['team_dragon_kills_z'] * 0.3 +
        s16_df['team_tower_kills_z'] * 0.2
    )
    
    # Standardize the raw comp score strictly WITHIN the role so every role averages 50
    s16_df['champ_score_z'] = s16_df.groupby('teamPosition')['raw_champ_score'].transform(lambda x: (x - x.mean()) / x.std())
    
    # Map to 0-100 scale: 50 is average, each std dev is 10 points
    s16_df['champ_score'] = 50.0 + (s16_df['champ_score_z'] * 10.0)
    s16_df['champ_score'] = s16_df['champ_score'].fillna(50.0).clip(lower=0.0, upper=100.0).round(2)
    
    # Now build the compositions format
    print("Formatting compositions data...")
    comp_data = s16_df[['matchId', 'championId', 'championName', 'teamId', 'teamPosition', 'win', 'champ_score', 'patch']].copy()
    comp_data = comp_data.rename(columns={
        'matchId': 'match_id',
        'championId': 'champion_id',
        'championName': 'champion_name',
        'teamId': 'team_id',
        'teamPosition': 'position'
    })
    
    print(f"Saving to {out_file}...")
    comp_data.to_csv(out_file, index=False)
    print("Done!")

if __name__ == "__main__":
    preprocess_matches()
