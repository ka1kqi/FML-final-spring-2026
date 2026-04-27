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
    print("Aggregating team stats...")
    team_stats = s16_df.groupby(['matchId', 'teamId', 'win']).agg({
        'kills': 'sum',
        'deaths': 'sum',
        'assists': 'sum',
        'goldEarned': 'sum',
        'totalDamageDealtToChampions': 'sum',
        'gameDuration': 'first',
        'team_dragon_kills': 'first',
        'team_baron_kills': 'first',
        'team_tower_kills': 'first'
    }).reset_index()
    
    # Calculate per-minute and ratio stats
    team_stats['gpm'] = team_stats['goldEarned'] / (team_stats['gameDuration'] / 60.0)
    team_stats['dpm'] = team_stats['totalDamageDealtToChampions'] / (team_stats['gameDuration'] / 60.0)
    team_stats['kda'] = (team_stats['kills'] + team_stats['assists']) / team_stats['deaths'].clip(lower=1)
    
    # Z-score normalize the metrics across all teams
    for col in ['gpm', 'dpm', 'kda', 'team_dragon_kills', 'team_tower_kills']:
        team_stats[f'{col}_z'] = (team_stats[col] - team_stats[col].mean()) / team_stats[col].std()
        
    # Composite score
    # Win brings a flat +5 or 0 base
    team_stats['raw_comp_score'] = (
        team_stats['win'].astype(int) * 5.0 +
        team_stats['kda_z'] * 1.5 +
        team_stats['gpm_z'] * 1.0 +
        team_stats['dpm_z'] * 1.0 +
        team_stats['team_dragon_kills_z'] * 0.5 +
        team_stats['team_tower_kills_z'] * 0.5
    )
    
    # Standardize the raw comp score so mean=0, std=1
    score_mean = team_stats['raw_comp_score'].mean()
    score_std = team_stats['raw_comp_score'].std()
    team_stats['comp_score_z'] = (team_stats['raw_comp_score'] - score_mean) / score_std
    
    # Map to 0-100 scale: 50 is average, each std dev is 10 points
    team_stats['comp_score'] = 50.0 + (team_stats['comp_score_z'] * 10.0)
    team_stats['comp_score'] = team_stats['comp_score'].clip(lower=0.0, upper=100.0).round(2)
    
    # Now merge back with the champion data to create the compositions format
    print("Formatting compositions data...")
    comp_data = s16_df[['matchId', 'championId', 'championName', 'teamId', 'teamPosition', 'win']].copy()
    comp_data = comp_data.rename(columns={
        'matchId': 'match_id',
        'championId': 'champion_id',
        'championName': 'champion_name',
        'teamId': 'team_id',
        'teamPosition': 'position'
    })
    
    # Join the comp_score
    scores_dict = team_stats.set_index(['matchId', 'teamId'])['comp_score'].to_dict()
    comp_data['comp_score'] = comp_data.set_index(['match_id', 'team_id']).index.map(scores_dict)
    
    print(f"Saving to {out_file}...")
    comp_data.to_csv(out_file, index=False)
    print("Done!")

if __name__ == "__main__":
    preprocess_matches()
