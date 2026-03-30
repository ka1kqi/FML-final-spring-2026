"""
Ablation study: train and evaluate models on progressively richer feature sets.
Feature sets:
  1. Draft only (champion IDs)
  2. Draft + in-game stats (gold, damage, vision)
  3. Draft + stats + items
"""

import pandas as pd


def run_ablation(data_path: str = "data/processed/matches.csv") -> pd.DataFrame:
    """
    For each feature set and each model type, train, evaluate, and
    return a DataFrame of results (feature_set, model, accuracy, f1, auc).
    """
    raise NotImplementedError


def plot_ablation(results: pd.DataFrame, output_path: str = "notebooks/ablation.png") -> None:
    """Generate a grouped bar chart comparing models across feature sets."""
    raise NotImplementedError
