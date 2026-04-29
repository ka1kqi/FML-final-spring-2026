"""
End-to-end training pipeline for the draft recommender.

Usage:
    python -m src.models.train_draft_models

Steps:
    1. Load compositions_s16.csv
    2. Train Performance Embeddings (Custom Matrix Factorization)
    3. Compute champion average comp scores
    4. Build draft training features (simulating draft order)
    5. Train HistGradientBoostingRegressor
    6. Evaluate and print metrics
    7. Save everything to data/processed/draft_models/
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.models.train_embeddings import train_champion2vec, most_similar
from src.features.synergy_features import compute_champion_scores
from src.models.draft_classifier import (
    build_training_data, train_draft_model, evaluate_model, save_draft_model,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "draft_models"


def main():
    print("=" * 60)
    print("  Draft Recommender Training Pipeline")
    print("=" * 60)

    # --- Step 1: Load data ---
    print("\n[1/6] Loading composition data...")
    comp_file = PROJECT_ROOT / "data/raw/compositions_s16.csv"
    if not comp_file.exists():
        comp_file = PROJECT_ROOT / "src/data/compositions.csv"
        print(f"  Warning: using fallback {comp_file.name}")
    comp_df = pd.read_csv(comp_file)
    print(f"  Loaded {len(comp_df)} rows, {comp_df['match_id'].nunique()} matches")
    
    # --- Step 1.5: Temporal Split (Phase 1) ---
    print("\n[1.5/6] Splitting data chronologically by patch (Train: 16.1-16.6, Test: 16.7+)...")
    comp_df['patch_minor'] = comp_df['patch'].apply(lambda x: int(str(x).split('.')[1]))
    
    train_df = comp_df[comp_df['patch_minor'] <= 6].copy()
    test_df = comp_df[comp_df['patch_minor'] >= 7].copy()
    
    train_matches = train_df['match_id'].nunique()
    test_matches = test_df['match_id'].nunique()
    print(f"  Train matches: {train_matches} | Test matches: {test_matches}")

    # --- Step 2: Train Champion2Vec from scratch ---
    print("\n[2/6] Training Champion2Vec embeddings from scratch on training data...")
    embed_dict, vocab = train_champion2vec(train_df, embed_dim=64)
    print(f"  Trained embeddings for {len(embed_dict)} champions")

    # Embedding quality check
    print("\n  Embedding quality check:")
    for champ in ["Yasuo", "Jinx", "Thresh", "LeeSin"]:
        if champ in embed_dict:
            sims = most_similar(champ, embed_dict, top_k=3)
            sim_str = ", ".join(f"{n} ({s:.3f})" for n, s in sims)
            print(f"    {champ}: {sim_str}")

    # --- Step 3: Compute champion scores ---
    print("\n[3/6] Computing champion average scores on training data...")
    champ_scores = compute_champion_scores(train_df)
    top_scores = sorted(champ_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"  Top 5 avg scores: {[(n, f'{r:.1f}') for n, r in top_scores]}")

    # --- Step 4: Build training data ---
    print("\n[4/6] Building draft training data (simulating draft order)...")
    X_train, y_train = build_training_data(train_df, embed_dict, champ_scores)
    print(f"  Train Samples: {len(y_train)} | Features: {X_train.shape[1]}")
    print(f"  Target mean score in training data: {y_train.mean():.1f}")

    # --- Step 4.5: Build testing data ---
    print("\n[4.5/6] Building draft testing data...")
    X_test, y_test = build_training_data(test_df, embed_dict, champ_scores)
    print(f"  Test Samples: {len(y_test)} | Features: {X_test.shape[1]}")

    # --- Step 5: Train HistGradientBoostingRegressor ---
    print("\n[5/6] Training HistGradientBoostingRegressor draft classifier...")
    lgb_model = train_draft_model(X_train, y_train, X_val=X_test, y_val=y_test)

    # --- Step 6: Evaluate ---
    print("\n[6/6] Evaluating...")
    metrics = evaluate_model(lgb_model, X_test, y_test)
    print(f"  RMSE:      {metrics['rmse']:.4f}")
    print(f"  MAE:       {metrics['mae']:.4f}")
    print(f"  R-squared: {metrics['r2']:.4f}")

    # --- Save everything ---
    print(f"\nSaving to {OUTPUT_DIR}/...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save LightGBM model
    save_draft_model(lgb_model, OUTPUT_DIR / "draft_model.joblib")

    # Save embeddings
    embed_weights = np.array([embed_dict[c] for c in vocab])
    np.savez(
        OUTPUT_DIR / "champion2vec.npz",
        weights=embed_weights,
        vocab=np.array(vocab),
    )

    # Save champion scores
    with open(OUTPUT_DIR / "champ_scores.json", "w") as f:
        json.dump(champ_scores, f, indent=2)

    print(f"  Saved: draft_model.joblib, champion2vec.npz, champ_scores.json")
    print("\nDone!")


if __name__ == "__main__":
    main()
