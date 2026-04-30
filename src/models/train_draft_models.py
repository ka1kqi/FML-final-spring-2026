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

from src.models.train_embeddings import (
    train_champion2vec, most_similar, top_synergies, top_counters,
    compute_primary_roles, train_champion_role_2vec,
)
from src.features.synergy_features import compute_champion_scores
from src.models.draft_classifier import (
    build_training_data, train_draft_model, evaluate_model, save_draft_model,
)
from src.models.match_classifier import (
    build_match_training_data, train_match_classifier, evaluate_match_model,
    save_match_model,
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
    
    # --- Step 1.5: Random match-level 80/20 split ---
    # We're modeling the *current meta* (composition strength given today's
    # game state), not forecasting future patches. Random split is the right
    # eval frame: train and val both reflect the same population of metas.
    print("\n[1.5/6] Random 80/20 split by match_id (current-meta evaluation)...")
    rng = np.random.default_rng(42)
    all_match_ids = comp_df["match_id"].unique()
    rng.shuffle(all_match_ids)
    n_train = int(0.8 * len(all_match_ids))
    train_ids = set(all_match_ids[:n_train])
    val_ids = set(all_match_ids[n_train:])

    train_df = comp_df[comp_df["match_id"].isin(train_ids)].copy()
    test_df = comp_df[comp_df["match_id"].isin(val_ids)].copy()
    print(f"  Train matches: {train_df['match_id'].nunique()} | Val matches: {test_df['match_id'].nunique()}")

    # --- Step 2: Train Champion2Vec from scratch ---
    print("\n[2/6] Training Champion2Vec embeddings (bias terms + L2-normalized blocks)...")
    embed_dict, vocab, biases = train_champion2vec(train_df, embed_dim=64)
    print(f"  Trained embeddings for {len(embed_dict)} champions")
    print(f"  Synergy mu = {biases['mu_syn']:+.3f}  | Matchup mu = {biases['mu_match']:+.3f}")
    print(f"  |b_syn_u| range: [{biases['b_syn_u'].min():+.3f}, {biases['b_syn_u'].max():+.3f}]")

    # Embedding quality check
    print("\n  Archetype neighbors (cosine over full embedding):")
    for champ in ["Yasuo", "Jinx", "Thresh", "LeeSin"]:
        if champ in embed_dict:
            sims = most_similar(champ, embed_dict, top_k=3)
            sim_str = ", ".join(f"{n} ({s:.3f})" for n, s in sims)
            print(f"    {champ}: {sim_str}")

    primary_role = compute_primary_roles(train_df)

    print("\n  Top synergy allies (U_syn[c] . V_syn[ally], same-role filtered):")
    for champ in ["Yasuo", "Jinx", "Thresh", "LeeSin"]:
        if champ in embed_dict:
            syns = top_synergies(champ, embed_dict, top_k=3, primary_role=primary_role)
            syn_str = ", ".join(f"{n} ({s:+.2f})" for n, s in syns)
            print(f"    {champ} ({primary_role.get(champ,'?')}): {syn_str}")

    print("\n  Top counters (U_match[c] . V_match[enemy]):")
    for champ in ["Yasuo", "Jinx", "Thresh", "LeeSin"]:
        if champ in embed_dict:
            ctrs = top_counters(champ, embed_dict, top_k=3)
            ctr_str = ", ".join(f"{n} ({s:+.2f})" for n, s in ctrs)
            print(f"    {champ}: {ctr_str}")

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

    # --- Step 5: Train HistGradientBoostingClassifier ---
    print("\n[5/6] Training HistGradientBoostingClassifier on win/loss...")
    print(f"  Train base rate (P(win)): {y_train.mean():.4f}")
    lgb_model = train_draft_model(X_train, y_train, X_val=X_test, y_val=y_test)

    # --- Step 6: Evaluate ---
    print("\n[6/6] Evaluating...")
    metrics = evaluate_model(lgb_model, X_test, y_test)
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  AUC:       {metrics['auc']:.4f}")
    print(f"  Log loss:  {metrics['log_loss']:.4f}  (base = {-np.log(0.5):.4f})")
    print(f"  Brier:     {metrics['brier']:.4f}  (base = 0.25)")
    print(f"  Test base rate: {metrics['base_rate']:.4f}")

    # --- Step 7: Per-match aggregate classifier ---
    print("\n[7] Building per-match training/test data (1 row per match)...")
    Xm_train, ym_train = build_match_training_data(train_df, embed_dict, champ_scores)
    Xm_test, ym_test = build_match_training_data(test_df, embed_dict, champ_scores)
    print(f"  Match-level: train={len(ym_train)} | test={len(ym_test)} | features={Xm_train.shape[1]}")

    print(f"  Train base rate P(blue_win): {ym_train.mean():.4f}")
    match_model = train_match_classifier(Xm_train, ym_train, X_val=Xm_test, y_val=ym_test)

    print("\n[7.1] Evaluating calibrated per-match classifier on validation...")
    m_metrics = evaluate_match_model(match_model, Xm_test, ym_test)
    print(f"  Accuracy:  {m_metrics['accuracy']:.4f}")
    print(f"  AUC:       {m_metrics['auc']:.4f}")
    print(f"  Log loss:  {m_metrics['log_loss']:.4f}  (base = {-np.log(0.5):.4f})")
    print(f"  Brier:     {m_metrics['brier']:.4f}  (base = 0.25)")
    print(f"  Test base rate: {m_metrics['base_rate']:.4f}")

    # --- Save everything ---
    print(f"\nSaving to {OUTPUT_DIR}/...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_match_model(match_model, OUTPUT_DIR / "match_model.joblib")

    # Save LightGBM model
    save_draft_model(lgb_model, OUTPUT_DIR / "draft_model.joblib")

    # Save embeddings + bias terms (each 16-d block of `weights` is unit-normalized)
    embed_weights = np.array([embed_dict[c] for c in vocab])
    np.savez(
        OUTPUT_DIR / "champion2vec.npz",
        weights=embed_weights,
        vocab=np.array(vocab),
        b_syn_u=biases["b_syn_u"],
        b_syn_v=biases["b_syn_v"],
        mu_syn=np.float32(biases["mu_syn"]),
        b_match_u=biases["b_match_u"],
        b_match_v=biases["b_match_v"],
        mu_match=np.float32(biases["mu_match"]),
    )

    # --- Step 8: Role-aware embedding (champion, role) units ---
    # Used by /api/role_analysis to answer queries like
    # "Akali in MID vs Ahri in MID" — each role gets its own profile.
    print("\n[8] Training role-aware (champion, role) embedding...")
    role_embed, role_vocab, role_biases = train_champion_role_2vec(train_df)
    role_weights = np.array([role_embed[k] for k in role_vocab])
    np.savez(
        OUTPUT_DIR / "champion_role_2vec.npz",
        weights=role_weights,
        vocab=np.array(role_vocab),
        b_syn_u=role_biases["b_syn_u"],
        b_syn_v=role_biases["b_syn_v"],
        mu_syn=np.float32(role_biases["mu_syn"]),
        b_match_u=role_biases["b_match_u"],
        b_match_v=role_biases["b_match_v"],
        mu_match=np.float32(role_biases["mu_match"]),
    )
    print(f"  Saved: champion_role_2vec.npz ({len(role_vocab)} units)")

    # Save champion scores
    with open(OUTPUT_DIR / "champ_scores.json", "w") as f:
        json.dump(champ_scores, f, indent=2)

    print(f"  Saved: draft_model.joblib, champion2vec.npz, champ_scores.json")
    print("\nDone!")


if __name__ == "__main__":
    main()
