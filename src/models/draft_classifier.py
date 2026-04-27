"""
Draft Win Classifier: predicts P(win | candidate champion, draft context).

Trains a HistGradientBoosting binary classifier on features derived from
champion embeddings and draft state.

Uses sklearn's HistGradientBoostingClassifier (inspired by LightGBM,
but pure sklearn — no external native dependencies needed).
"""

import json
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.features.synergy_features import build_candidate_features, compute_champion_scores

# Standard ranked draft order
DRAFT_ORDER = [
    ("Blue", 0), ("Red", 0), ("Red", 1), ("Blue", 1), ("Blue", 2),
    ("Red", 2), ("Red", 3), ("Blue", 3), ("Blue", 4), ("Red", 4),
]


def build_training_data(comp_df, embed_dict, champ_scores):
    """
    Build training data by simulating draft order for each historical match.
    Each pick step becomes a training sample with features and comp_score label.
    """
    X_list = []
    y_list = []
    skipped = 0

    for match_id, group in comp_df.groupby("match_id"):
        blue = group[group["team_id"] == 100]
        red = group[group["team_id"] == 200]
        if len(blue) != 5 or len(red) != 5:
            skipped += 1
            continue

        blue_names = blue["champion_name"].tolist()
        red_names = red["champion_name"].tolist()
        # Team score is the same for all players on the team, just grab the first
        blue_score = float(blue["comp_score"].iloc[0])
        red_score = float(red["comp_score"].iloc[0])

        if not all(n in embed_dict for n in blue_names + red_names):
            skipped += 1
            continue

        blue_picked_so_far = []
        red_picked_so_far = []

        for side, slot in DRAFT_ORDER:
            if side == "Blue":
                champ = blue_names[slot]
                allies = list(blue_picked_so_far)
                enemies = list(red_picked_so_far)
                label = blue_score
            else:
                champ = red_names[slot]
                allies = list(red_picked_so_far)
                enemies = list(blue_picked_so_far)
                label = red_score

            features = build_candidate_features(
                champ, allies, enemies, embed_dict, champ_scores
            )
            X_list.append(features)
            y_list.append(label)

            if side == "Blue":
                blue_picked_so_far.append(champ)
            else:
                red_picked_so_far.append(champ)

    if skipped > 0:
        print(f"  Skipped {skipped} matches")

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def train_draft_model(X_train, y_train, X_val=None, y_val=None):
    """
    Train a HistGradientBoosting regressor for draft score prediction.

    Uses sklearn's HistGradientBoostingRegressor which is inspired by
    LightGBM but requires no external native libraries.
    """
    model = HistGradientBoostingRegressor(
        max_iter=300,
        max_leaf_nodes=63,
        learning_rate=0.05,
        max_depth=None,
        min_samples_leaf=20,
        l2_regularization=1e-4,
        random_state=42,
        verbose=1,
        validation_fraction=0.1 if X_val is None else None,
        n_iter_no_change=50,
        early_stopping=True if X_val is None else False,
    )

    model.fit(X_train, y_train)
    return model


def predict_score(model, X):
    """Get predicted comp score from the trained model."""
    return model.predict(X)


def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return metrics dict."""
    preds = predict_score(model, X_test)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
    }


def save_draft_model(model, path):
    """Save the trained model using joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, str(path))


def load_draft_model(path):
    """Load a trained model from joblib file."""
    return joblib.load(str(path))
