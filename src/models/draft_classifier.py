"""
Draft Win Classifier: predicts P(win | candidate champion, draft context).

Trains a HistGradientBoosting binary classifier on features derived from
champion embeddings and draft state. The label is the picking team's
final win/loss for that match — much lower variance than the per-game
champ_score, so the booster can extract real signal.
"""

import joblib
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, log_loss, roc_auc_score, brier_score_loss,
)

from src.features.synergy_features import build_candidate_features

DRAFT_ORDER = [
    ("Blue", 0), ("Red", 0), ("Red", 1), ("Blue", 1), ("Blue", 2),
    ("Red", 2), ("Red", 3), ("Blue", 3), ("Blue", 4), ("Red", 4),
]


def build_training_data(comp_df, embed_dict, champ_scores):
    """
    Simulate the draft order for each historical match. Each pick step
    becomes one training row (features for the candidate; label is
    whether the picking team won that match).
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
        blue_win = int(bool(blue["win"].iloc[0]))
        red_win = 1 - blue_win

        if not all(n in embed_dict for n in blue_names + red_names):
            skipped += 1
            continue

        blue_picked_so_far: list[str] = []
        red_picked_so_far: list[str] = []

        for side, slot in DRAFT_ORDER:
            if side == "Blue":
                champ = blue_names[slot]
                allies, enemies = list(blue_picked_so_far), list(red_picked_so_far)
                label = blue_win
            else:
                champ = red_names[slot]
                allies, enemies = list(red_picked_so_far), list(blue_picked_so_far)
                label = red_win

            features = build_candidate_features(
                champ, allies, enemies, embed_dict, champ_scores
            )
            X_list.append(features)
            y_list.append(label)

            if side == "Blue":
                blue_picked_so_far.append(champ)
            else:
                red_picked_so_far.append(champ)

    if skipped:
        print(f"  Skipped {skipped} matches")

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int8)


def train_draft_model(X_train, y_train, X_val=None, y_val=None):
    """
    HistGradientBoostingClassifier on (features, win) with logistic loss.

    sklearn's API does not accept an external validation set, so we
    always early-stop on a 10% slice of the training data. The X_val/
    y_val args are kept for compatibility but only used post-fit by
    the caller for held-out metrics.
    """
    model = HistGradientBoostingClassifier(
        max_iter=400,
        max_leaf_nodes=31,
        learning_rate=0.03,
        min_samples_leaf=200,
        l2_regularization=1.0,
        random_state=42,
        verbose=1,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        scoring="loss",
    )
    model.fit(X_train, y_train)
    return model


def predict_win_prob(model, X):
    """Return P(win) for each row."""
    return model.predict_proba(X)[:, 1]


def evaluate_model(model, X_test, y_test):
    """Classification metrics that are actually meaningful here."""
    probs = predict_win_prob(model, X_test)
    preds = (probs >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "log_loss": float(log_loss(y_test, probs, labels=[0, 1])),
        "auc": float(roc_auc_score(y_test, probs)),
        "brier": float(brier_score_loss(y_test, probs)),
        "base_rate": float(np.mean(y_test)),
    }


def save_draft_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, str(path))


def load_draft_model(path):
    return joblib.load(str(path))
