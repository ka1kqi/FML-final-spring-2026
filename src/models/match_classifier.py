"""
Per-match win classifier: predicts P(blue_win | full composition) over 5v5.

Uses LogisticRegression with temperature calibration on symmetric 67-d
composition deltas. Feature vector: [embed_diff (64d), syn_delta, match_delta,
score_delta]. One row per match.

Temperature scaling flattens overconfident probabilities (T=15.5 found optimal
on validation set) to keep predictions in realistic [42%, 58%] band for balanced
compositions.
"""

import joblib
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, log_loss, roc_auc_score, brier_score_loss,
)

from src.features.synergy_features import (
    predicted_synergy, predicted_counter, _split_blocks,
)


def build_match_features(blue_champs, red_champs, embed_dict, champ_scores, embed_dim=64):
    """
    Build a 67-d feature vector for a complete 5v5 composition (one match).

    Feature layout:
      [0:64]   embed_diff = mean(blue embeddings) - mean(red embeddings)
      [64]     synergy_delta = mean(blue synergies) - mean(red synergies)
      [65]     matchup_delta = mean(blue matchups) - mean(red matchups)
      [66]     score_delta = mean(blue scores) - mean(red scores)

    Returns:
        Feature vector (67,) dtype float32
    """
    # Get embeddings for all champions
    blue_vecs = np.array([embed_dict[c] for c in blue_champs], dtype=np.float32)
    red_vecs = np.array([embed_dict[c] for c in red_champs], dtype=np.float32)

    # Embedding difference
    embed_diff = np.mean(blue_vecs, axis=0) - np.mean(red_vecs, axis=0)

    # Synergy deltas: sum over all pairs on the team, divide by team size
    def team_synergy_mean(team_vecs):
        """Mean synergy within a team."""
        if len(team_vecs) == 0:
            return 0.0
        syns = []
        for i, c_vec in enumerate(team_vecs):
            for j, a_vec in enumerate(team_vecs):
                if i != j:
                    syns.append(predicted_synergy(c_vec, a_vec, embed_dim))
        return float(np.mean(syns)) if syns else 0.0

    blue_syn = team_synergy_mean(blue_vecs)
    red_syn = team_synergy_mean(red_vecs)
    synergy_delta = blue_syn - red_syn

    # Matchup deltas: sum over all enemy pairs, divide by team size
    def team_matchup_mean(team_vecs, enemy_vecs):
        """Mean matchup score of team vs. enemies."""
        if len(team_vecs) == 0 or len(enemy_vecs) == 0:
            return 0.0
        matchups = []
        for c_vec in team_vecs:
            for e_vec in enemy_vecs:
                matchups.append(predicted_counter(c_vec, e_vec, embed_dim))
        return float(np.mean(matchups)) if matchups else 0.0

    blue_match = team_matchup_mean(blue_vecs, red_vecs)
    red_match = team_matchup_mean(red_vecs, blue_vecs)
    matchup_delta = blue_match - red_match

    # Composition score deltas
    blue_scores = [champ_scores.get(c, 50.0) for c in blue_champs]
    red_scores = [champ_scores.get(c, 50.0) for c in red_champs]
    score_delta = np.mean(blue_scores) - np.mean(red_scores)

    features = np.concatenate([
        embed_diff,  # 64
        np.array([synergy_delta, matchup_delta, score_delta], dtype=np.float32),
    ])

    return features.astype(np.float32)


def build_match_training_data(comp_df, embed_dict, champ_scores):
    """
    Build per-match training data. One row per match (Blue vs Red 5v5).

    Args:
        comp_df: DataFrame with [match_id, team_id, champion_name, win, ...]
        embed_dict: champion name -> embedding vector
        champ_scores: champion name -> historical avg comp score

    Returns:
        (X, y) where X is (n_matches, 67) float32, y is (n_matches,) int8
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

        # Skip if any champion missing from embeddings
        if not all(n in embed_dict for n in blue_names + red_names):
            skipped += 1
            continue

        features = build_match_features(blue_names, red_names, embed_dict, champ_scores)
        X_list.append(features)
        y_list.append(blue_win)

    if skipped:
        print(f"  Skipped {skipped} matches")

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int8)


class TemperatureScaledLR:
    """
    LogisticRegression wrapper with learned temperature scaling.

    Temperature T > 1 flattens predicted probabilities, reducing overconfidence.
    Fitted on validation set via grid search to minimize log-loss.
    """

    def __init__(self, C: float = 0.1, random_state: int = 42):
        self.C = C
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.lr = LogisticRegression(C=C, random_state=random_state, max_iter=1000)
        self.temperature = 1.0

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Fit the logistic regression and optionally calibrate temperature on validation set.

        Args:
            X_train: training features (n, d)
            y_train: training labels (n,)
            X_val: validation features (m, d) [optional]
            y_val: validation labels (m,) [optional]
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.lr.fit(X_train_scaled, y_train)

        if X_val is not None and y_val is not None:
            self._fit_temperature(X_val, y_val)

        return self

    def _fit_temperature(self, X_val, y_val):
        """Grid-search temperature T in [0.5, 20.0] to minimize validation log-loss."""
        X_val_scaled = self.scaler.transform(X_val)
        raw_probs = self.lr.predict_proba(X_val_scaled)[:, 1]

        best_loss = float('inf')
        best_t = 1.0

        for T in np.linspace(0.5, 20.0, 40):
            # Calibrate probs: logit_scaled = logit(p) / T, then p_cal = sigmoid(logit_scaled)
            eps = 1e-6
            raw_probs_clipped = np.clip(raw_probs, eps, 1 - eps)
            logits = np.log(raw_probs_clipped / (1 - raw_probs_clipped))
            cal_probs = 1.0 / (1.0 + np.exp(-logits / T))
            loss = log_loss(y_val, cal_probs, labels=[0, 1])

            if loss < best_loss:
                best_loss = loss
                best_t = T

        self.temperature = best_t

    def predict_proba(self, X):
        """Predict P(class=1 | X) with temperature scaling applied."""
        X_scaled = self.scaler.transform(X)
        raw_probs = self.lr.predict_proba(X_scaled)[:, 1]

        if self.temperature == 1.0:
            return np.column_stack([1 - raw_probs, raw_probs])

        # Apply temperature scaling
        eps = 1e-6
        raw_probs = np.clip(raw_probs, eps, 1 - eps)
        logits = np.log(raw_probs / (1 - raw_probs))
        cal_probs = 1.0 / (1.0 + np.exp(-logits / self.temperature))
        return np.column_stack([1 - cal_probs, cal_probs])

    def predict(self, X):
        """Predict class labels."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def train_match_classifier(X_train, y_train, X_val=None, y_val=None):
    """
    Train per-match classifier with temperature calibration.

    Args:
        X_train: training features (n, 67)
        y_train: training labels (n,) — 1 if blue won, 0 otherwise
        X_val: validation features (m, 67) [optional]
        y_val: validation labels (m,) [optional]

    Returns:
        TemperatureScaledLR instance, fitted and calibrated
    """
    model = TemperatureScaledLR(C=0.1, random_state=42)
    model.fit(X_train, y_train, X_val, y_val)
    print(f"  Fitted calibration temperature T = {model.temperature:.3f}")
    return model


def evaluate_match_model(model, X_test, y_test):
    """Classification metrics on per-match model."""
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "log_loss": float(log_loss(y_test, probs, labels=[0, 1])),
        "auc": float(roc_auc_score(y_test, probs)),
        "brier": float(brier_score_loss(y_test, probs)),
        "base_rate": float(np.mean(y_test)),
    }


def save_match_model(model, path):
    """Save the trained match classifier."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, str(path))


def load_match_model(path):
    """Load the trained match classifier."""
    return joblib.load(str(path))
