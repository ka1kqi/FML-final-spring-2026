"""
Per-match win classifier.

One training row per completed 5v5 match (vs. the per-pick model's 10
rows per match). Aggregating reduces label variance roughly 10x and
makes the small composition-level signal extractable.

Features are strictly symmetric under blue<->red swap: every component
is a `blue_minus_red` quantity, so swapping sides flips the sign of
every feature *and* the label, which is the only structurally honest
encoding.
"""

from pathlib import Path
import joblib

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, log_loss, roc_auc_score, brier_score_loss,
)

EMBED_DIM = 64
QUARTER = EMBED_DIM // 4
N_FEATURES = EMBED_DIM + 3  # 67


def _team_synergy_total(vecs):
    """Sum of U_syn[i] . V_syn[j] over all ordered (i,j) pairs in a team, i != j."""
    if len(vecs) < 2:
        return 0.0
    U = np.stack([v[0:QUARTER] for v in vecs])           # (n, 16)
    V = np.stack([v[QUARTER:2*QUARTER] for v in vecs])   # (n, 16)
    M = U @ V.T                                           # (n, n)
    np.fill_diagonal(M, 0.0)
    return float(M.sum())


def _cross_matchup_total(my_vecs, opp_vecs):
    """Sum of U_match[i] . V_match[j] over (my i, opp j) pairs.
       Positive = my team predicted to score above average vs. opp."""
    if not my_vecs or not opp_vecs:
        return 0.0
    Um = np.stack([v[2*QUARTER:3*QUARTER] for v in my_vecs])
    Vm = np.stack([v[3*QUARTER:4*QUARTER] for v in opp_vecs])
    return float((Um @ Vm.T).sum())


def build_match_features(blue_names, red_names, embed_dict, champ_scores):
    """Produce a single 67-d feature vector for one full draft."""
    bv = [embed_dict[n] for n in blue_names]
    rv = [embed_dict[n] for n in red_names]

    blue_mean = np.mean(bv, axis=0)
    red_mean = np.mean(rv, axis=0)
    embed_diff = blue_mean - red_mean

    blue_syn = _team_synergy_total(bv)
    red_syn = _team_synergy_total(rv)

    blue_vs_red = _cross_matchup_total(bv, rv)
    red_vs_blue = _cross_matchup_total(rv, bv)

    blue_comp = float(np.mean([champ_scores.get(n, 50.0) for n in blue_names]))
    red_comp = float(np.mean([champ_scores.get(n, 50.0) for n in red_names]))

    feats = np.empty(N_FEATURES, dtype=np.float32)
    feats[0:EMBED_DIM] = embed_diff
    feats[EMBED_DIM]   = blue_syn - red_syn
    feats[EMBED_DIM+1] = blue_vs_red - red_vs_blue
    feats[EMBED_DIM+2] = blue_comp - red_comp
    return feats


def build_match_training_data(comp_df, embed_dict, champ_scores):
    """One row per match. Label = blue_win in {0, 1}."""
    X_list, y_list = [], []
    skipped = 0
    for match_id, group in comp_df.groupby("match_id"):
        blue = group[group["team_id"] == 100]
        red = group[group["team_id"] == 200]
        if len(blue) != 5 or len(red) != 5:
            skipped += 1
            continue
        blue_names = blue["champion_name"].tolist()
        red_names = red["champion_name"].tolist()
        if not all(n in embed_dict for n in blue_names + red_names):
            skipped += 1
            continue
        blue_win = int(bool(blue["win"].iloc[0]))
        X_list.append(build_match_features(blue_names, red_names, embed_dict, champ_scores))
        y_list.append(blue_win)

    if skipped:
        print(f"  Skipped {skipped} matches")
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int8)


class TemperatureScaledLR:
    """
    Wraps a fitted Pipeline[StandardScaler, LogisticRegression] with
    a temperature parameter T applied to the logit before sigmoid.

    Prediction: p = sigmoid(decision_function(x) / T)

    T > 1 flattens overconfident extremes toward 0.5; T < 1 sharpens.
    Tuned by minimizing validation log-loss.
    """

    def __init__(self, pipeline, temperature: float = 1.0):
        self.pipeline = pipeline
        self.temperature = float(temperature)

    def _scaled_logits(self, X):
        return self.pipeline.decision_function(X) / self.temperature

    def predict_proba(self, X):
        z = self._scaled_logits(X)
        p1 = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _fit_temperature(pipeline, X_val, y_val) -> float:
    """Grid-search T to minimize validation log-loss. Wide range —
    if the model has very weak signal, optimal T diverges (predict
    near-50%), and we need the upper bound to be permissive."""
    logits = pipeline.decision_function(X_val)
    eps = 1e-7
    grid = np.concatenate([
        np.linspace(0.5, 4.9, 45),
        np.linspace(5.0, 20.0, 31),
    ])
    best_T, best_loss = 1.0, float("inf")
    for T in grid:
        z = logits / T
        p = 1.0 / (1.0 + np.exp(-z))
        p = np.clip(p, eps, 1 - eps)
        loss = float(-(y_val * np.log(p) + (1 - y_val) * np.log(1 - p)).mean())
        if loss < best_loss:
            best_T, best_loss = float(T), loss
    return best_T


def train_match_classifier(X_train, y_train, X_val=None, y_val=None):
    """
    Logistic regression with standardized features, then temperature
    scaling on a validation set so output probabilities are calibrated.
    Falls back to T=1.0 if no validation data provided.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs")),
    ])
    pipeline.fit(X_train, y_train)

    if X_val is not None and y_val is not None and len(y_val) > 0:
        T = _fit_temperature(pipeline, X_val, y_val)
        print(f"  Fitted calibration temperature T = {T:.3f}")
    else:
        T = 1.0
    return TemperatureScaledLR(pipeline, temperature=T)


def evaluate_match_model(model, X_test, y_test):
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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, str(path))


def load_match_model(path):
    return joblib.load(str(path))
