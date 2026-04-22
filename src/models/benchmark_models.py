from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import torch

from src.inference.champion_vocab import load_champion_vocab
from src.inference.e2e_infer import load_e2e_model
from src.models.baseline import train_logistic_regression, train_random_forest


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def build_dataset():
    data_file = PROJECT_ROOT / "data/raw/compositions_50k.csv"
    if not data_file.exists():
        data_file = PROJECT_ROOT / "src/data/compositions.csv"
    df = pd.read_csv(data_file)

    champ_to_id, _, champions = load_champion_vocab(PROJECT_ROOT)
    num_champs = len(champions)

    X_baseline = []
    blue_id_rows = []
    red_id_rows = []
    y_rows = []

    for _, group in df.groupby("match_id"):
        blue = group[group["team_id"] == 100]
        red = group[group["team_id"] == 200]
        if len(blue) != 5 or len(red) != 5:
            continue

        blue_names = blue["champion_name"].tolist()
        red_names = red["champion_name"].tolist()
        blue_ids = [champ_to_id[name] for name in blue_names]
        red_ids = [champ_to_id[name] for name in red_names]
        label = 1 if bool(blue["win"].iloc[0]) else 0

        x = np.zeros(num_champs * 2, dtype=np.float32)
        for cid in blue_ids:
            x[cid] = 1.0
        for cid in red_ids:
            x[num_champs + cid] = 1.0

        X_baseline.append(x)
        blue_id_rows.append(blue_ids)
        red_id_rows.append(red_ids)
        y_rows.append(label)

    return (
        np.array(X_baseline),
        np.array(blue_id_rows, dtype=np.int64),
        np.array(red_id_rows, dtype=np.int64),
        np.array(y_rows, dtype=np.int64),
        len(champions),
    )


def evaluate_predictions(y_true, prob):
    pred = (prob >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
        "auc": float(roc_auc_score(y_true, prob)),
    }


def run_benchmark() -> dict:
    X, blue_ids, red_ids, y, vocab_size = build_dataset()
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    lr = train_logistic_regression(X_train, y_train)
    lr_prob = lr.predict_proba(X_test)[:, 1]
    lr_metrics = evaluate_predictions(y_test, lr_prob)

    rf = train_random_forest(X_train, y_train)
    rf_prob = rf.predict_proba(X_test)[:, 1]
    rf_metrics = evaluate_predictions(y_test, rf_prob)

    model_path = PROJECT_ROOT / "data/processed/e2e_model.pth"
    e2e = load_e2e_model(model_path, vocab_size=vocab_size)
    with torch.no_grad():
        blue_t = torch.LongTensor(blue_ids[test_idx])
        red_t = torch.LongTensor(red_ids[test_idx])
        e2e_prob = e2e(blue_t, red_t).squeeze(1).numpy()
    e2e_metrics = evaluate_predictions(y_test, e2e_prob)

    return {
        "Logistic Regression": lr_metrics,
        "Random Forest": rf_metrics,
        "End-to-End Transformer": e2e_metrics,
    }


def main():
    results = run_benchmark()

    print("=== Benchmark Results (same test split) ===")
    for name, metrics in results.items():
        print(
            f"{name:24s} | "
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"F1: {metrics['f1']:.4f} | "
            f"AUC: {metrics['auc']:.4f}"
        )


if __name__ == "__main__":
    main()
