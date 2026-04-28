"""
local_autogluon/autogluon_pipeline.py
=====================================

Drop-in AutoGluon Tabular pipeline for the LoL draft win-prob task.
Reuses the main repo's data loader / feature engineering, but replaces all
of our hand-written models (LightGBM / TeamCompNet / Wide&Deep / Stacker)
with AutoGluon's automated stacking ensemble.

Three subcommands:

    train     - load data, build features, fit AutoGluon predictor
    evaluate  - score a saved predictor on the test split
    recommend - greedy top-k recommendation using the saved predictor

The whole ``local_autogluon/`` folder is git-ignored.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import lol_draft_pipeline as P  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _build_feature_frame(df: pd.DataFrame, vocab, handcrafted, cfg,
                         extra_vocabs=None) -> pd.DataFrame:
    """Reuse the pipeline's baseline feature matrix and append the label."""
    X, _ = P.build_baseline_feature_matrix(df, vocab, handcrafted, cfg, extra_vocabs or {})
    X = X.copy()
    X["blue_win"] = df["blue_win"].values
    return X


def _import_autogluon():
    try:
        from autogluon.tabular import TabularPredictor  # noqa: WPS433
        return TabularPredictor
    except Exception as exc:
        raise SystemExit(
            "AutoGluon is not installed.\n"
            "  pip install -r local_autogluon/requirements_autogluon.txt\n"
            f"Underlying error: {exc}"
        )


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #


def train(args: argparse.Namespace) -> None:
    TabularPredictor = _import_autogluon()

    cfg = P.PipelineConfig(
        data_dir=args.data_dir, artifacts_dir=args.artifacts_dir,
        max_rows=args.max_rows, fast_dev_run=args.fast_dev_run,
    )
    splits, vocab, handcrafted, _, extra_vocabs = P.prepare_data(cfg)

    train_df = _build_feature_frame(splits.train, vocab, handcrafted, cfg, extra_vocabs)
    val_df = _build_feature_frame(splits.val, vocab, handcrafted, cfg, extra_vocabs)
    test_df = _build_feature_frame(splits.test, vocab, handcrafted, cfg, extra_vocabs)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[autogluon] training: train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print(f"[autogluon] preset={args.preset}  time_limit={args.time_limit}s")

    # AutoGluon's medium_quality preset disables bagging, which makes
    # use_bag_holdout incompatible. Keep our val rows in train_data and let
    # AutoGluon's internal holdout do the validation - far more compatible.
    combined = pd.concat([train_df, val_df], axis=0, ignore_index=True)

    predictor = TabularPredictor(
        label="blue_win",
        problem_type="binary",
        eval_metric="roc_auc",
        path=str(out_dir),
        verbosity=3,  # show why a model fails
    )
    fit_kwargs: Dict[str, object] = dict(
        train_data=combined,
        time_limit=args.time_limit,
        ag_args_fit={"num_cpus": 4},  # avoid M-series "auto" detection bug
    )
    if args.hyperparameters_only_gbm:
        # Robust path: only LightGBM. Survives M-series + Python 3.13 backend
        # incompatibilities that silently break XGB/CatBoost/Torch NN.
        fit_kwargs["hyperparameters"] = {"GBM": [{}, {"extra_trees": True}]}
    else:
        fit_kwargs["presets"] = args.preset
    predictor.fit(raise_on_no_models_fitted=True, **fit_kwargs)

    # Test metrics
    y_te = test_df["blue_win"].values
    test_features = test_df.drop(columns=["blue_win"])
    proba = predictor.predict_proba(test_features)
    pos_label = 1 if 1 in proba.columns else proba.columns[-1]
    test_prob = proba[pos_label].values
    metrics = P.compute_metrics(y_te, test_prob)
    print("\n=== test metrics ===")
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "log_loss", "brier_score"):
        print(f"  {k}: {metrics[k]:.4f}")

    leaderboard = predictor.leaderboard(test_df, silent=True)
    leaderboard.to_csv(out_dir / "leaderboard_test.csv", index=False)
    print(f"\n[autogluon] leaderboard saved to {out_dir/'leaderboard_test.csv'}")
    print(leaderboard.head(10).to_string(index=False))

    # Persist the artefacts the recommend subcommand needs.
    (out_dir / "vocab.json").write_text(json.dumps(vocab))
    P.save_pickle(out_dir / "handcrafted.pkl", handcrafted)
    (out_dir / "feature_columns.json").write_text(json.dumps(list(test_features.columns)))
    (out_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))


# --------------------------------------------------------------------------- #
# evaluate
# --------------------------------------------------------------------------- #


def evaluate(args: argparse.Namespace) -> None:
    TabularPredictor = _import_autogluon()
    out_dir = Path(args.out_dir).resolve()
    predictor = TabularPredictor.load(str(out_dir))

    cfg = P.PipelineConfig(data_dir=args.data_dir, artifacts_dir=args.artifacts_dir,
                           max_rows=args.max_rows)
    splits, vocab, handcrafted, _, extra_vocabs = P.prepare_data(cfg)
    test_df = _build_feature_frame(splits.test, vocab, handcrafted, cfg, extra_vocabs)
    y_te = test_df["blue_win"].values
    proba = predictor.predict_proba(test_df.drop(columns=["blue_win"]))
    pos_label = 1 if 1 in proba.columns else proba.columns[-1]
    test_prob = proba[pos_label].values
    metrics = P.compute_metrics(y_te, test_prob)
    print(json.dumps(
        {k: metrics[k] for k in
         ("accuracy", "precision", "recall", "f1", "roc_auc", "log_loss", "brier_score")},
        indent=2,
    ))

    print("\nLeaderboard on test:")
    print(predictor.leaderboard(test_df, silent=True).head(15).to_string(index=False))


# --------------------------------------------------------------------------- #
# recommend
# --------------------------------------------------------------------------- #


def recommend(args: argparse.Namespace) -> None:
    TabularPredictor = _import_autogluon()
    out_dir = Path(args.out_dir).resolve()
    predictor = TabularPredictor.load(str(out_dir))
    vocab = json.loads((out_dir / "vocab.json").read_text())
    handcrafted = P.load_pickle(out_dir / "handcrafted.pkl")
    feature_cols = json.loads((out_dir / "feature_columns.json").read_text())
    cfg = P.PipelineConfig()

    state = P.DraftState(
        blue_picks=P.parse_pick_string(args.blue_picks),
        red_picks=P.parse_pick_string(args.red_picks),
        bans=P.parse_bans_string(args.bans),
    )
    used = state.used_champions()
    legal = [c for c in vocab.keys() if c != P.UNKNOWN_TOKEN and c not in used]

    # Build N candidate rows (vectorised, single predict_proba call).
    base_row = {f"{s}_{r}_champion": P.UNKNOWN_TOKEN
                for s in ("blue", "red") for r in P.ROLES}
    for r, c in state.blue_picks.items():
        base_row[f"blue_{r}_champion"] = c
    for r, c in state.red_picks.items():
        base_row[f"red_{r}_champion"] = c
    target_col = f"{args.side}_{args.role}_champion"
    rows = []
    for cand in legal:
        row = dict(base_row)
        row[target_col] = cand
        rows.append(row)
    df = pd.DataFrame(rows)
    feats, _ = P.build_baseline_feature_matrix(df, vocab, handcrafted, cfg)
    feats = feats.reindex(columns=feature_cols, fill_value=0)
    proba = predictor.predict_proba(feats)
    pos_label = 1 if 1 in proba.columns else proba.columns[-1]
    blue_wp = proba[pos_label].values
    my_wp = blue_wp if args.side == "blue" else (1 - blue_wp)

    order = np.argsort(-my_wp)[: args.top_k]
    print()
    print(f"{'Rank':<5}{'Champion':<18}{'WinProb':<10}{'Synergy':<10}{'Counter':<10}")
    print("-" * 60)
    for rank, idx in enumerate(order, 1):
        cand = legal[idx]
        my_picks = list(state.picks_for(args.side).values())
        opp_picks = list(state.picks_for("red" if args.side == "blue" else "blue").values())
        syn = float(np.mean([handcrafted.synergy(cand, c) for c in my_picks])) if my_picks else 0.0
        ctr = float(np.mean([handcrafted.counter(cand, e) for e in opp_picks])) if opp_picks else 0.0
        print(f"{rank:<5}{cand:<18}{my_wp[idx]:.4f}    {syn:+.4f}   {ctr:+.4f}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)

    def _common(sp):
        sp.add_argument("--data-dir", default="data")
        sp.add_argument("--artifacts-dir", default="artifacts")
        sp.add_argument("--out-dir", default="local_autogluon/predictor")
        sp.add_argument("--max-rows", type=int, default=None)
        sp.add_argument("--fast-dev-run", action="store_true")

    pt = sub.add_parser("train")
    _common(pt)
    pt.add_argument("--time-limit", type=int, default=600,
                    help="Seconds. AutoGluon stops once this budget is spent.")
    pt.add_argument("--preset", default="medium_quality",
                    choices=["medium_quality", "good_quality", "high_quality", "best_quality"])
    pt.add_argument("--hyperparameters-only-gbm", action="store_true",
                    help="Robust mode: only train LightGBM (skips backends that "
                         "may fail silently on M-series Mac + Python 3.13).")

    pe = sub.add_parser("evaluate")
    _common(pe)

    pr = sub.add_parser("recommend")
    _common(pr)
    pr.add_argument("--blue-picks", default="")
    pr.add_argument("--red-picks", default="")
    pr.add_argument("--bans", default="")
    pr.add_argument("--side", required=True, choices=["blue", "red"])
    pr.add_argument("--role", required=True, choices=list(P.ROLES))
    pr.add_argument("--top-k", type=int, default=5)

    args = p.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "recommend":
        recommend(args)


if __name__ == "__main__":
    main()
