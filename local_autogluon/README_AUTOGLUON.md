# AutoGluon LoL Draft Pipeline

A drop-in **AutoML** baseline for the same LoL draft win-prediction task as
the main `lol_draft_pipeline.py`, but using
[AutoGluon Tabular](https://auto.gluon.ai/) instead of our hand-rolled
LightGBM / TeamCompNet / Wide & Deep / Stacker.

> Whole `local_autogluon/` directory is **git-ignored**.

---

## What is AutoGluon and why use it here

AutoGluon Tabular is Amazon's open-source **AutoML** library. Given a
`(features, label)` DataFrame it automatically:

1. **Trains many base learners**: LightGBM, XGBoost, CatBoost, Random
   Forest, Extra Trees, K-NN, FastAI tabular NN, Torch tabular NN.
2. **Bagging** each base learner across K folds (default K=8) for OOF
   predictions.
3. **Multi-layer stacking**: 2nd-level learners trained on the OOF
   predictions of layer 1, optionally a 3rd weighted ensemble.
4. **Greedy selection ensemble** at the top, à la Caruana 2004.
5. **Built-in feature preprocessing**: missing-value handling, categorical
   encoding (via integer mapping → tree models, or one-hot for linear),
   text features, datetime features.
6. **Time-budgeted training** — you specify `time_limit` and AutoGluon
   stops once the budget is spent, picking the best model so far.

For our problem this gives a strong, reproducible baseline that:
- Eliminates manual model selection / hyperparam tuning.
- Provides a sanity check on whether our hand-coded `Stacker` is competitive.
- Often **out-of-the-box matches or exceeds** what we'd get from manual
  tuning at 10 minutes of compute, and continues improving with budget.

The trade-off:
- **Heavy install** (~1-2 GB on first run, pulls torch + lightgbm + xgboost
  + catboost).
- **Black-box-ish**: harder to introspect than our explicit pipeline.
- **No bespoke ranking-loss / policy-head / MCTS recommender** - it only
  optimises win-prob accuracy. The recommender step here is a thin
  greedy top-k wrapper that asks the trained predictor for `P(blue wins)`
  for each candidate completion.

---

## How AutoGluon's stacking compares to our hand-rolled `Stacker`

| | Our `Stacker` | AutoGluon |
|---|---|---|
| Base models | 4 (baseline LGB / TeamCompNet / hybrid / Wide&Deep) | ~7 (LGB / XGB / CatBoost / RF / ET / KNN / NN) |
| Bagging | none (single train + val) | K-fold OOF (default 8) |
| Stacking layers | 1 (logistic regression on val preds) | up to 3, with greedy selection ensemble at top |
| Hyperparam search | none | Bayesian / random over a search space |
| Calibration | isotonic on val | implicit through ensembling |
| Compute budget | ~minutes | minutes-to-hours, configurable |
| Code surface | ~150 lines | ~150 lines wrapper + 100k LOC inside autogluon |

For a final-project demo, having both is great: AutoGluon as the AutoML
upper bound, our pipeline as the bespoke deep-model story.

---

## Install

```bash
pip install -r local_autogluon/requirements_autogluon.txt
```

Notes:
- First install pulls **~1-2 GB** of dependencies (torch, lightgbm,
  xgboost, catboost). Be patient.
- macOS: LightGBM still needs `brew install libomp` if you haven't already.
- AutoGluon is happy with CPU-only; Torch backend will use Apple MPS / CUDA
  automatically when available.

---

## Train

```bash
# Quick smoke (5 min budget, default preset)
python local_autogluon/autogluon_pipeline.py train \
  --data-dir data --artifacts-dir artifacts \
  --out-dir local_autogluon/predictor \
  --time-limit 300 --preset medium_quality

# Production run (good_quality preset, 30 min budget)
python local_autogluon/autogluon_pipeline.py train \
  --time-limit 1800 --preset good_quality

# Best-quality (much heavier, hours)
python local_autogluon/autogluon_pipeline.py train \
  --time-limit 3600 --preset best_quality
```

Presets in increasing cost / quality:
- `medium_quality` — a few base learners, no extensive bagging. Fast.
- `good_quality` — adds bagging + 1 stack layer. Recommended starting point.
- `high_quality` — more base models + 2 stack layers.
- `best_quality` — all base models, deep bagging, exhaustive search.

The pipeline reuses the same data loader as `lol_draft_pipeline.py` so
schema detection (patch / rank / bans) auto-applies if those columns are
in your CSV.

---

## Evaluate

```bash
python local_autogluon/autogluon_pipeline.py evaluate \
  --out-dir local_autogluon/predictor
```

Prints test metrics + AutoGluon's leaderboard table (one row per ensemble
member with its OOF + test scores), so you can see which base learners
contributed the most.

---

## Recommend

```bash
python local_autogluon/autogluon_pipeline.py recommend \
  --out-dir local_autogluon/predictor \
  --side blue --role top \
  --blue-picks 'jungle=LeeSin,mid=Ahri,adc=Jinx,support=Lulu' \
  --red-picks  'top=Fiora,jungle=Hecarim,mid=Orianna,adc=Kaisa,support=Nautilus' \
  --top-k 5
```

For each legal candidate:
1. Build a complete-draft feature row with that candidate filling the
   `(side, role)` slot.
2. Batched single `predict_proba` call.
3. Sort by win prob from the picking side's perspective.
4. Append synergy / counter scores (computed from the train-only stats).

---

## Outputs

```
local_autogluon/predictor/
├── ag-... internal AutoGluon directory (models, metadata, fold preds)
├── leaderboard_test.csv   - all ensemble members with test scores
├── test_metrics.json      - same metrics format as our main pipeline
├── vocab.json             - champion -> int mapping (for recommend)
├── handcrafted.pkl        - synergy/counter stats (train-only fit)
└── feature_columns.json   - ordered feature list to keep predict() consistent
```

---

## Common gotchas

- **First training prints a flood of progress logs** - that's normal.
  AutoGluon trains many models in series.
- **OOM on `best_quality` preset** with the new 7.9k matches data is
  unlikely (small data), but for the big `compositions_50k.csv` set
  you may want to cap memory by passing fewer presets.
- **AutoGluon predictions don't have a calibrator wrapper**. They're
  already well-calibrated due to ensembling, but if you want to compose
  them with the main pipeline's `Recommender`, fit one yourself on val
  predictions.
- **Pickle compatibility**: AutoGluon saves to `out_dir/`, you must use
  the same Python + AutoGluon version to reload. Don't mix major versions.

---

## Layout

```
local_autogluon/
├── autogluon_pipeline.py        # train / evaluate / recommend CLI
├── requirements_autogluon.txt    # autogluon.tabular pin
└── README_AUTOGLUON.md           # this file
```
