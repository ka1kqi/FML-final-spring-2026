# Local Dashboard for the LoL Draft Pipeline

A **local-only** Streamlit + Plotly dashboard for inspecting training
progress, comparing models, exploring champion embeddings, running
recommendations, and auditing leakage on the LoL draft-time pipeline.

> **Why this directory is git-ignored**
> The dashboard is a developer/demo tool — it reads on-disk artifacts
> that contain model weights, metrics, and personal experiment runs.
> None of that should land in git. The repo's top-level `.gitignore`
> already excludes `local_dashboard/`, `artifacts/`, model weights
> (`*.pt`, `*.pth`, `*.npy`, `*.pkl`, ...), and Streamlit caches.

---

## Install

```bash
# 1) Pipeline deps (one-time)
pip install -r requirements.txt           # pandas / numpy / sklearn / lightgbm / torch

# 2) Dashboard-only deps
pip install -r local_dashboard/requirements_dashboard.txt

# (macOS only) LightGBM needs OpenMP at runtime
brew install libomp
```

---

## Produce artifacts

The dashboard reads from `artifacts/runs/<run_id>/`. Run training once
(use `--fast-dev-run` for a 30-second smoke run):

```bash
# Quick smoke run (1 epoch, 2k matches)
python lol_draft_pipeline.py train \
  --data-dir data --artifacts-dir artifacts --fast-dev-run

# Or a full run (auto-generated run id)
python lol_draft_pipeline.py train \
  --data-dir data --artifacts-dir artifacts --run-id auto

# Or pin a specific run id
python lol_draft_pipeline.py train --run-id experiment_42
```

The pipeline writes per-run artifacts into:

```
artifacts/runs/<run_id>/
├── config.json
├── champion_to_idx.json
├── events.jsonl                    # streamed training events
├── metrics_summary.json            # overview tab
├── metrics_baseline.json
├── metrics_teamcompnet.json
├── metrics_hybrid.json
├── metrics_wide_deep.json
├── model_comparison.csv
├── calibration.csv
├── feature_importance.csv          # LightGBM only
├── feature_columns.json
├── predictions_test.csv            # threshold tab + roc/pr curves
├── confusion_matrices.json
├── embedding_champions.csv         # 32-d champion embeddings
├── recommendation_examples.json    # canned beam-search demo
├── leakage_audit.json
├── schema_report.json
└── _latest_run_id.txt              # pointer file
```

---

## Launch the dashboard

```bash
streamlit run local_dashboard/app.py -- --artifacts-dir artifacts
```

Notes:
* The double `--` separates Streamlit's args from ours; everything
  after it is parsed by the dashboard.
* The sidebar lets you change `artifacts-dir` at runtime, pick a run,
  toggle auto-refresh, and choose a refresh interval (2-30 s).
* Auto-refresh is implemented as `time.sleep + st.rerun` so it works
  on any Streamlit ≥ 1.27.

---

## Tabs

| # | Tab | What it shows |
|---|-----|---------------|
| 1 | Overview | Headline metrics, run status, dataset shape, leakage summary, model comparison. |
| 2 | Live Training | Streaming Plotly line charts off `events.jsonl`. Pick model + metric. Tail of recent events. |
| 3 | Model Comparison | Final test table with best-value highlighting, grouped bar chart, normalised radar (`log_loss`/`brier` inverted). |
| 4 | Calibration | Reliability diagram, bucket counts. Falls back to recomputing from `predictions_test.csv` if `calibration.csv` is absent. |
| 5 | Confusion / Threshold | Per-model confusion matrix, threshold slider, ROC + PR curves recomputed live. |
| 6 | Feature Importance | LightGBM feature importance (split + gain). Filter by substring; embedding features are coloured separately. |
| 7 | Champion Embeddings | PCA / UMAP scatter of TeamCompNet champion vectors, with cosine-NN explorer. |
| 8 | Recommendation Playground | Build any draft, run greedy or beam-search recommendations against any trained model. Calls back into `lol_draft_pipeline`. |
| 9 | Beam Search Visualizer | Reads canned `recommendation_examples.json` (or a custom `beam_search_trace.json`) and shows the top-k breakdown. |
| 10 | Leakage & Schema | Renders `leakage_audit.json` with red warnings if any blocklisted column is detected. Also displays `schema_report.json`. |
| 11 | Artifacts Browser | File listing with size / mtime, JSON / CSV / TXT preview, download button. |

---

## Troubleshooting

**"No runs found"**
Train at least once: `python lol_draft_pipeline.py train --fast-dev-run`.
The dashboard reads `artifacts/runs/`. If you used a custom `--artifacts-dir`,
type it into the sidebar.

**`predictions_test.csv` missing**
Older runs (pre-dashboard refactor) didn't emit per-row predictions.
Rerun training; the pipeline now writes them on every run.

**`embedding_champions.csv` missing**
TeamCompNet didn't run (e.g. PyTorch unavailable). The pipeline logs a warning
in that case. Either install torch or train just the LightGBM baseline.

**LightGBM import error on macOS**
LightGBM needs OpenMP: `brew install libomp`. The pipeline auto-falls back
to `sklearn.HistGradientBoosting` if LightGBM is unavailable, in which case
`feature_importance.csv` will be empty (HistGB has no native importance).

**Streamlit fails to start**
Check `streamlit --version` is ≥ 1.27. If you get cache-related warnings,
delete `.streamlit/` in your home directory.

**UMAP isn't an option in the embedding tab**
That projector only appears when `umap-learn` is installed. PCA always works
and is the default.

---

## Cleaning up local artifacts

Everything dashboard-related lives outside git history:

```bash
rm -rf artifacts/                  # all training runs / weights / logs
rm -rf .streamlit/                 # Streamlit's user cache
rm -rf .local_dashboard_cache/     # (created only if you wire one up)
```

Verify nothing leaked into git:

```bash
git status                         # should show no artifacts/ or local_dashboard/
git check-ignore -v artifacts/ local_dashboard/
```

---

## Layout

```
local_dashboard/
├── app.py                          # Streamlit entry point (11 tabs)
├── dashboard_utils.py              # IO helpers, metric/calibration recompute, embedding utils
├── requirements_dashboard.txt      # streamlit + plotly + umap-learn (optional)
├── sample_dashboard_config.json    # documents runtime knobs (not auto-loaded)
└── README_DASHBOARD.md             # this file
```
