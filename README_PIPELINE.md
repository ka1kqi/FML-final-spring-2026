# LoL Draft Pipeline (`lol_draft_pipeline.py`)

End-to-end **draft-time** win prediction and champion recommendation
pipeline for League of Legends.  The whole pipeline is in a single
Python file — load `data/`, build features, train four models, run
recommendations.  Strict no-leakage discipline: only blue/red picks
+ roles + side (and bans / patch / rank when present) feed the main
models.  Post-game stats (kda, gold, damage, vision, items) are
**explicitly excluded** and audited at every run.

> See **[FINAL_REPORT.md](FINAL_REPORT.md)** for the full project
> narrative (data provenance, methodology, AutoGluon comparison,
> time-split caveats, future-work levers).

---

## What gets trained

| Model | Backend | Purpose |
| --- | --- | --- |
| `baseline` | LightGBM (HistGB fallback) | Strong tabular baseline over role-slot champion ids + handcrafted synergy / counter / win-rate features. |
| `teamcompnet` | PyTorch | Champion embeddings + role embeddings + pairwise (synergy / matchup) interaction + MLP head. |
| `hybrid` | LightGBM | Baseline features **plus** team-embedding features extracted from a frozen TeamCompNet. Usually the best general-purpose recommender scorer. |
| `wide_deep` | PyTorch | Wide branch (one-hot role-slot LR) + Deep branch (TeamCompNet body), summed or concatenated. |

A fifth deliverable, the **top-k / beam-search recommender**, wraps any
trained scorer behind a unified API and exposes both a CLI and a
`Recommender` class.

---

## Setup

```bash
brew install libomp           # macOS only — required by LightGBM
pip install -r requirements.txt
```

The pipeline only relies on `numpy`, `pandas`, `scikit-learn`,
`lightgbm`, and `torch`.  If LightGBM is unavailable it gracefully falls
back to `sklearn.ensemble.HistGradientBoostingClassifier`.

---

## Data assumptions

The script auto-detects the long-form participant CSV under `data/`,
preferring `data/processed/*.csv` over `data/raw/*.csv`. It expects
(case-insensitive) columns:

* `match_id` (or `matchId`), `champion_name` (or `championName`),
  `team_id` (100 / 200), `position` (or `teamPosition`:
  `TOP / JUNGLE / MIDDLE / BOTTOM / UTILITY`), `win`.
* Optional: `patch` (or `gameVersion`), `rank`, `bans`,
  `timestamp` (or `gameCreation`). When present they are auto-wired
  into the LightGBM feature matrix as a categorical id, a continuous
  numeric (`patch_numeric`, `match_time_days`), and ban one-hots.

### Recommended workflow for fresh Riot match-v5 data

```bash
# Pivot bans, drop post-game columns, write to data/processed/matches.csv
python prepare_riot_v5_data.py
```

The pipeline then auto-uses the processed file. Time-based train/val/test
split activates automatically when `timestamp` is present.

### Getting the raw matches.csv

The 44 MB raw match-v5 dump is **not committed** (too large for git). Grab
it from the cloud-storage link in [FINAL_REPORT.md](FINAL_REPORT.md#where-to-get-the-raw-data),
place at `~/Downloads/data_processed/matches.csv`, then run the conversion
script above.

Each match is pivoted into one row with role-specific champion columns:
`{blue,red}_{top,jungle,mid,adc,support}_champion`.  Riot positions
`MIDDLE` → `mid`, `BOTTOM` → `adc`, `UTILITY` → `support`.

If your data carries a usable `timestamp`, splits are time-based
(70 / 15 / 15).  Otherwise a stratified random split is used and the
script logs a warning — patches drift, so time-based is preferred.

---

## CLI

```bash
# Train everything end-to-end
python lol_draft_pipeline.py train --data-dir data --artifacts-dir artifacts

# Train just one stage
python lol_draft_pipeline.py train-baseline    --data-dir data --artifacts-dir artifacts
python lol_draft_pipeline.py train-teamcompnet --data-dir data --artifacts-dir artifacts
python lol_draft_pipeline.py train-hybrid      --data-dir data --artifacts-dir artifacts
python lol_draft_pipeline.py train-wide-deep   --data-dir data --artifacts-dir artifacts

# Reload all artifacts and recompute test metrics + recommender hit-rate
python lol_draft_pipeline.py evaluate --data-dir data --artifacts-dir artifacts

# Get top-k draft recommendations
python lol_draft_pipeline.py recommend \
  --artifacts-dir artifacts \
  --blue-picks "jungle=LeeSin,mid=Ahri,adc=Jinx,support=Lulu" \
  --red-picks  "top=Fiora,jungle=Hecarim,mid=Orianna,adc=KaiSa,support=Nautilus" \
  --bans "Yone,Aatrox,Rell" \
  --side blue --role top --top-k 5 --model hybrid

# Beam-search lookahead (depth=cfg.beam_depth, default 2; opponent minimax)
python lol_draft_pipeline.py recommend \
  --side blue --role top --top-k 5 --model wide_deep --beam-search
```

Speed knobs you can pass to any train command:

```
--max-rows 5000           # cap matches loaded (smoke / dev)
--fast-dev-run            # 1 epoch + 2k matches (CI-style smoke)
--epochs 30 --batch-size 256 --embedding-dim 32 --hidden-dim 128
--learning-rate 1e-3 --patience 5
--lgb-n-estimators 500 --lgb-learning-rate 0.03
```

---

## Outputs

After `python lol_draft_pipeline.py train` you'll find:

```
artifacts/
├── config.json
├── champion_to_idx.json          # vocabulary (UNK at index 0)
├── handcrafted_stats.pkl         # synergy / counter / champion winrate
├── lightgbm_baseline.pkl         # model bundle
├── lightgbm_baseline_features.json
├── teamcompnet.pt                # PyTorch state dict
├── champion_embeddings.npy       # extracted embedding matrix
├── lightgbm_with_embeddings.pkl
├── lightgbm_with_embeddings_features.json
├── wide_deep.pt
├── metrics_*.json                # per-model val + test metrics + history
├── metrics.json                  # aggregated summary
└── model_comparison.csv          # accuracy / f1 / auc / log_loss / brier
```

`evaluate` adds `metrics_evaluate.json` and `metrics_recommender.json`
(recall@1/3/5 + MRR over hidden-pick reconstructions on the test split).

`recommend` writes the latest top-k to `artifacts/recommendation_examples.json`
and prints a table:

```
Rank Champion          WinProb   Delta     Synergy   Counter   Notes
------------------------------------------------------------------------------------------
1    Malphite          0.5481   +0.0312   +0.0182   +0.0124   strong engage; good with Orianna/Jinx
2    Ornn              0.5410   +0.0241   +0.0140   +0.0061   stable frontline; scaling
...
```

---

## Leakage discipline

Every run prints:

```
==== LEAKAGE AUDIT ====
Included (draft-time only): champion (per role × per side), side, ...
Confirmed excluded post-game columns: kills, deaths, assists, ...
=======================
```

`POST_GAME_COLUMNS` lists the banned set.  Synergy, counter, and
champion-winrate stats are fit on the **training split only** with
Bayesian smoothing so val/test don't leak through residuals.

---

## Recommender internals

* `Recommender.top_k(state, side, role, k)` — greedy enumeration over
  every legal candidate, ranked by post-pick win probability for the
  picking side.  Supports a batched scorer for ~30× speed-ups.
* `Recommender.beam_search(state, side, role, beam_width, depth, k)` —
  minimax beam search.  At our turns the top `beam_width` children are
  retained; at opponent turns we assume they pick the move that
  minimises our value.  Falls back to `top_k` for `depth <= 1`.
* `evaluate_recommender(recommender, test_df, vocab, n_samples)` —
  hides one slot per match, asks the recommender to rank candidates,
  reports recall@k and MRR.

The recommender is model-agnostic: pass any `score_fn(DraftState) ->
blue_win_prob` (and optionally a `batch_score_fn` for speed).

---

## Hyperparameter defaults

| Group | Param | Default |
| --- | --- | --- |
| PyTorch | embedding_dim | 32 |
| | hidden_dim | 128 |
| | dropout | 0.2 |
| | batch_size | 256 |
| | learning_rate | 1e-3 |
| | epochs | 30 |
| | patience | 5 |
| LightGBM | n_estimators | 500 |
| | learning_rate | 0.03 |
| | num_leaves | 31 |
| Recommender | beam_width | 5 |
| | beam_depth | 2 |
| | top_k | 5 |

All knobs live on the `PipelineConfig` dataclass; CLI flags override
any of them.
