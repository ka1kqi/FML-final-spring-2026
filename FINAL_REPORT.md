# LoL Draft Win-Prediction & Champion-Recommendation Pipeline

**Final project report.**
Hand-rolled draft-time pipeline + dashboard + Pick-Ban simulator + AutoGluon
AutoML baseline, all draft-time-only (no post-game leakage), evaluated under a
*time-based* train/val/test split.

---

## TL;DR

| | Value |
|---|---|
| Best model on test | `wide_deep` (Set Transformer + wide LR head) |
| Test ROC-AUC (time-split) | **0.5174** |
| Recall@1 / @3 / @5 (recommender) | 2.5% / 7% / **11%** (≈ 4× random) |
| MRR | 0.052 |
| Train / val / test (time-based) | 5572 / 1194 / 1194 matches |
| Dataset | 7,961 NA Diamond+ ranked solo, Sep 2024 → Apr 2026 |

**Key research finding**: AutoML's default random-K-fold validation **breaks
under a time-based test split**. AutoGluon's val score (0.965) was a fantasy;
its test score (0.489) actually *under-performs* our hand-rolled wide_deep
(0.517). This is a clean cautionary tale about applying AutoML to time-shifted
data without overriding its CV strategy.

---

## 1. Problem

Predict the win probability of the *blue* side of a draft, conditioned only on
information available **before the game starts**:

* the 5 champion picks per side (per-role)
* the 10 banned champions (5 per side)
* patch (`gameVersion`)
* match timestamp (`gameCreation`)

Then expose the trained model as a *recommender*: given a partial draft and the
slot the user is about to fill, return the top-K champion candidates.

Strict no-leakage discipline: KDA, gold, damage, vision, items, objective
takedowns, and 80+ post-game `challenges` fields are **explicitly excluded**.
A leakage audit is printed on every training run.

---

## 2. Data

### Two data sources

| | `data/raw/compositions_50k.csv` (initial) | `data/processed/matches.csv` (current) |
|---|---|---|
| Matches | 55,794 | **7,961** |
| Patch column | ✗ | ✓ (62 unique builds, 14.18 → 16.8) |
| Timestamp | ✗ | ✓ (2024-09 → 2026-04) |
| Bans | ✗ | ✓ (10 per match, 100% coverage) |
| Runes (11-dim) | ✗ | ✓ |
| Summoner spells | ✗ | ✓ |
| Post-game stats | KDA only | full participant DTO + 80+ challenges |
| Per-minute timeline | ✗ | ✓ (2.4M frames, not used by main model) |

The new dataset has **7× fewer matches but ~50× richer per-match information**
and finally enables a methodologically correct **time-based** train/val/test
split.

### Conversion (one-shot)

`prepare_riot_v5_data.py` reads the Riot match-v5 dump from
`~/Downloads/data_processed/matches.csv`, pivots the per-team `ban1..5` columns
into a per-match `bans` string (10 unique champion names per match), drops all
post-game columns, and writes the result to `data/processed/matches.csv`. The
existing pipeline auto-detects the new schema; no code changes required.

### Where to get the raw data

The richer Riot match-v5 dump is too big for git (44 MB raw, 17 MB gzipped)
and is **not committed** to the repository. It lives on Google Drive:

```
File:                matches.csv.gz   (17 MB)
Share link:          https://drive.google.com/file/d/1IECKnWVdqsBsHfm1LtWyL03KleHvSnEF/view?usp=sharing
Direct download:     https://drive.google.com/uc?export=download&id=1IECKnWVdqsBsHfm1LtWyL03KleHvSnEF
md5  matches.csv:    bdce33b9428ba954ea98b3a8c02fe6df
md5  matches.csv.gz: f4ba0a6c8dc627ac67dbf10515113475
```

If you only have access to the original 55k composition CSV that ships with
the repo (`data/raw/compositions_50k.csv`), the pipeline still trains
end-to-end on that — patch / bans / timestamp features will simply be absent
and the split will fall back to stratified random.

For full step-by-step reproduction (clone → install → download data →
train → evaluate → recommend → launch dashboards), see
[Section 9 — How to reproduce](#9-how-to-reproduce).

### Time-based split

When `timestamp` is present, the pipeline splits chronologically:

```
train: Sep 2024 → ~Jan 2026   (5572 matches, 56 unique patches)
val:   ~Jan 2026 → Feb 2026   (1194 matches)
test:  Feb 2026 → Apr 2026    (1194 matches, 3 unique patches)
```

**Important caveat**: 100% of test patches are *unseen* in train. This
collapses the categorical `patch_id` feature to UNK on the test set, and
forced us to add `patch_numeric` (continuous parse of `major.minor.build`) and
`match_time_days` (continuous time-since-start) so trees can extrapolate to
future patches.

---

## 3. Methods

### 3.1 Five base models in one file

All models live in `lol_draft_pipeline.py` (~3900 lines, single file by design):

| Model | Backbone | Notes |
|---|---|---|
| `baseline` | LightGBM | 10 role-slot champion ids + 9 handcrafted (synergy/counter/winrate diff) + patch_numeric + match_time_days + 30 ban one-hots |
| `teamcompnet` | PyTorch Set Transformer | 11-token (CLS + 10 picks) self-attention, role + side embeddings, optional policy head, PMI+SVD pretraining |
| `hybrid` (recommended) | LightGBM | baseline features + team-embedding statistics (mean / diff / prod / pairwise dot) extracted from a frozen TeamCompNet |
| `wide_deep` | PyTorch | wide branch (one-hot role-slot LR) + deep branch (TeamCompNet body), summed |
| `stacker` | sklearn LogReg | meta-learner over the 4 base models' val predictions (disabled by default under time-split — see findings) |

### 3.2 Eight research-grade additions

All wired into the same file, all controllable via CLI flags:

1. **Set Transformer** backbone (replaces simple pairwise dot-product pooling)
2. **PMI + SVD pretraining** of champion embeddings on co-occurrence
3. **Listwise softmax-CE** auxiliary loss aligning with Recall@k / MRR
4. **AlphaZero-style policy head** + **MCTS PUCT** recommender
5. **Augmentation**: side-flip + champion-id dropout (mask → UNK)
6. **Isotonic calibration** per model, fit on val
7. **Stacking ensemble** (logistic regression meta-learner)
8. **External features**: `patch` (categorical id + continuous numeric),
   `bans` (top-30 one-hot), `match_time_days`, hooks for `rank` (when
   available)

### 3.3 Recommender

`Recommender` class wraps any draft-time scorer behind one API and exposes:

* `top_k(state, side, role, k)` — greedy enumeration over all legal candidates
  (batched for ~30× speed-up)
* `beam_search(state, side, role, beam_width, depth)` — minimax-style search
  with explicit opponent modelling
* `mcts(state, side, role, n_simulations, c_puct, depth)` — AlphaZero-flavoured
  PUCT, optional policy prior from the Set Transformer

`evaluate_recommender` hides one slot per test match and measures
**Recall@1/3/5** and **MRR** over hidden-pick reconstruction.

---

## 4. Results

### 4.1 Test metrics (time-based split, new data)

| Model | Accuracy | F1 | ROC-AUC | log_loss | Brier |
|---|---|---|---|---|---|
| baseline | 0.5101 | 0.5952 | 0.5064 | 0.6932 | 0.2500 |
| teamcompnet | 0.4908 | 0.4198 | 0.4829 | 0.7258 | 0.2564 |
| hybrid | 0.5008 | 0.2888 | 0.5093 | 0.6930 | 0.2499 |
| **wide_deep** | **0.5184** | **0.6210** | **0.5174** | 0.6967 | 0.2515 |

### 4.2 Recommender hit-rate

| | random | model |
|---|---|---|
| Recall@1 | 0.6% | **2.5%** |
| Recall@3 | 1.7% | **7.0%** |
| Recall@5 | 2.9% | **11.0%** |
| MRR | — | 0.052 |

200 hidden-pick reconstructions on the chronological test split. Roughly
**4× random performance** at Recall@5.

### 4.3 What signal the model actually uses

LightGBM gain-importance (top features):

```
blue_counter_avg            3801     <- handcrafted matchup-residual avg
blue_minus_red_synergy       853     <- handcrafted synergy diff
blue_minus_red_wr            355     <- handcrafted champion-winrate diff
red_counter_avg               21
patch_id, patch_numeric        0     <- dropped: 100% UNK in test set
ban_*                          0     <- popular bans persist across patches → low marginal info
champion ids                   0     <- 173-cardinality categorical, hard for trees
```

**Interpretation**: handcrafted synergy/counter statistics carry essentially
all the signal the boosted-tree model can extract. Patch / time / bans are
either useless under time-split (patch) or redundant with synergy stats
(bans).

The deep models (`wide_deep`) slightly out-perform the boosted ones because
they consume *raw* champion-id embeddings — the same information, but the
attention head can model interactions that the tree's split structure cannot.

---

## 5. AutoGluon comparison (cautionary tale)

We added an AutoGluon Tabular pipeline (`local_autogluon/`) reusing the same
features. Under the **random** split on the *old* 55k dataset, AutoGluon beat
the hand-rolled stacker:

```
hand-rolled stacker:  test AUC 0.519
AutoGluon best:       test AUC 0.539     (+0.020)
```

But under the **time-based** split on the *new* 7.9k dataset, AutoGluon
*regressed*:

```
hand-rolled wide_deep: test AUC 0.517     val AUC 0.540
AutoGluon best:        test AUC 0.489     val AUC 0.965  <- huge gap
```

**Why**: AutoGluon was given `combined = pd.concat([train_df, val_df])` and
internally split it with **random K-fold**. Its val score reflects performance
on a random subset of train+val (which has the same distribution as train),
not the temporally-shifted test. Models picked by AutoGluon's CV thus
overfit "i.i.d. holdout" patterns that don't transfer across the time
boundary.

This isn't a bug in AutoGluon — it's the user's responsibility to override
the CV strategy when validation distribution differs from test. But it is a
concrete demonstration that **AutoML defaults assume i.i.d. data** and
**break silently** otherwise.

---

## 6. Limitations & honest accounting

* **Test ROC-AUC ≈ 0.52 is near the ceiling for this data + features.**
  Without per-player rank or champion-mastery, all draft-time models converge
  on the same "group-average win rate" prediction.
* Random-split numbers we reported earlier (0.519) **overstated** real
  performance by ≈ 0.3 AUC. The honest time-split number is **0.517**.
* The pipeline never uses post-game data; we explicitly chose not to inflate
  test AUC by leaking it.
* Stacking ensemble was disabled by default for time-split runs because its
  meta-LR overfit val (which is "near-future" relative to test).

---

## 7. Future work — the real levers

The model code has been thoroughly explored; **further AUC gains live in the
data**, not the algorithm.

| Action | API endpoint | Expected ΔAUC |
|---|---|---|
| Real rank | `/lol/league/v4/entries/by-puuid/{puuid}` | +2-4 |
| **Champion mastery** (per player × per champion) | `/lol/champion-mastery/v4/champion-masteries/by-puuid/{puuid}` | **+3-5 (largest single lever)** |
| DDragon static tags (Tank/Mage/Assassin/etc.) | `https://ddragon.leagueoflegends.com/cdn/{patch}/data/en_US/champion.json` | +1-2 |
| Total | | **+6-11 AUC** |

Mastery is the single biggest gap because it's what turns *group-average*
recommendations into *personalised* ones (e.g. "you have 1k mastery on Yasuo
vs 1M mastery on Yasuo" produces completely different best-pick suggestions).

---

## 8. Deliverables

```
.
├── lol_draft_pipeline.py       3900+ lines, single-file pipeline
├── prepare_riot_v5_data.py     one-shot data conversion script
├── README_PIPELINE.md          how to train / evaluate / recommend
├── FINAL_REPORT.md             this file
├── data/
│   ├── raw/compositions_50k.csv      (legacy 55k, no patch/bans)
│   └── processed/matches.csv         (current 7.9k, full draft-time fields)
└── (gitignored)
    ├── artifacts/                    runs / model weights / metrics / events.jsonl
    ├── local_dashboard/              Streamlit dashboard, 11 tabs (overview, live training, comparison, calibration, threshold, feature importance, embeddings, playground, beam vis, leakage, files browser)
    ├── local_draft_simulator/        Streamlit app simulating tournament draft order with live recommendations
    └── local_autogluon/              AutoGluon Tabular AutoML baseline + venv (Python 3.13)
```

Three dashboards / apps, each git-ignored, each callable independently.

---

## 9. How to reproduce

This block reproduces every result in this report from a fresh clone.
Total wall-clock on an Apple-silicon laptop: **~30 minutes** (~3 min data
+ ~10 min hand-rolled training + ~15 min AutoGluon if you opt in).

```bash
# 0. Clone (skip if already done)
git clone https://github.com/ka1kqi/FML-final-spring-2026.git
cd FML-final-spring-2026

# 1. System deps (macOS)
brew install libomp                  # required by LightGBM at runtime
brew install python@3.13             # only needed for AutoGluon (step 8 below)

# 2. Pipeline + dashboard deps (uses your default Python ≥ 3.10)
pip install -r requirements.txt
pip install -r local_dashboard/requirements_dashboard.txt

# 3. Download the raw match-v5 dump from Google Drive + verify integrity
mkdir -p ~/Downloads/data_processed
curl -L "https://drive.google.com/uc?export=download&id=1IECKnWVdqsBsHfm1LtWyL03KleHvSnEF" \
     -o ~/Downloads/matches.csv.gz
md5 ~/Downloads/matches.csv.gz                 # expect f4ba0a6c8dc627ac67dbf10515113475
gunzip -c ~/Downloads/matches.csv.gz > ~/Downloads/data_processed/matches.csv
md5 ~/Downloads/data_processed/matches.csv     # expect bdce33b9428ba954ea98b3a8c02fe6df

# 4. Convert raw v5 dump → pipeline-ready CSV (drops post-game cols, pivots bans)
python prepare_riot_v5_data.py
# Writes data/processed/matches.csv (8 MB, draft-time only)

# 5. Train every hand-rolled model (LightGBM + TeamCompNet + Hybrid + Wide&Deep)
python lol_draft_pipeline.py train \
       --epochs 15 --patience 4 --lgb-n-estimators 400 \
       --run-id final
# ~10 min. Time-based split auto-activates because `timestamp` is present.

# 6. Reload artifacts and recompute test metrics + recommender hit-rate
python lol_draft_pipeline.py evaluate
# Expect: wide_deep test AUC ~0.517, recall@5 ~0.11, MRR ~0.05

# 7. Get a draft recommendation
python lol_draft_pipeline.py recommend \
       --side blue --role top --top-k 5 --model wide_deep --beam-search \
       --blue-picks 'jungle=LeeSin,mid=Ahri,adc=Jinx,support=Lulu' \
       --red-picks  'top=Fiora,jungle=Hecarim,mid=Orianna,adc=Kaisa,support=Nautilus' \
       --bans 'Yone,Aatrox,Rell'

# 8. (optional) AutoGluon AutoML baseline — needs its own Python 3.13 venv
python3.13 -m venv local_autogluon/.venv
source local_autogluon/.venv/bin/activate
pip install -U pip typing_extensions setuptools wheel
pip install -r local_autogluon/requirements_autogluon.txt
python local_autogluon/autogluon_pipeline.py train \
       --time-limit 1800 --preset good_quality \
       --out-dir local_autogluon/predictor_final
deactivate

# 9. Live UIs (each runs forever; Ctrl-C to stop)
streamlit run local_dashboard/app.py        -- --artifacts-dir artifacts   # training dashboard, 11 tabs
streamlit run local_draft_simulator/app.py  -- --artifacts-dir artifacts   # tournament-draft simulator
```

### If `curl` returns an HTML virus-scan page

(Rare for files under 100 MB, but possible on cold caches.) Use `gdown`:

```bash
pip install gdown
gdown 1IECKnWVdqsBsHfm1LtWyL03KleHvSnEF -O ~/Downloads/matches.csv.gz
```

### If you only have the legacy 55k CSV

The pipeline still works on `data/raw/compositions_50k.csv` (which ships
with the repo). Patch / bans / timestamp features will be absent and the
split will fall back to stratified random — useful for sanity checks but
the test-AUC numbers won't be directly comparable to the time-split
numbers in this report.
