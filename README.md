# LoL Team Comp & Item Build Evaluator

A machine-learning pipeline that predicts League of Legends match outcomes from team compositions, in-game statistics, and item builds using data from the Riot Developer API.

---

## Repository Structure

```
ml-final/
├── configs/
│   └── config.yaml                # Hyperparameters, API settings, feature and training configuration
├── data/
│   ├── raw/                       # Raw match JSON pulled from the Riot API (git-ignored)
│   └── processed/                 # Cleaned, tabular CSVs ready for modeling (git-ignored)
├── notebooks/
│   └── eda.ipynb                  # Exploratory data analysis and visualization notebook
├── src/
│   ├── collection/
│   │   ├── riot_api.py            # Thin wrapper around the Riot match-v5 / league-v4 / summoner-v4 endpoints
│   │   └── match_crawler.py       # Seeds ranked players, crawls match histories, deduplicates, and stores JSON
│   ├── preprocessing/
│   │   ├── parser.py              # Parses raw match JSON into a flat DataFrame (one row per match)
│   │   └── filters.py             # Removes remakes, early surrenders, and non-ranked queues
│   ├── features/
│   │   ├── encoding.py            # Champion encoding: one-hot vectors and learned embeddings
│   │   ├── team_comp.py           # Team-level features: champion win rates, synergy/counter matrices, bans
│   │   ├── game_stats.py          # Player stat features: gold, damage dealt/taken, vision score, CS
│   │   └── items.py               # Item build features: end-game item binary vectors per player
│   ├── models/
│   │   ├── baseline.py            # Logistic Regression and Random Forest baselines
│   │   ├── boosting.py            # Gradient-boosted models: XGBoost and LightGBM
│   │   └── neural_net.py          # PyTorch neural network with learned champion embeddings
│   └── evaluation/
│       ├── metrics.py             # Accuracy, F1, AUC-ROC, confusion matrix, significance tests vs 50%
│       └── ablation.py            # Ablation study comparing feature subsets (draft-only → +stats → +items)
├── .env.example                   # Template for RIOT_API_KEY (copy to .env and fill in)
├── .gitignore                     # Ignores data files, caches, secrets
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/<your-org>/ml-final.git
cd ml-final

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Riot API key
cp .env.example .env
# Edit .env and paste your key from https://developer.riotgames.com/
```

---

## Pipeline Overview

1. **Collect** — `src/collection/` pulls ranked match data from the Riot API and stores raw JSON.
2. **Preprocess** — `src/preprocessing/` parses JSON into tabular form and filters out bad games.
3. **Feature Engineering** — `src/features/` builds champion encodings, team-comp features, game stats, and item vectors.
4. **Train** — `src/models/` trains Logistic Regression, Random Forest, XGBoost, and a neural-net model.
5. **Evaluate** — `src/evaluation/` computes metrics and runs an ablation study across feature sets.

---

## Division of Labor

| Member | Module(s) | Deliverable |
|--------|-----------|-------------|
| **Member 1** | `src/collection/` | Riot API client, match crawler, raw data collection |
| **Member 2** | `src/preprocessing/`, `src/features/` | Data parsing, filtering, all feature engineering |
| **Member 3** | `src/models/` | All model implementations and hyperparameter tuning |
| **Member 4** | `src/evaluation/`, `notebooks/` | Metrics, ablation study, EDA notebook, final report |

---

## Data

- **Source:** [Riot Developer API](https://developer.riotgames.com/) — match-v5 endpoint
- **Target Rank:** Diamond (Ranked Solo/Duo, queue 420)
- **Volume:** 20,000–50,000 matches
- **Split:** Time-based (train on earlier patches, test on later patches)

---

## Models

| Model | Type | Purpose |
|-------|------|---------|
| Logistic Regression | Baseline | Interpretable lower bound |
| Random Forest | Ensemble | Non-linear baseline |
| XGBoost / LightGBM | Boosting | Strong tabular learner |
| Neural Network + Embeddings | Deep Learning | Learned champion representations |

---

## References

- [Riot Developer API Docs](https://developer.riotgames.com/apis)
- [LoLDraftAI — Draft Prediction Model](https://loldraftai.com/blog/loldraftai-explained)
- [MDPI 2025 — Ensemble ML for LoL Win Prediction](https://www.mdpi.com/2076-3417/15/10/5241)

---

## Quick Test Commands

```bash
# Benchmark Logistic Regression, Random Forest, and End-to-End model
python3 -m src.models.benchmark_models

# Launch interactive draft simulator web app
streamlit run app/streamlit_app.py
```
