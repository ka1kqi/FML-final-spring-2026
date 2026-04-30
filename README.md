# League of Legends Draft Simulator: Season 16 Meta-Engine

A machine learning application that evaluates League of Legends draft compositions. It uses a custom **Stochastic Gradient Descent (SGD) Matrix Factorization** algorithm to generate individual champion performance embeddings, and feeds them into a **HistGradientBoostingRegressor** to evaluate 5v5 compositions in real-time.

---

## Two-Model Architecture

This project uses two complementary models:

1. **Champion2Vec + HistGradientBoostingRegressor** predicts a candidate champion's
   *performance score* (0–100) for the current draft state. The Champion2Vec embeddings
   are trained from scratch using a custom NumPy SGD matrix factorization over
   synergy and matchup matrices.

2. **Wide & Deep** predicts the *draft-level blue-side win probability* from the
   full 10-champion composition.

The web demo uses pretrained artifacts that ship in
`data/processed/draft_models/`. Retraining is optional and is not required to run
the demo. The displayed win probability comes from the Wide & Deep model when
available; if the `wide_deep.pt` artifact is missing, the app falls back to the
original score-derived display probability and labels it as a fallback in the
response (`prob_source: "score_heuristic_fallback"`).

---

## 🚀 Live Interactive Draft Simulator

This project features a fully interactive, beautifully designed live web application built with a **Flask API** and a **Vanilla JS/CSS Frontend**. 

As you lock in picks and bans, the UI pings the ML backend to instantly calculate:
* The individual performance scores of the locked-in champions.
* The real-time Win Probability shifting between the Blue and Red teams.
* The Top-k recommended champions for the current draft slot.

---

## 🧠 Machine Learning Architecture

Our pipeline focuses on **Individual Champion Performance Score** rather than a simple team win/loss classification.

### 1. Data Preprocessing
We extract KDA, Gold Per Minute (GPM), and Damage Per Minute (DPM) for 360,000+ individual player records from Season 16 matches. These stats are Z-score normalized to generate a `champ_score` (0-100 scale), isolating how well a player performed regardless of their team's ultimate outcome.

### 2. Asymmetric Matrix Factorization
We build two massive performance matrices:
* **Synergy Matrix:** Tracks Champion A's average individual score when Champion B is on their team (Asymmetric).
* **Matchup Matrix:** Tracks Champion A's average individual score when playing against Champion B.

We train a **Custom pure-numpy SGD Matrix Factorization** algorithm over 50 epochs to compress these matrices into a highly dense **64-Dimensional Performance Embedding** for all 170+ champions.

### 3. Gradient Boosting Prediction
To predict outcomes, we simulate chronological historical drafts. For any given pick, we calculate:
* The candidate's 64-D embedding.
* The mathematical average embedding of locked-in Allies & Enemies.
* The explicit Cosine Similarity (Synergy/Counter) between them.

A `HistGradientBoostingRegressor` is trained on these features to predict the individual `champ_score` of any champion inserted into a draft board.

---

## 📂 Repository Structure

```text
FML-final-spring-2026/
├── app/                  # Flask Web Server & Live Frontend UI
│   ├── server.py         # REST API endpoints (/api/recommend, /api/evaluate)
│   └── static/           # Vanilla CSS/JS and HTML
├── data/                 
│   ├── processed/        # Saved ML Models (.joblib, .npz, .json, .pt)
│   └── raw/              # Season 16 raw data & parsed compositions
├── src/                  
│   ├── collection/       # Riot API data fetchers
│   ├── data/             # Individual player stat extraction
│   ├── features/         # Feature engineering (Cosine Similarities)
│   ├── inference/        # Live draft recommendation engine
│   └── models/           # Custom SGD & Gradient Boosting logic
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## Running the demo

```bash
pip install -r requirements.txt
python app/server.py
```

The demo starts on `http://localhost:8080`. No Riot API key, raw data download,
or retraining is needed — pretrained artifacts under
`data/processed/draft_models/` are sufficient.

To retrain (optional):

```bash
pip install -r requirements-dev.txt
python -m src.models.train_draft_models   # Champion2Vec + HGBR
python -m src.models.train_wide_deep      # Wide & Deep (requires compositions_s16.csv)
```
