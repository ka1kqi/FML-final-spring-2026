# League of Legends Draft Simulator: Season 16 Meta-Engine

A cutting-edge machine learning application that predicts League of Legends match outcomes strictly from the Draft Phase. It uses a custom **Stochastic Gradient Descent (SGD) Matrix Factorization** algorithm to generate granular individual champion performance embeddings, and feeds them into a **HistGradientBoostingRegressor** to evaluate 5v5 compositions in real-time.

---

## 🚀 Live Interactive Draft Simulator

This project features a fully interactive, beautifully designed live web application built with a **Flask API** and a **Vanilla JS/CSS Frontend**. 

As you lock in picks and bans, the UI pings the ML backend to instantly calculate:
* The exact individual performance scores of the locked-in champions.
* The real-time Win Probability shifting between the Blue and Red teams.
* The Top-k mathematically optimal champion recommendations for the current draft slot.

---

## 🧠 Machine Learning Architecture

Our pipeline abandons the traditional "Team Win/Loss" classification approach. Instead, we use an **Individual Champion Performance Score**.

### 1. Data Preprocessing
We extract KDA, Gold Per Minute (GPM), and Damage Per Minute (DPM) for 360,000+ individual player records from Season 16 matches. These stats are Z-score normalized to generate a `champ_score` (0-100 scale), isolating exactly how well a player performed regardless of their team's ultimate outcome.

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

A `HistGradientBoostingRegressor` is trained on these features to accurately predict the individual `champ_score` of any champion inserted into a draft board.

---

## 📂 Repository Structure

```text
FML-final-spring-2026/
├── app/                  # Flask Web Server & Live Frontend UI
│   ├── server.py         # REST API endpoints (+ /similar, /api/similar)
│   └── static/           # Vanilla CSS/JS and HTML
├── data/                 
│   ├── processed/        # Saved ML Models (.joblib, .npz, .json)
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

## ⚙️ Quick Start

**1. Install Dependencies**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Launch the Live Web App**
```bash
python app/server.py
```
Open `http://127.0.0.1:8080` in your browser to experience the real-time AI drafting engine!

Additional pages:
- **Champion Similarity (Pivot Pool)**: `http://127.0.0.1:8080/similar`

**3. Retrain the Models from Scratch**
*(Optional: If you pull new Riot API data)*
```bash
python -m src.models.train_draft_models
```
