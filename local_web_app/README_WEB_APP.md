# Local Web App — LoL Draft AI

Polished single-page Flask + vanilla-JS draft simulator that calls
straight into the trained `lol_draft_pipeline` artifacts. Mirrors the
visual polish of a teammate's earlier app, but with **calibrated win
probabilities** (not a linear `0.5 + (score-50)*0.01` hack), real
test-set provenance shown in the meta bar, and live access to all of
our recommender algorithms (greedy / beam search / **MCTS**).

> The whole `local_web_app/` source is committed to the repo. Caches
> and Python venvs inside it are git-ignored.

---

## Quick start

```bash
# 1. Install Flask (one-time; pipeline + dashboard deps already needed)
pip install -r local_web_app/requirements_web_app.txt

# 2. Make sure at least one model is trained
python lol_draft_pipeline.py train --fast-dev-run

# 3. Launch
python local_web_app/server.py
# → open http://127.0.0.1:8090
```

Default port `8090` (Streamlit dashboard uses 8501, simulator usually
8501/8502, so they don't clash).

---

## What's on the page

* **Top meta bar**: real provenance — best test-AUC, recall@5, dataset
  size, split type. No fabricated numbers. Pulled live from
  `artifacts/runs/<latest>/metrics_summary.json`.
* **Model dropdown**: pick any trained model (`wide_deep`, `hybrid`,
  `baseline`, `teamcompnet`, `stacker`). Recommendations recompute on
  change.
* **Search dropdown**: `greedy` (default, fastest), `beam` (1-step
  minimax lookahead), `MCTS` (AlphaZero-style PUCT, opponent-aware).
* **Ban phase bar**: 5 slots per side, active slot glows gold.
* **Team panels**: 5 role-fixed pick slots per side with live win-prob.
* **Champion grid**: 170+ champion portraits (DDragon CDN), filterable
  by role and searchable by name.
* **AI Picks**: top-5 candidates with calibrated win prob, delta over
  current state, and human-readable notes ("good synergy with current
  allies", "warning: weak matchup vs enemy picks", etc.).
* **Lock / Skip Ban / Reset Draft** buttons.
* **Completion overlay**: portraits + final calibrated win-prob bar.

---

## Why we built this in addition to the dashboard / simulator

| App | Audience | Strength |
|---|---|---|
| `local_dashboard/` | engineer | Inspect training: loss / calibration / embeddings / leakage audit |
| `local_draft_simulator/` | engineer | Same draft flow, tighter Streamlit feel |
| `local_web_app/` | non-technical viewer | Real product feel, single page, no scrolling, looks like champion-select |

The web app is the **demo-day** UI. The dashboard is for the **paper**.

---

## Architecture

```
local_web_app/
├── server.py                     ~190 LoC. Flask + 4 endpoints. Imports
│                                 lol_draft_pipeline directly; caches
│                                 score functions per model; reads the
│                                 latest run's metrics_summary.json.
├── static/
│   ├── index.html                ~120 LoC. Layout: meta bar / ban phase /
│   │                             team panels / champ grid / AI picks /
│   │                             completion overlay.
│   ├── style.css                 ~520 LoC. LoL-themed dark palette with
│   │                             gold accent. Responsive grid for
│   │                             champion portraits.
│   └── app.js                    ~330 LoC. Owns the 20-step draft order,
│                                 selection state, calls /api/recommend
│                                 and /api/evaluate to keep the win-prob
│                                 gauges and AI Picks live.
├── requirements_web_app.txt      flask
└── README_WEB_APP.md             this file
```

### Backend endpoints

| Method | Path | Returns |
|---|---|---|
| `GET`  | `/api/meta`        | dataset / model metadata for the meta bar |
| `GET`  | `/api/champions`   | sorted champion list with DDragon image URL + roles played in our data |
| `POST` | `/api/recommend`   | top-K calibrated recommendations + current `blue_win_prob` |
| `POST` | `/api/evaluate`    | `blue_win_prob` for a (possibly partial) draft |

### Calibrated probability vs raw score

Unlike the teammate's app, which converts a regressor's continuous score
to a "probability" via `0.5 + (score-50)*0.01`, we use the actual
`P(blue_win)` output of the trained classifier, **after** isotonic
calibration on the validation split. So a 60% number really means
"the model expects blue to win ~60% of the time at this draft state".

---

## Troubleshooting

* **"Artifacts not found"** — train first: `python lol_draft_pipeline.py train --fast-dev-run`.
* **Champion portraits don't load** — DDragon CDN occasionally rate-limits;
  the grid still works, just shows blank tiles. Refresh in 30 s.
* **MCTS feels slow on click** — set `mcts_simulations` to a smaller
  number in `app.js` (default 64). 32 is fine on partial drafts.
* **Port 8090 already in use** — change the port in `server.py`'s
  `app.run(...)` call.
* **Champion not in vocab warning in console** — that champion appeared
  too rarely in training data; it's still pickable but the model treats
  it as `<UNK>`.

---

## Layout

```
local_web_app/
├── README_WEB_APP.md
├── requirements_web_app.txt
├── server.py
└── static/
    ├── app.js
    ├── index.html
    └── style.css
```
