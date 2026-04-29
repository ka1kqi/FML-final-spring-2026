# LoL Pick-Ban Draft Simulator

Local Streamlit app that walks the **official competitive draft order**
(6 bans → 6 picks → 4 bans → 4 picks) and surfaces the model's
recommendation on every turn that belongs to your side.

> This entire directory (`local_draft_simulator/`) is git-ignored.

---

## Why a separate page

The main `local_dashboard/` is for *training inspection* (loss curves,
embeddings, calibration, leakage audit). This page is the **operational
flow**: you actually run a draft against an opponent, step by step,
and the model tells you what to pick.

---

## Setup

```bash
# pipeline + dashboard deps already installed
pip install -r requirements.txt
pip install -r local_dashboard/requirements_dashboard.txt

# train at least one stage (so artifacts exist)
python lol_draft_pipeline.py train --fast-dev-run
```

## Launch

```bash
streamlit run local_draft_simulator/app.py -- --artifacts-dir artifacts
```

The `--` is required so Streamlit forwards the flag to the script.

---

## Draft order (Tournament Draft)

```
Phase 1 bans:   B1 R1 B2 R2 B3 R3
Phase 1 picks:  B1 R1 R2 B2 B3 R3
Phase 2 bans:   R4 B4 R5 B5
Phase 2 picks:  R4 B4 B5 R5
```

Total = 10 bans + 10 picks = 20 actions.

---

## Workflow

1. **Sidebar**:
   * Choose your side (blue / red), the model (`stacker` is best),
     top-k, and a search strategy:
     * `greedy top-k` (default — fastest)
     * `beam search` (1-step lookahead with opponent minimax)
     * `MCTS` (AlphaZero-style PUCT, slowest but smartest with the policy head).
   * Assign which role each pick slot in your team will fill. Roles are
     *static per slot* — change them before draft starts.

2. **Main area**:
   * Two team panels showing all 5 bans + 5 picks per side. The currently
     active slot is highlighted gold.
   * A win-prob gauge updates every action.
   * Below, the active step gets a banner ("YOU / OPPONENT — BLUE BAN #1").

3. **Active step**:
   * **Your pick** → top-k candidates with win prob, delta, synergy,
     counter, and a one-click "✅ Pick" button.
   * **Your ban** → heuristic ban list: champions whose pick by the
     opponent would lower your win prob the most.
   * **Opponent's turn** → use the right-hand "Manual entry" selector
     to record what they did.

4. **End**:
   * Final draft summary, full history table, final win prob.

5. **Reset**:
   * Sidebar "🔁 Reset draft" wipes everything.

---

## Model details

* The simulator imports `lol_draft_pipeline.py` directly and reuses
  `Recommender.top_k / beam_search / mcts`.
* Score functions are cached via `@st.cache_resource`, so re-runs
  within the same session don't re-load LightGBM / PyTorch.
* If you have the Set Transformer + policy head trained, MCTS gets a
  real prior; otherwise it falls back to a value-shaped softmax prior.

---

## Troubleshooting

* **"No trained models found"** — train first:
  `python lol_draft_pipeline.py train --fast-dev-run`
* **Champion name not in vocab** — expected if you trained on a small
  subset. The simulator only lets you pick from the saved vocab.
* **Recommendations always identical** — the model is undertrained.
  Rerun training with more epochs / more data
  (e.g. `--max-rows 50000 --epochs 20`).
* **Strange win-prob jumps** — expected on partial drafts; UNK slots
  carry low signal until enough champions are in.

---

## Layout

```
local_draft_simulator/
├── app.py                  # Streamlit page (single file, ~400 lines)
└── README_SIMULATOR.md     # this file
```
