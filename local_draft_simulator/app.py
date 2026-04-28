"""
LoL Pick-Ban Draft Simulator (Streamlit)
========================================

Simulates the official LoL competitive draft order (Tournament Draft) and
on every turn that belongs to the user's side surfaces the model's top-k
recommendation. The opponent's actions are entered manually (or auto-picked
from the model's view of "best for them").

Run::

    streamlit run local_draft_simulator/app.py -- --artifacts-dir artifacts

This whole directory is git-ignored.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import lol_draft_pipeline as P  # noqa: E402


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_argv() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--artifacts-dir", default="artifacts")
    args, _ = parser.parse_known_args()
    return args


CLI = _parse_argv()


# --------------------------------------------------------------------------- #
# Draft phase definition - official Tournament Draft order
# --------------------------------------------------------------------------- #
# Phase 1: 3 bans each, alternating (B1, R1, B2, R2, B3, R3)
# Phase 1 picks: B1, R1, R2, B2, B3, R3      (snake)
# Phase 2: 2 bans each, red first (R4, B4, R5, B5)
# Phase 2 picks: R4, B4, B5, R5              (snake)

DRAFT_PHASES: List[Tuple[str, str, int]] = [
    ("ban",  "blue", 0), ("ban",  "red",  0),
    ("ban",  "blue", 1), ("ban",  "red",  1),
    ("ban",  "blue", 2), ("ban",  "red",  2),
    ("pick", "blue", 0), ("pick", "red",  0), ("pick", "red",  1),
    ("pick", "blue", 1), ("pick", "blue", 2), ("pick", "red",  2),
    ("ban",  "red",  3), ("ban",  "blue", 3),
    ("ban",  "red",  4), ("ban",  "blue", 4),
    ("pick", "red",  3), ("pick", "blue", 3),
    ("pick", "blue", 4), ("pick", "red",  4),
]

ROLES: Tuple[str, ...] = ("top", "jungle", "mid", "adc", "support")
ROLE_EMOJI = {"top": "⚔️", "jungle": "🌲", "mid": "✨", "adc": "🏹", "support": "🛡️"}


# --------------------------------------------------------------------------- #
# Streamlit page
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="LoL Pick-Ban Simulator",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .pick-card { background: rgba(127, 127, 127, 0.06); padding: 0.55rem; border-radius: 8px; min-height: 78px; }
      .pick-card-active { background: rgba(255, 215, 0, 0.20); border: 1px solid rgba(255, 215, 0, 0.6); }
      .ban-card { background: rgba(207, 34, 46, 0.10); padding: 0.45rem; border-radius: 8px; min-height: 50px; }
      .ban-card-active { background: rgba(207, 34, 46, 0.25); border: 1px solid rgba(207, 34, 46, 0.7); }
      .ban-card-empty { color: #666; }
      .ban-strike { text-decoration: line-through; color: #aaa; }
      .role-tag { color: #888; font-size: 0.75rem; }
      .step-banner { background: linear-gradient(90deg, rgba(50,150,250,0.10), rgba(50,150,250,0.0));
                     padding: 0.6rem 1rem; border-radius: 8px; }
      .your-turn { background: rgba(46, 204, 113, 0.15); border: 1px solid rgba(46,204,113,0.5); }
      .their-turn { background: rgba(207, 34, 46, 0.10); border: 1px solid rgba(207,34,46,0.4); }
    </style>
    """,
    unsafe_allow_html=True,
)


st.title("🎯  LoL Pick-Ban Draft Simulator")
st.caption("Tournament-style draft. Your side gets live model recommendations on every turn.")


# --------------------------------------------------------------------------- #
# Sidebar - configuration
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.subheader("Configuration")
    artifacts_dir = st.text_input("artifacts dir", value=CLI.artifacts_dir)
    art = Path(artifacts_dir)

    if not (art / "champion_to_idx.json").exists():
        st.error(f"Vocab not found at {art/'champion_to_idx.json'}.\nTrain the pipeline first:\n\n"
                 f"`python lol_draft_pipeline.py train --fast-dev-run`")
        st.stop()

    vocab: Dict[str, int] = json.loads((art / "champion_to_idx.json").read_text())
    champ_list = sorted([c for c in vocab.keys() if c != "<UNK>"])

    available_models: List[str] = []
    for m, fn in [
        ("stacker", "stacker.pkl"),
        ("hybrid", "lightgbm_with_embeddings.pkl"),
        ("baseline", "lightgbm_baseline.pkl"),
        ("teamcompnet", "teamcompnet.pt"),
        ("wide_deep", "wide_deep.pt"),
    ]:
        if (art / fn).exists():
            available_models.append(m)
    if not available_models:
        st.error("No trained models found. Train at least one stage first.")
        st.stop()

    model_name = st.selectbox("Model", available_models, index=0)
    top_k = st.number_input("Top-K", min_value=3, max_value=15, value=5)

    search_mode = st.radio(
        "Search strategy", ["greedy top-k", "beam search", "MCTS"],
        horizontal=False, index=0,
    )
    if search_mode == "beam search":
        beam_width = st.slider("beam width", 2, 10, 5)
        beam_depth = st.slider("beam depth", 1, 4, 2)
    if search_mode == "MCTS":
        mcts_sims = st.slider("MCTS simulations", 16, 256, 64)
        mcts_cpuct = st.slider("c_puct", 0.5, 3.0, 1.5, 0.1)

    st.divider()
    my_side = st.radio("My side", ["blue", "red"], horizontal=True)

    st.subheader("Role assignment")
    st.caption("Set which role each pick slot in your team will fill. The recommender uses this when scoring candidates for that slot.")
    if "blue_slot_role" not in st.session_state:
        st.session_state.blue_slot_role = {i: ROLES[i] for i in range(5)}
    if "red_slot_role" not in st.session_state:
        st.session_state.red_slot_role = {i: ROLES[i] for i in range(5)}

    for i in range(5):
        st.session_state.blue_slot_role[i] = st.selectbox(
            f"🔵 Blue pick #{i+1}", ROLES,
            index=ROLES.index(st.session_state.blue_slot_role[i]),
            key=f"brole_{i}",
        )
    for i in range(5):
        st.session_state.red_slot_role[i] = st.selectbox(
            f"🔴 Red pick #{i+1}", ROLES,
            index=ROLES.index(st.session_state.red_slot_role[i]),
            key=f"rrole_{i}",
        )

    st.divider()
    if st.button("🔁 Reset draft", use_container_width=True):
        for k in ("draft", "history"):
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()


# --------------------------------------------------------------------------- #
# Session state - the draft itself
# --------------------------------------------------------------------------- #

if "draft" not in st.session_state:
    st.session_state.draft = {
        "blue_bans": [None] * 5,
        "red_bans":  [None] * 5,
        "blue_picks": {},  # role -> champion
        "red_picks":  {},
        "step": 0,
    }
if "history" not in st.session_state:
    st.session_state.history: List[Dict[str, object]] = []


# --------------------------------------------------------------------------- #
# Score-fn loaders (cached so flipping turns doesn't re-load LightGBM)
# --------------------------------------------------------------------------- #


@st.cache_resource(show_spinner=False)
def _load_score_fns(model_name: str, artifacts_dir: str):
    cfg = P.PipelineConfig(artifacts_dir=str(artifacts_dir))
    handcrafted = P.load_pickle(Path(artifacts_dir) / "handcrafted_stats.pkl")
    score_fn, batch_fn = P._build_score_fns(model_name, cfg, vocab, handcrafted)
    return cfg, handcrafted, score_fn, batch_fn


cfg, handcrafted, score_fn, batch_fn = _load_score_fns(model_name, str(artifacts_dir))
recommender = P.Recommender(
    score_fn=score_fn,
    batch_score_fn=batch_fn,
    vocab=vocab,
    handcrafted=handcrafted,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def build_state() -> "P.DraftState":
    """Materialise the current DraftState from session-state."""
    s = P.DraftState()
    s.blue_picks = dict(st.session_state.draft["blue_picks"])
    s.red_picks = dict(st.session_state.draft["red_picks"])
    s.bans = [b for b in st.session_state.draft["blue_bans"] if b] + \
             [b for b in st.session_state.draft["red_bans"] if b]
    return s


def used_champions() -> set:
    state = build_state()
    return state.used_champions()


def apply_action(action_type: str, side: str, slot: int,
                 champion: str, role: Optional[str]) -> None:
    """Mutate session state to apply one ban or pick and advance the step."""
    d = st.session_state.draft
    if action_type == "ban":
        d[f"{side}_bans"][slot] = champion
    else:
        if role is None:
            role = ROLES[slot]
        d[f"{side}_picks"][role] = champion
    d["step"] += 1
    st.session_state.history.append({
        "step": d["step"], "type": action_type, "side": side,
        "slot": slot + 1, "champion": champion, "role": role,
    })


def role_for_slot(side: str, slot: int) -> str:
    return st.session_state[f"{side}_slot_role"][slot]


# --------------------------------------------------------------------------- #
# Visual layout
# --------------------------------------------------------------------------- #


def _ban_cell(b: Optional[str], active: bool) -> str:
    cls = "ban-card ban-card-active" if active else "ban-card"
    if b:
        body = f"<span class='ban-strike'>{b}</span>"
    else:
        body = "<span class='ban-card-empty'>—</span>"
    return f"<div class='{cls}'>{body}</div>"


def _pick_cell(role: str, champ: Optional[str], active: bool) -> str:
    cls = "pick-card pick-card-active" if active else "pick-card"
    body = f"<b>{champ}</b>" if champ else "<span class='ban-card-empty'>(empty)</span>"
    return (
        f"<div class='{cls}'>"
        f"<span class='role-tag'>{ROLE_EMOJI.get(role, '')} {role.upper()}</span><br>{body}"
        f"</div>"
    )


def _team_panel(side: str, label: str, current_action) -> None:
    bans = st.session_state.draft[f"{side}_bans"]
    picks = st.session_state.draft[f"{side}_picks"]
    role_map = st.session_state[f"{side}_slot_role"]
    st.markdown(f"### {label}")

    # Bans row
    cols = st.columns(5)
    for i, c in enumerate(cols):
        with c:
            active = (current_action and current_action[0] == "ban"
                      and current_action[1] == side and current_action[2] == i)
            st.markdown(_ban_cell(bans[i], active), unsafe_allow_html=True)

    # Picks row (in role order)
    cols = st.columns(5)
    for i, c in enumerate(cols):
        with c:
            r = role_map.get(i, ROLES[i])
            champ = picks.get(r)
            active = (current_action and current_action[0] == "pick"
                      and current_action[1] == side and current_action[2] == i)
            st.markdown(_pick_cell(r, champ, active), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Win-prob gauge
# --------------------------------------------------------------------------- #


def _winprob_gauge(my_wp: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=my_wp * 100,
        number={"suffix": "%", "valueformat": ".1f"},
        title={"text": f"Your ({my_side}) win prob"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#2ecc71" if my_wp >= 0.5 else "#e74c3c"},
            "steps": [
                {"range": [0, 45], "color": "rgba(231, 76, 60, 0.20)"},
                {"range": [45, 55], "color": "rgba(255, 255, 255, 0.05)"},
                {"range": [55, 100], "color": "rgba(46, 204, 113, 0.20)"},
            ],
            "threshold": {"line": {"color": "white", "width": 2}, "value": 50},
        },
    ))
    fig.update_layout(height=220, margin=dict(t=40, b=10, l=20, r=20))
    return fig


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #

step = st.session_state.draft["step"]
total = len(DRAFT_PHASES)
current_action = DRAFT_PHASES[step] if step < total else None

# Top: progress bar
st.progress(step / total, text=f"Draft step {step} / {total}")

# Live winprob
state = build_state()
cur_blue_wp = float(score_fn(state)) if (state.blue_picks or state.red_picks) else 0.5
my_wp = cur_blue_wp if my_side == "blue" else (1.0 - cur_blue_wp)

c_left, c_right = st.columns([2, 1])
with c_left:
    _team_panel("blue", "🔵  Blue Side", current_action)
    st.markdown(" ")
    _team_panel("red", "🔴  Red Side", current_action)
with c_right:
    st.plotly_chart(_winprob_gauge(my_wp), use_container_width=True)
    st.caption(f"Model: `{model_name}` · Strategy: `{search_mode}`")
    if my_wp >= 0.55:
        st.success("Composition is favourable.")
    elif my_wp <= 0.45:
        st.error("Composition is losing - look for high-leverage picks.")
    else:
        st.info("Composition roughly balanced.")

st.divider()


# --------------------------------------------------------------------------- #
# Current step UI
# --------------------------------------------------------------------------- #


if current_action is None:
    st.success("🎉 Draft complete!")
    st.metric("Final blue win prob", f"{cur_blue_wp:.4f}")
    st.metric(f"Your final ({my_side}) win prob", f"{my_wp:.4f}")

    log_df = pd.DataFrame(st.session_state.history)
    if not log_df.empty:
        st.markdown("#### Draft log")
        st.dataframe(log_df, use_container_width=True)
    st.stop()


action_type, side, slot = current_action
is_my_turn = (side == my_side)
banner_cls = "step-banner your-turn" if is_my_turn else "step-banner their-turn"
who = "YOU" if is_my_turn else "OPPONENT"
verb = "BAN" if action_type == "ban" else "PICK"
slot_role = role_for_slot(side, slot) if action_type == "pick" else None
slot_label = f"#{slot+1}"
if action_type == "pick":
    slot_label += f"  ({ROLE_EMOJI[slot_role]} {slot_role})"

st.markdown(
    f"<div class='{banner_cls}'><b>Step {step+1}/{total}</b> &nbsp; · &nbsp;"
    f"{who} &nbsp; <b>{side.upper()}</b> {verb} {slot_label}</div>",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Recommendations (only when picking; ban suggestions are simpler)
# --------------------------------------------------------------------------- #


def _run_recommendations(state, side, role) -> List[Dict]:
    if search_mode == "MCTS":
        return recommender.mcts(
            state, side, role, n_simulations=mcts_sims,
            c_puct=mcts_cpuct, k=int(top_k),
        )
    if search_mode == "beam search":
        return recommender.beam_search(
            state, side, role, beam_width=beam_width,
            depth=beam_depth, k=int(top_k),
        )
    return recommender.top_k(state, side, role, k=int(top_k))


def _run_ban_suggestions(state, opponent_side: str) -> List[Dict]:
    """Cheap heuristic: ban whichever champion would help the opponent the most.

    For each candidate, fill the *opponent's* most-empty slot and rank by how
    much it would lower OUR win probability. We don't know which role the
    opponent would pick the banned champion in, so we approximate by checking
    the next role they're likely to fill.
    """
    # Find next pick action for opponent (or any empty slot if not found)
    opp_empty = []
    role_map = st.session_state[f"{opponent_side}_slot_role"]
    picks = state.picks_for(opponent_side)
    for i in range(5):
        r = role_map[i]
        if r not in picks:
            opp_empty.append(r)
    if not opp_empty:
        return []
    role = opp_empty[0]
    cands = recommender.candidates(state)
    scores = recommender._score_candidates(state, opponent_side, role, cands)
    me_signed = scores if my_side == "blue" else (1.0 - scores)
    # We want to ban the candidate that maximises opponent's win, i.e. minimises ours.
    if my_side == opponent_side:
        return []
    order = np.argsort(me_signed)  # ascending = worst for us
    out = []
    for idx in order[: int(top_k)]:
        c = cands[idx]
        out.append({
            "champion": c,
            "win_prob_if_pickedby_opp": float(scores[idx]),
            "our_winprob_after": float(me_signed[idx]),
            "delta_for_us": float(me_signed[idx] - my_wp),
        })
    return out


col_l, col_r = st.columns([2, 1])

with col_l:
    if is_my_turn and action_type == "pick":
        st.markdown(f"#### Recommended picks for **{slot_role}**")
        with st.spinner("Scoring candidates ..."):
            results = _run_recommendations(state, side, slot_role)
        if not results:
            st.warning("No legal candidates.")
        else:
            for i, r in enumerate(results, 1):
                cs = st.columns([0.5, 2.5, 1.2, 1.0, 1.0, 3.0, 1.6])
                cs[0].markdown(f"**{i}**")
                cs[1].markdown(f"### {r['champion']}")
                cs[2].metric("WinProb", f"{r['win_prob']:.4f}")
                cs[3].metric("Δ", f"{r['delta']:+.4f}")
                cs[4].metric("Synergy", f"{r['synergy']:+.4f}")
                cs[5].caption(r["notes"])
                if cs[6].button("✅ Pick", key=f"rec_{step}_{r['champion']}", use_container_width=True):
                    apply_action("pick", side, slot, r["champion"], slot_role)
                    st.rerun()

    elif is_my_turn and action_type == "ban":
        st.markdown("#### Suggested bans (champions opponent would value most)")
        suggestions = _run_ban_suggestions(state, opponent_side="red" if my_side == "blue" else "blue")
        if not suggestions:
            st.info("Heuristic ban list unavailable - just choose manually.")
        else:
            df = pd.DataFrame(suggestions)
            df.insert(0, "rank", range(1, len(df) + 1))
            st.dataframe(df, use_container_width=True)
            top_choice = suggestions[0]["champion"]
            if st.button(f"🚫 Ban {top_choice} (top heuristic)", key=f"rec_ban_{step}"):
                apply_action("ban", side, slot, top_choice, None)
                st.rerun()

    else:
        st.info(f"Opponent's turn. Enter what they {action_type} below.")

with col_r:
    st.markdown("#### Manual entry")
    st.caption("Use this for opponent moves, or to override a recommendation.")
    used = used_champions()
    legal = [c for c in champ_list if c not in used]
    pick_label = "champion to ban" if action_type == "ban" else "champion picked"
    chosen = st.selectbox(pick_label, ["(choose)"] + legal, key=f"manual_{step}")
    if chosen != "(choose)":
        if st.button("Confirm", type="primary", key=f"confirm_{step}", use_container_width=True):
            role = slot_role if action_type == "pick" else None
            apply_action(action_type, side, slot, chosen, role)
            st.rerun()


# --------------------------------------------------------------------------- #
# History expander
# --------------------------------------------------------------------------- #

with st.expander("📜 Draft history", expanded=False):
    if not st.session_state.history:
        st.caption("Nothing picked yet.")
    else:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True)
