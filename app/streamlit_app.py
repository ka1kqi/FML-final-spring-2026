from pathlib import Path
import random
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.champion_vocab import load_champion_vocab, load_role_champion_options
from src.inference.e2e_infer import load_e2e_model, predict_blue_win_prob
from src.models.benchmark_models import run_benchmark

MODEL_PATH = PROJECT_ROOT / "data/processed/e2e_model.pth"


@st.cache_resource
def get_model_and_vocab():
    champ_to_id, _, champions = load_champion_vocab(PROJECT_ROOT)
    model = load_e2e_model(MODEL_PATH, vocab_size=len(champions))
    role_options = load_role_champion_options(
        PROJECT_ROOT,
        min_games_for_role=50,
        min_role_share=0.6,
    )
    return model, champ_to_id, champions, role_options


def validate_picks(blue, red):
    all_picks = blue + red
    if any(not pick for pick in all_picks):
        return "Please select all 10 champions."
    if len(set(all_picks)) != len(all_picks):
        return "Duplicate champion detected across teams."
    return ""


def validate_bans(blue_bans, red_bans):
    if len(blue_bans) > 5 or len(red_bans) > 5:
        return "Each team can ban at most 5 champions."
    all_bans = [b for b in blue_bans + red_bans if b]
    if len(set(all_bans)) != len(all_bans):
        return "Duplicate champion detected in bans."
    return ""


ROLES = ["Top", "Jungle", "Mid", "ADC", "Support"]
# Ranked draft order requested by user:
# B1 -> R1,R2 -> B2,B3 -> R3,R4 -> B4,B5 -> R5
DRAFT_ORDER = [
    ("Blue", 0),
    ("Red", 0),
    ("Red", 1),
    ("Blue", 1),
    ("Blue", 2),
    ("Red", 2),
    ("Red", 3),
    ("Blue", 3),
    ("Blue", 4),
    ("Red", 4),
]


def _initialize_draft_state():
    if "blue_bans" not in st.session_state:
        st.session_state.blue_bans = []
    if "red_bans" not in st.session_state:
        st.session_state.red_bans = []
    if "blue_picks" not in st.session_state:
        st.session_state.blue_picks = [None] * 5
    if "red_picks" not in st.session_state:
        st.session_state.red_picks = [None] * 5
    if "blue_roles" not in st.session_state:
        st.session_state.blue_roles = random.sample(ROLES, len(ROLES))
    if "red_roles" not in st.session_state:
        st.session_state.red_roles = random.sample(ROLES, len(ROLES))


def _used_champions(blue_picks, red_picks, blue_bans=None, red_bans=None):
    blue_bans = blue_bans or []
    red_bans = red_bans or []
    return {c for c in blue_picks + red_picks + blue_bans + red_bans if c is not None}


def _first_unused(options, used):
    for champ in options:
        if champ not in used:
            return champ
    return options[0] if options else None


def _complete_with_defaults(blue_picks, red_picks, role_options,
                            blue_roles=None, red_roles=None,
                            blue_bans=None, red_bans=None):
    blue_full = list(blue_picks)
    red_full = list(red_picks)
    used = _used_champions(blue_full, red_full, blue_bans, red_bans)

    blue_roles = blue_roles or ROLES
    red_roles = red_roles or ROLES

    # Fill each empty slot using its pre-assigned role
    for i in range(5):
        if blue_full[i] is None:
            pick = _first_unused(role_options.get(blue_roles[i], []), used)
            blue_full[i] = pick
            if pick:
                used.add(pick)

    for i in range(5):
        if red_full[i] is None:
            pick = _first_unused(role_options.get(red_roles[i], []), used)
            red_full[i] = pick
            if pick:
                used.add(pick)

    return blue_full, red_full


def _suggest_top5(
    model,
    champ_to_id,
    role_options,
    blue_picks,
    red_picks,
    blue_bans,
    red_bans,
    blue_roles,
    red_roles,
    side,
    slot_idx,
    selected_role,
):
    candidate_pool = role_options.get(selected_role, [])
    used = _used_champions(blue_picks, red_picks, blue_bans, red_bans)

    scored = []
    for champ in candidate_pool:
        if champ in used:
            continue

        blue_trial = list(blue_picks)
        red_trial = list(red_picks)
        blue_roles_trial = list(blue_roles)
        red_roles_trial = list(red_roles)

        if side == "Blue":
            blue_trial[slot_idx] = champ
            blue_roles_trial[slot_idx] = selected_role
        else:
            red_trial[slot_idx] = champ
            red_roles_trial[slot_idx] = selected_role

        blue_full, red_full = _complete_with_defaults(
            blue_trial, red_trial, role_options,
            blue_roles_trial, red_roles_trial,
            blue_bans, red_bans,
        )
        blue_prob = predict_blue_win_prob(model, blue_full, red_full, champ_to_id)
        side_prob = blue_prob if side == "Blue" else (1.0 - blue_prob)
        scored.append((champ, side_prob))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:5]


def main():
    st.set_page_config(page_title="LoL Draft Simulator", layout="wide")
    st.title("LoL Interactive Draft Simulator")
    st.caption("Predict blue-side win probability from champion draft.")

    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()

    model, champ_to_id, champion_options, role_options = get_model_and_vocab()

    tab_sim, tab_metrics = st.tabs(["Draft Simulator", "Model Quality"])

    with tab_sim:
        st.subheader("Ranked Pick Phase Simulator")
        _initialize_draft_state()
        blue_picks = st.session_state.blue_picks
        red_picks = st.session_state.red_picks

        current_pick_step = sum(p is not None for p in blue_picks + red_picks)

        top_col, reset_col = st.columns([3, 1])
        top_col.caption("Ban phase has no order constraint. Pick order: B1 → R1,R2 → B2,B3 → R3,R4 → B4,B5 → R5. Roles are randomly assigned — swap them below.")
        if reset_col.button("Reset Draft"):
            st.session_state.blue_bans = []
            st.session_state.red_bans = []
            st.session_state.blue_picks = [None] * 5
            st.session_state.red_picks = [None] * 5
            st.session_state.blue_roles = random.sample(ROLES, len(ROLES))
            st.session_state.red_roles = random.sample(ROLES, len(ROLES))
            st.rerun()

        st.markdown("### Ban Phase (No Order)")
        ban_input_left, ban_input_right = st.columns(2)
        with ban_input_left:
            blue_bans = st.multiselect(
                "Blue bans (max 5)",
                champion_options,
                default=st.session_state.blue_bans,
                max_selections=5,
                key="blue_bans",
            )
        with ban_input_right:
            red_options = [c for c in champion_options if c not in set(blue_bans)]
            red_bans = st.multiselect(
                "Red bans (max 5)",
                red_options,
                default=[c for c in st.session_state.red_bans if c not in set(blue_bans)],
                max_selections=5,
                key="red_bans",
            )

        ban_error = validate_bans(blue_bans, red_bans)
        if ban_error:
            st.error(ban_error)
            return

        ban_left, ban_right = st.columns(2)
        with ban_left:
            st.markdown("### Blue Bans")
            for b in (blue_bans or ["---"]):
                st.write(f"- {b or '---'}")
        with ban_right:
            st.markdown("### Red Bans")
            for b in (red_bans or ["---"]):
                st.write(f"- {b or '---'}")

        board_left, board_right = st.columns(2)
        with board_left:
            st.markdown("### Blue Team")
            for i in range(5):
                role = st.session_state.blue_roles[i]
                champ = blue_picks[i]
                st.write(f"- {role}: {champ or '---'}")
        with board_right:
            st.markdown("### Red Team")
            for i in range(5):
                role = st.session_state.red_roles[i]
                champ = red_picks[i]
                st.write(f"- {role}: {champ or '---'}")

        # --- Swap Roles UI ---
        st.markdown("### Swap Roles")
        swap_left, swap_right = st.columns(2)
        with swap_left:
            blue_roles_list = st.session_state.blue_roles
            swap_a = st.selectbox("Blue slot A", range(5),
                                  format_func=lambda i: f"{blue_roles_list[i]}: {blue_picks[i] or '---'}",
                                  key="blue_swap_a")
            swap_b = st.selectbox("Blue slot B", range(5),
                                  format_func=lambda i: f"{blue_roles_list[i]}: {blue_picks[i] or '---'}",
                                  index=1, key="blue_swap_b")
            if st.button("Swap Blue Roles", key="swap_blue"):
                st.session_state.blue_roles[swap_a], st.session_state.blue_roles[swap_b] = (
                    st.session_state.blue_roles[swap_b], st.session_state.blue_roles[swap_a]
                )
                st.rerun()
        with swap_right:
            red_roles_list = st.session_state.red_roles
            swap_c = st.selectbox("Red slot A", range(5),
                                  format_func=lambda i: f"{red_roles_list[i]}: {red_picks[i] or '---'}",
                                  key="red_swap_a")
            swap_d = st.selectbox("Red slot B", range(5),
                                  format_func=lambda i: f"{red_roles_list[i]}: {red_picks[i] or '---'}",
                                  index=1, key="red_swap_b")
            if st.button("Swap Red Roles", key="swap_red"):
                st.session_state.red_roles[swap_c], st.session_state.red_roles[swap_d] = (
                    st.session_state.red_roles[swap_d], st.session_state.red_roles[swap_c]
                )
                st.rerun()

        if current_pick_step < len(DRAFT_ORDER):
            side, slot_idx = DRAFT_ORDER[current_pick_step]
            current_role = (
                st.session_state.blue_roles[slot_idx]
                if side == "Blue"
                else st.session_state.red_roles[slot_idx]
            )
            st.markdown(f"### Current Turn: **{side}** picks **{current_role}**")

            suggestions = _suggest_top5(
                model=model,
                champ_to_id=champ_to_id,
                role_options=role_options,
                blue_picks=blue_picks,
                red_picks=red_picks,
                blue_bans=blue_bans,
                red_bans=red_bans,
                blue_roles=st.session_state.blue_roles,
                red_roles=st.session_state.red_roles,
                side=side,
                slot_idx=slot_idx,
                selected_role=current_role,
            )
            st.markdown("#### Top 5 Suggested Picks")
            if suggestions:
                for champ, prob in suggestions:
                    st.write(f"- {champ}: {prob * 100:.2f}% expected win chance for {side}")
            else:
                st.info("No available candidates for this role.")

            available_options = [
                c for c in role_options.get(current_role, champion_options)
                if c not in _used_champions(blue_picks, red_picks, blue_bans, red_bans)
            ]
            pick_choice = st.selectbox(
                f"Lock in {side} {current_role}",
                available_options,
                key=f"pick_step_{current_pick_step}",
            )
            if st.button("Lock Pick", type="primary"):
                if side == "Blue":
                    blue_picks[slot_idx] = pick_choice
                else:
                    red_picks[slot_idx] = pick_choice
                st.session_state.blue_picks = blue_picks
                st.session_state.red_picks = red_picks
                st.rerun()
        else:
            st.success("Draft complete.")
            ban_error = validate_bans(blue_bans, red_bans)
            if ban_error:
                st.error(ban_error)
                return
            error_msg = validate_picks(blue_picks, red_picks)
            if error_msg:
                st.error(error_msg)
                return
            blue_prob = predict_blue_win_prob(model, blue_picks, red_picks, champ_to_id)
            red_prob = 1.0 - blue_prob
            metric_left, metric_right = st.columns(2)
            metric_left.metric("Blue Win Probability", f"{blue_prob * 100:.2f}%")
            metric_right.metric("Red Win Probability", f"{red_prob * 100:.2f}%")

    with tab_metrics:
        st.subheader("Model Benchmark (same test split)")
        st.caption("Compares Logistic Regression, Random Forest, and End-to-End Transformer.")
        if st.button("Run Benchmark", key="run_benchmark_btn"):
            with st.spinner("Training/evaluating models... this may take a while."):
                results = run_benchmark()

            rows = []
            for model_name, metrics in results.items():
                rows.append(
                    {
                        "model": model_name,
                        "accuracy": metrics["accuracy"],
                        "f1": metrics["f1"],
                        "auc": metrics["auc"],
                    }
                )

            benchmark_df = pd.DataFrame(rows)
            st.dataframe(benchmark_df, use_container_width=True)
            st.bar_chart(benchmark_df.set_index("model"))


if __name__ == "__main__":
    main()
