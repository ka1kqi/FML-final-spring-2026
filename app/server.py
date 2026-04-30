"""
Flask API server for the LoL Draft Recommender.

Endpoints:
    GET  /              — Serve the draft screen HTML
    GET  /api/champions — Return champion list with role data
    POST /api/recommend — Get top-5 recommendations for current draft step
"""

import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.champion_vocab import load_champion_vocab, load_role_champion_options
from src.inference.draft_recommender import (
    load_draft_resources,
    recommend_at_step,
    get_current_step,
    DRAFT_ORDER,
)
from src.features.synergy_features import build_candidate_features
from src.models.match_classifier import build_match_features, load_match_model
import numpy as np
import pandas as pd
from typing import List, Set, Dict

app = Flask(__name__, static_folder="static", static_url_path="/static")

DRAFT_MODELS_DIR = PROJECT_ROOT / "data" / "processed" / "draft_models"

# ---------- load resources once at startup ----------

champ_to_id, id_to_champ, champion_list = load_champion_vocab(PROJECT_ROOT)
role_options = load_role_champion_options(
    PROJECT_ROOT, min_games_for_role=50, min_role_share=0.6,
)
draft_model, embed_dict, champ_scores = load_draft_resources(DRAFT_MODELS_DIR)

match_model_path = DRAFT_MODELS_DIR / "match_model.joblib"
match_model = load_match_model(match_model_path) if match_model_path.exists() else None
if match_model is None:
    print("WARNING: match_model.joblib not found — /api/evaluate will fall back to per-pick averaging")

# Build a champion -> list-of-roles mapping for the frontend
champ_roles: dict[str, list[str]] = {}
for role, champs in role_options.items():
    for c in champs:
        champ_roles.setdefault(c, []).append(role)

# Load full composition data to calculate exact stats
print("Calculating exact stats for Analysis Page...")
comp_df = pd.read_csv(PROJECT_ROOT / "data/raw/compositions_s16.csv", usecols=["champion_name", "position", "win"])

champ_role_stats: dict[str, dict[str, float]] = {}
champ_win_rates: dict[str, float] = {}
lane_pools: dict[str, list[str]] = {r: [] for r in ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]}

# Per-role play counts -> pick_rate + log-z popularity bonus used to re-rank recs.
role_pick_counts: dict[str, dict[str, int]] = {r: {} for r in lane_pools}
for (champ, role), grp in comp_df.groupby(["champion_name", "position"]):
    if role in role_pick_counts:
        role_pick_counts[role][champ] = int(len(grp))

role_pick_rate: dict[str, dict[str, float]] = {r: {} for r in lane_pools}
role_pop_bonus: dict[str, dict[str, float]] = {r: {} for r in lane_pools}
for r, counts in role_pick_counts.items():
    if not counts:
        continue
    total = sum(counts.values())
    logs = {c: np.log(n) for c, n in counts.items()}
    mean = float(np.mean(list(logs.values())))
    std = float(np.std(list(logs.values()))) or 1e-6
    for c, n in counts.items():
        role_pick_rate[r][c] = n / total
        role_pop_bonus[r][c] = (logs[c] - mean) / std

for champ, group in comp_df.groupby("champion_name"):
    role_counts = group["position"].value_counts(normalize=True)
    champ_role_stats[champ] = {role: round(pct * 100, 1) for role, pct in role_counts.items()}
    champ_win_rates[champ] = round(group["win"].mean() * 100, 2)
    for role, pct in role_counts.items():
        if pct >= 0.10:
            lane_pools[role].append(champ)

print("Stats, Lane Pools, and popularity bonuses calculated.")

# Build champ_can_play map for role-coverage validation
# Maps champion -> set of roles they've been played in (≥20 games)
champ_can_play: Dict[str, Set[str]] = {}
role_game_counts: Dict[str, Dict[str, int]] = {r: {} for r in ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]}

for (champ, role), grp in comp_df.groupby(["champion_name", "position"]):
    count = len(grp)
    if role in role_game_counts:
        role_game_counts[role][champ] = count
        if count >= 20:  # threshold for role validity
            champ_can_play.setdefault(champ, set()).add(role)

# Fill in any champions without explicit role data (make them available in all roles for graceful degradation)
for champ in champion_list:
    if champ not in champ_can_play:
        champ_can_play[champ] = set()  # champion has no role coverage

print(f"Role-coverage validation ready ({len([c for c in champ_can_play.values() if c])} champs with role data)")

# App role label -> data role enum used by all stats above.
APP_TO_DATA_ROLE = {"Top": "TOP", "Jungle": "JUNGLE", "Mid": "MIDDLE",
                    "ADC": "BOTTOM", "Support": "UTILITY"}

# Popularity blend in logit space. Held conservative because the
# per-pick model's raw prob is concentrated near 0.5 — too strong a
# weight here makes popularity dominate ranking. +2-sigma → +0.3 logit
# → ~+7% pp lift from a 50% baseline.
POPULARITY_WEIGHT_LOGIT = 0.15


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return float(np.log(p / (1 - p)))


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _can_assign_roles(team_champs: List[str], roles: List[str],
                      champ_can_play: Dict[str, Set[str]]) -> bool:
    """
    Bipartite matching: can we assign each champion to a distinct role?

    Uses permutation-based search for exact solution. If any champion
    cannot play any role, returns False immediately.
    """
    if len(team_champs) != len(roles):
        return False

    # Quick check: can each champion play at least one of the roles?
    for champ in team_champs:
        available_roles = champ_can_play.get(champ, set())
        if not available_roles:
            return False  # Champion has no role data

    # Try to find a valid assignment using permutations
    from itertools import permutations
    for role_perm in permutations(roles):
        valid = True
        for champ, role in zip(team_champs, role_perm):
            available_roles = champ_can_play.get(champ, set())
            if role not in available_roles:
                valid = False
                break
        if valid:
            return True

    return False

# DDragon name overrides (dataset name -> DDragon key)
DDRAGON_KEY_MAP = {
    "FiddleSticks": "Fiddlesticks",
    "Chogath": "Chogath",
    "Velkoz": "Velkoz",
    "Khazix": "Khazix",
    "Belveth": "Belveth",
    "KSante": "KSante",
    "Nunu": "Nunu",
    "Renata": "Renata",
    "Wukong": "MonkeyKing",
    "Leblanc": "Leblanc",
}

DDRAGON_VERSION = "16.8.1"

# ---------- routes ----------

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/champions")
def api_champions():
    """Return all champions with their roles and DDragon image key."""
    data = []
    for name in champion_list:
        ddragon_key = DDRAGON_KEY_MAP.get(name, name)
        img_url = (
            f"https://ddragon.leagueoflegends.com/cdn/"
            f"{DDRAGON_VERSION}/img/champion/{ddragon_key}.png"
        )
        data.append({
            "name": name,
            "id": ddragon_key,
            "img": img_url,
            "roles": champ_roles.get(name, []),
        })
    return jsonify(data)


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    """Return top-5 recommendations for the current draft step."""
    body = request.get_json(force=True)
    blue_picks = body.get("blue_picks", [None] * 5)
    red_picks = body.get("red_picks", [None] * 5)
    blue_bans = body.get("blue_bans", [])
    red_bans = body.get("red_bans", [])
    step = body.get("step", None)
    role_filter = body.get("role", None)

    # Auto-detect step if not provided
    if step is None:
        step = get_current_step(blue_picks, red_picks)

    if step >= len(DRAFT_ORDER):
        return jsonify({"side": None, "slot": None, "recommendations": []})

    side, slot = DRAFT_ORDER[step]

    # Build candidate pool from role filter or all champions
    candidate_pool = None
    if role_filter and role_filter in role_options:
        candidate_pool = role_options[role_filter]

    banned = [b for b in (blue_bans + red_bans) if b]

    recs = recommend_at_step(
        step=step,
        blue_picks=blue_picks,
        red_picks=red_picks,
        model=draft_model,
        embed_dict=embed_dict,
        champ_scores=champ_scores,
        candidate_pool=candidate_pool,
        banned=banned,
        top_k=50,
    )

    data_role = APP_TO_DATA_ROLE.get(role_filter)

    recommendations = []
    for name, win_prob_raw in recs:
        pop_bonus = role_pop_bonus.get(data_role, {}).get(name, 0.0) if data_role else 0.0
        pick_rate = role_pick_rate.get(data_role, {}).get(name, 0.0) if data_role else 0.0

        # Blend in logit space: keeps probabilities calibrated.
        win_prob = _sigmoid(_logit(win_prob_raw) + POPULARITY_WEIGHT_LOGIT * pop_bonus)

        recommendations.append({
            "champion": name,
            "score": round(win_prob * 100, 1),
            "score_raw": round(win_prob_raw * 100, 1),
            "pick_rate": round(pick_rate * 100, 1),
            "win_prob": round(win_prob, 4),
        })

    recommendations.sort(key=lambda x: x["score"], reverse=True)

    return jsonify({
        "step": step,
        "side": side,
        "slot": slot,
        "recommendations": recommendations,
    })


@app.route("/api/evaluate", methods=["POST"])
def api_evaluate():
    """Return the final win probability for the completed draft."""
    body = request.get_json(force=True)
    blue_picks = body.get("blue_picks", [])
    red_picks = body.get("red_picks", [])

    if len(blue_picks) != 5 or len(red_picks) != 5 or any(p is None for p in blue_picks + red_picks):
        return jsonify({
            "blue_score": 50.0,
            "blue_win_prob": 0.5,
            "red_score": 50.0,
            "red_win_prob": 0.5
        })

    # Check role coverage for both teams
    ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
    blue_roles_valid = _can_assign_roles(blue_picks, ROLES, champ_can_play)
    red_roles_valid = _can_assign_roles(red_picks, ROLES, champ_can_play)

    if match_model is not None:
        # Per-match aggregate classifier — proper calibrated win probability.
        feats = build_match_features(blue_picks, red_picks, embed_dict, champ_scores)
        blue_logit = _logit(float(match_model.predict_proba(feats.reshape(1, -1))[0, 1]))
    else:
        # Fallback: average leave-one-out P(win) on the per-pick model.
        probs = []
        for i, candidate in enumerate(blue_picks):
            allies = [p for j, p in enumerate(blue_picks) if j != i]
            features = build_candidate_features(candidate, allies, red_picks, embed_dict, champ_scores)
            probs.append(float(draft_model.predict_proba(features.reshape(1, -1))[0, 1]))
        blue_logit = _logit(sum(probs) / len(probs))

    # Apply role-coverage penalty: -1.0 logit ≈ -26pp from 50%
    ROLE_COVERAGE_PENALTY = 1.0

    if not blue_roles_valid:
        blue_logit -= ROLE_COVERAGE_PENALTY
    if not red_roles_valid:
        blue_logit += ROLE_COVERAGE_PENALTY

    blue_win_prob = _sigmoid(blue_logit)
    red_win_prob = 1.0 - blue_win_prob

    return jsonify({
        "blue_score": round(blue_win_prob * 100, 1),
        "blue_win_prob": round(blue_win_prob, 4),
        "red_score": round(red_win_prob * 100, 1),
        "red_win_prob": round(red_win_prob, 4),
    })


@app.route("/api/analysis", methods=["GET"])
def api_analysis():
    champ = request.args.get("champ")
    compare = request.args.get("compare")
    selected_role = request.args.get("role") # e.g. MIDDLE
    
    if not champ or champ not in embed_dict:
        return jsonify({"error": "Invalid or missing champion"}), 400
        
    a = embed_dict[champ]
    u_syn_a, v_syn_a = a[0:16], a[16:32]
    u_match_a, v_match_a = a[32:48], a[48:64]
    
    response = {
        "champion": champ,
        "win_rate": champ_win_rates.get(champ, 50.0),
        "roles": champ_role_stats.get(champ, {})
    }
    
    # Calculate scores against all other champions
    synergy_data = []
    counter_data = []
    
    for champ_b, b in embed_dict.items():
        if champ == champ_b: continue
        u_syn_b, v_syn_b = b[0:16], b[16:32]
        u_match_b, v_match_b = b[32:48], b[48:64]
        
        # PATH A: Direct Score Prediction (Delta from 50.0)
        syn_delta = float(np.dot(u_syn_a, v_syn_b) + np.dot(u_syn_b, v_syn_a))
        a_counters_b_delta = float(np.dot(u_match_a, v_match_b) - np.dot(u_match_b, v_match_a))
        
        # Synergy is 50 + delta, but for the "Score" list we display the delta
        synergy_data.append({"champion": champ_b, "score": syn_delta})
        counter_data.append({"champion": champ_b, "score": a_counters_b_delta})
        
    # 1. Group Synergies by Role (Top 3 per role). Skip the query
    # champion's own primary role — same-role pairs can never be allies
    # in a real draft, and the MF prediction for them is contaminated
    # by role-archetype similarity rather than actual synergy.
    roles_order = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
    query_roles = champ_role_stats.get(champ, {})
    query_primary_role = max(query_roles.items(), key=lambda x: x[1])[0] if query_roles else None

    role_synergies = {}
    for r in roles_order:
        if r == query_primary_role:
            role_synergies[r] = []
            continue
        pool = lane_pools.get(r, [])
        r_synergies = [s for s in synergy_data if s["champion"] in pool]
        r_synergies.sort(key=lambda x: x["score"], reverse=True)
        role_synergies[r] = r_synergies[:3]

    response["role_synergies"] = role_synergies
    response["primary_role"] = query_primary_role
    
    # 2. Filter Counters by Lane (if role selected)
    if selected_role and selected_role in lane_pools:
        lane_pool = lane_pools[selected_role]
        filtered_counters = [c for c in counter_data if c["champion"] in lane_pool]
    else:
        filtered_counters = counter_data

    filtered_counters.sort(key=lambda x: x["score"], reverse=True)
    
    response["counters"] = filtered_counters[:5]
    response["countered_by"] = [{"champion": c["champion"], "score": -c["score"]} for c in filtered_counters[-5:]][::-1]
    
    if compare and compare in embed_dict:
        syn_val = next(s["score"] for s in synergy_data if s["champion"] == compare)
        match_val = next(s["score"] for s in counter_data if s["champion"] == compare)
        
        # PATH A: We provide the absolute predicted synergy score (50 + delta)
        # and the raw counter delta.
        response["comparison"] = {
            "champion": compare,
            "synergy_score": round(50.0 + syn_val, 2), 
            "matchup_score": round(match_val, 2),
            "status": f"{champ} vs {compare}"
        }
        
    return jsonify(response)


if __name__ == "__main__":
    print(f"Champions loaded: {len(champion_list)}")
    print(f"Draft model: {type(draft_model).__name__}")
    print(f"Embeddings: {len(embed_dict)} champions")
    app.run(debug=True, port=8080)
