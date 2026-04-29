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
import numpy as np
import pandas as pd

app = Flask(__name__, static_folder="static", static_url_path="/static")

DRAFT_MODELS_DIR = PROJECT_ROOT / "data" / "processed" / "draft_models"

# ---------- load resources once at startup ----------

champ_to_id, id_to_champ, champion_list = load_champion_vocab(PROJECT_ROOT)
role_options = load_role_champion_options(
    PROJECT_ROOT, min_games_for_role=50, min_role_share=0.6,
)
draft_model, embed_dict, champ_scores = load_draft_resources(DRAFT_MODELS_DIR)

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

for champ, group in comp_df.groupby("champion_name"):
    # Role Dist
    role_counts = group["position"].value_counts(normalize=True)
    champ_role_stats[champ] = {role: round(pct * 100, 1) for role, pct in role_counts.items()}
    
    # Win Rate
    champ_win_rates[champ] = round(group["win"].mean() * 100, 2)
    
    # Lane Pool (10% threshold)
    for role, pct in role_counts.items():
        if pct >= 0.10:
            lane_pools[role].append(champ)

print("Stats and Lane Pools calculated.")

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

    recommendations = []
    for name, score in recs:
        # Direct 1:1 map: score 52.5 -> 0.525 win prob
        win_prob = 0.50 + (score - 50.0) * 0.01
        win_prob = max(0.0, min(1.0, win_prob))
        
        recommendations.append({
            "champion": name,
            "score": round(score, 1),
            "win_prob": round(win_prob, 4)
        })

    # Re-sort recommendations by amplified score just to be safe
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

    # Evaluate the composition by taking the average predicted score
    # of the model evaluating each of the 5 Blue champions in context.
    blue_scores = []
    for i, candidate in enumerate(blue_picks):
        allies = [p for j, p in enumerate(blue_picks) if j != i]
        enemies = red_picks
        features = build_candidate_features(
            candidate, allies, enemies, embed_dict, champ_scores
        )
        score = draft_model.predict(features.reshape(1, -1))[0]
        blue_scores.append(float(score))

    avg_blue_score = sum(blue_scores) / len(blue_scores)
    
    # Since the score centers around 50 for an average match, red's score is roughly mirrored
    avg_red_score = 100.0 - avg_blue_score

    blue_win_prob = max(0.0, min(1.0, 0.50 + (avg_blue_score - 50.0) * 0.01))
    red_win_prob = 1.0 - blue_win_prob

    return jsonify({
        "blue_score": round(avg_blue_score, 1),
        "blue_win_prob": blue_win_prob,
        "red_score": round(avg_red_score, 1),
        "red_win_prob": red_win_prob
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
        
        syn = float(np.dot(u_syn_a, v_syn_b) + np.dot(u_syn_b, v_syn_a))
        a_counters_b = float(np.dot(u_match_a, v_match_b) - np.dot(u_match_b, v_match_a))
        
        synergy_data.append({"champion": champ_b, "score": syn})
        counter_data.append({"champion": champ_b, "score": a_counters_b})
        
    # 1. Group Synergies by Role (Top 3 for each other role)
    roles_order = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
    role_synergies = {}
    for r in roles_order:
        # Filter synergy_data to champs in this role pool
        pool = lane_pools.get(r, [])
        r_synergies = [s for s in synergy_data if s["champion"] in pool]
        r_synergies.sort(key=lambda x: x["score"], reverse=True)
        role_synergies[r] = r_synergies[:3]
    
    response["role_synergies"] = role_synergies
    
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
        
        response["comparison"] = {
            "champion": compare,
            "synergy_score": round(syn_val, 4),
            "matchup_score": round(match_val, 4),
            "status": f"{champ} vs {compare}"
        }
        
    return jsonify(response)


if __name__ == "__main__":
    print(f"Champions loaded: {len(champion_list)}")
    print(f"Draft model: {type(draft_model).__name__}")
    print(f"Embeddings: {len(embed_dict)} champions")
    app.run(debug=True, port=8080)
