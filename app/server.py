"""
Flask API server for the LoL Draft Recommender.

Endpoints:
    GET  /              — Serve the draft screen HTML
    GET  /similar       — Champion embedding similarity (pivot pool) page
    GET  /api/champions — Return champion list with role data
    GET  /api/similar   — Cosine-nearest champions in Champion2Vec space
    POST /api/recommend — Get top-5 recommendations for current draft step
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple

from flask import Flask, jsonify, request, send_from_directory

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.champion_vocab import load_champion_vocab, load_role_champion_options
from src.inference.draft_recommender import (
    load_draft_resources,
    get_current_step,
    DRAFT_ORDER,
)
from src.inference.wide_deep_adapter import WideDeepDraftAdapter
from src.inference.draft_recommender import recommend_hybrid
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
draft_model, embed_dict, champ_scores, embed_biases = load_draft_resources(DRAFT_MODELS_DIR)
HAS_BIASES = bool(embed_biases)

# Role-aware embedding (each unit is "Champion|ROLE"). Optional artifact —
# gracefully handle older training runs that didn't ship it.
ROLE_EMBED_PATH = DRAFT_MODELS_DIR / "champion_role_2vec.npz"
role_embed_dict: dict = {}
role_embed_biases: dict = {}
if ROLE_EMBED_PATH.exists():
    _r = np.load(str(ROLE_EMBED_PATH), allow_pickle=True)
    _r_vocab = _r["vocab"].tolist()
    role_embed_dict = {k: _r["weights"][i] for i, k in enumerate(_r_vocab)}
    if "b_syn_u" in _r.files:
        role_embed_biases = {
            "mu_syn": float(_r["mu_syn"]),
            "mu_match": float(_r["mu_match"]),
            "b_syn_u": {k: float(_r["b_syn_u"][i]) for i, k in enumerate(_r_vocab)},
            "b_syn_v": {k: float(_r["b_syn_v"][i]) for i, k in enumerate(_r_vocab)},
            "b_match_u": {k: float(_r["b_match_u"][i]) for i, k in enumerate(_r_vocab)},
            "b_match_v": {k: float(_r["b_match_v"][i]) for i, k in enumerate(_r_vocab)},
        }
    print(f"Loaded role-aware embedding: {len(role_embed_dict)} (champion, role) units")
else:
    print("WARNING: champion_role_2vec.npz not found — /api/role_analysis disabled")

match_model_path = DRAFT_MODELS_DIR / "match_model.joblib"
match_model = load_match_model(match_model_path) if match_model_path.exists() else None
if match_model is None:
    print("WARNING: match_model.joblib not found — /api/evaluate will fall back to per-pick averaging")

wide_deep_adapter = WideDeepDraftAdapter(model_dir=DRAFT_MODELS_DIR)
print(f"Wide & Deep adapter available: {wide_deep_adapter.available}")


def _adapter_status() -> tuple[str, str, list[str]]:
    """Return (prob_source, model_version, warnings) for the current adapter state.

    prob_source is "wide_deep" when the adapter is available, else
    "score_heuristic_fallback". warnings is a list ready to merge into the
    response payload.
    """
    if wide_deep_adapter.available:
        return "wide_deep", wide_deep_adapter.model_version, []
    return (
        "score_heuristic_fallback",
        wide_deep_adapter.model_version,
        ["Wide & Deep artifact missing — using score-derived heuristic for win_prob."],
    )


# Build a champion -> list-of-roles mapping for the frontend
champ_roles: dict[str, list[str]] = {}
for role, champs in role_options.items():
    for c in champs:
        champ_roles.setdefault(c, []).append(role)

# Load full composition data to calculate exact stats
print("Calculating exact stats for Analysis Page...")
comp_csv = PROJECT_ROOT / "data/raw/compositions_s16.csv"
champ_role_stats: dict[str, dict[str, float]] = {}
champ_win_rates: dict[str, float] = {}
lane_pools: dict[str, list[str]] = {r: [] for r in ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]}
# Pre-compute role-popularity, role-coverage, and analysis stats from the
# composition CSV. If the CSV is missing the server still boots — every
# downstream feature degrades gracefully (recommend ranks without popularity,
# evaluate skips role-coverage penalty, analysis page returns empty stats).
role_pick_counts: dict[str, dict[str, int]] = {r: {} for r in lane_pools}
role_pick_rate: dict[str, dict[str, float]] = {r: {} for r in lane_pools}
role_pop_bonus: dict[str, dict[str, float]] = {r: {} for r in lane_pools}
champ_roles_loose: dict[str, list[str]] = {}
champ_can_play: Dict[str, Set[str]] = {}

if comp_csv.exists():
    comp_df = pd.read_csv(comp_csv, usecols=["champion_name", "position", "win"])

    # Per-role play counts -> pick_rate + log-z popularity bonus used to re-rank recs.
    for (champ, role), grp in comp_df.groupby(["champion_name", "position"]):
        if role in role_pick_counts:
            role_pick_counts[role][champ] = int(len(grp))

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

    # Looser threshold for the analysis page: any role the champion plays >=10%
    # of the time. Akali plays Mid 73% and Top 27% — both should be analyzable.
    for champ, role_dict in champ_role_stats.items():
        sorted_roles = sorted(role_dict.items(), key=lambda x: x[1], reverse=True)
        playable = [r for r, pct in sorted_roles if pct >= 10.0]
        if playable:
            champ_roles_loose[champ] = playable

    # Build champ_can_play map for role-coverage validation: champion -> set of
    # roles they've been played in (≥20 games)
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
            champ_can_play[champ] = set()

    print(f"Stats, lane pools, popularity bonuses, role-coverage ready "
          f"({len([c for c in champ_can_play.values() if c])} champs with role data)")
else:
    print(f"Warning: {comp_csv.name} not found; analysis/role stats empty (graceful fallback).")

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

def _cosine_neighbors(
    anchor: str,
    k: int,
    role_filter: Optional[str] = None,
) -> Optional[List[dict]]:
    """Top-k cosine neighbors in full 64-D Champion2Vec space; optional role pool."""
    if anchor not in embed_dict:
        return None
    q = embed_dict[anchor]
    qn = float(np.linalg.norm(q))
    if qn < 1e-12:
        return []

    pairs: List[Tuple[str, float]] = []
    for name, vec in embed_dict.items():
        if name == anchor:
            continue
        vn = float(np.linalg.norm(vec))
        if vn < 1e-12:
            continue
        sim = float(np.dot(q, vec) / (qn * vn))
        pairs.append((name, sim))

    pairs.sort(key=lambda x: x[1], reverse=True)

    allowed: Optional[set] = None
    if role_filter and role_filter.lower() not in ("all", ""):
        rf = role_filter.strip()
        key = None
        if rf in role_options:
            key = rf
        else:
            for rk in role_options:
                if rk.lower() == rf.lower():
                    key = rk
                    break
        if key is not None:
            allowed = set(role_options[key])

    out: List[dict] = []
    for name, sim in pairs:
        if allowed is not None and name not in allowed:
            continue
        ddragon = DDRAGON_KEY_MAP.get(name, name)
        img = (
            f"https://ddragon.leagueoflegends.com/cdn/"
            f"{DDRAGON_VERSION}/img/champion/{ddragon}.png"
        )
        out.append({
            "name": name,
            "similarity": sim,
            "historical_wr": champ_win_rates.get(name),
            "img": img,
        })
        if len(out) >= k:
            break
    return out


# ---------- routes ----------

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/similar")
def similar_page():
    return send_from_directory("static", "similar.html")


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
        # `roles` (strict, >=60%) drives recommender candidate pools.
        # `roles_loose` (>=10%, data-format) drives the analysis page so
        # secondary lanes like Akali Top can be inspected.
        data.append({
            "name": name,
            "id": ddragon_key,
            "img": img_url,
            "roles": champ_roles.get(name, []),
            "roles_loose": champ_roles_loose.get(name, []),
        })
    return jsonify(data)


@app.route("/api/similar")
def api_similar():
    """Cosine-similar champions using the same Champion2Vec embeddings as draft."""
    champion = (request.args.get("champion") or "").strip()
    if not champion:
        return jsonify({"error": "missing champion"}), 400

    k_req = request.args.get("k", type=int)
    k = int(k_req) if k_req is not None else 8
    k = max(1, min(k, 24))

    role = (request.args.get("role") or "all").strip()
    role_arg: Optional[str] = None
    if role.lower() not in ("all", ""):
        role_arg = role

    neighbors = _cosine_neighbors(champion, k, role_arg)
    if neighbors is None:
        return jsonify({"error": f"unknown champion {champion!r}"}), 400

    return jsonify({
        "anchor": champion,
        "role_filter": role if role else "all",
        "neighbors": neighbors,
        "count": len(neighbors),
    })


@app.route("/api/similar_pair")
def api_similar_pair():
    """Direct cosine similarity between two champions in Champion2Vec space."""
    a = (request.args.get("a") or "").strip()
    b = (request.args.get("b") or "").strip()
    if not a or not b:
        return jsonify({"error": "missing a or b"}), 400
    if a not in embed_dict:
        return jsonify({"error": f"unknown champion {a!r}"}), 400
    if b not in embed_dict:
        return jsonify({"error": f"unknown champion {b!r}"}), 400

    va = embed_dict[a]
    vb = embed_dict[b]
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na < 1e-12 or nb < 1e-12:
        return jsonify({"a": a, "b": b, "cosine": 0.0})

    cos = float(np.dot(va, vb) / (na * nb))
    return jsonify({"a": a, "b": b, "cosine": cos})


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    """Top-K recommendations with hybrid (perf + W&D) ranking."""
    body = request.get_json(force=True)
    blue_picks = body.get("blue_picks", [None] * 5)
    red_picks = body.get("red_picks", [None] * 5)
    blue_bans = body.get("blue_bans", [])
    red_bans = body.get("red_bans", [])
    step = body.get("step", None)
    role_filter = body.get("role", None)
    top_k = int(body.get("top_k", 5))

    # Toggle-driven ranking: "wide_deep" -> hybrid (alpha=0.6), "heuristic" -> pure perf_score
    rank_source = (body.get("prob_source") or "wide_deep").lower()
    alpha = 1.0 if rank_source == "heuristic" else 0.6

    if step is None:
        step = get_current_step(blue_picks, red_picks)

    if step >= len(DRAFT_ORDER):
        prob_source, model_version, warnings = _adapter_status()
        return jsonify({
            "step": step,
            "side": None, "slot": None, "recommendations": [],
            "wide_deep_available": wide_deep_adapter.available,
            "prob_source": prob_source,
            "model_version": model_version,
            "warnings": warnings,
        })

    side, slot = DRAFT_ORDER[step]

    candidate_pool = None
    if role_filter and role_filter in role_options:
        candidate_pool = role_options[role_filter]

    banned = [b for b in (blue_bans + red_bans) if b]

    # When user picks "heuristic" mode we skip the W&D adapter entirely so
    # final_rank_score == norm_perf and the top-K is the true top-K by perf score.
    adapter_for_call = wide_deep_adapter if rank_source != "heuristic" else None

    recs = recommend_hybrid(
        step=step,
        blue_picks=blue_picks,
        red_picks=red_picks,
        model=draft_model,
        embed_dict=embed_dict,
        champ_scores=champ_scores,
        candidate_pool=candidate_pool,
        banned=banned,
        top_k=top_k,
        wide_deep_adapter=adapter_for_call,
        alpha=alpha,
        rerank_top_n=30,
    )

    prob_source, model_version, warnings = _adapter_status()

    # Post-process recommend_hybrid dicts: blend popularity bonus in logit space
    # (matches the original /api/recommend behaviour) and add a pick_rate field.
    data_role = APP_TO_DATA_ROLE.get(role_filter)
    for r in recs:
        pop_bonus = role_pop_bonus.get(data_role, {}).get(r["champion"], 0.0) if data_role else 0.0
        pick_rate = role_pick_rate.get(data_role, {}).get(r["champion"], 0.0) if data_role else 0.0
        # Blend the displayed win_prob in logit space; keep raw prob untouched.
        if pop_bonus and 0.0 < r["win_prob"] < 1.0:
            r["win_prob"] = round(_sigmoid(_logit(r["win_prob"]) + POPULARITY_WEIGHT_LOGIT * pop_bonus), 4)
        r["pick_rate"] = round(pick_rate * 100, 1)
    # Re-sort after popularity blend (some champions may swap order)
    recs.sort(key=lambda r: r["final_rank_score"], reverse=True)

    return jsonify({
        "step": step,
        "side": side,
        "slot": slot,
        "recommendations": recs,
        "wide_deep_available": wide_deep_adapter.available,
        "prob_source": prob_source,
        "model_version": model_version,
        "warnings": warnings,
    })


@app.route("/api/evaluate", methods=["POST"])
def api_evaluate():
    """Final composition win probability — uses W&D when available."""
    body = request.get_json(force=True)
    blue_picks = body.get("blue_picks", [])
    red_picks = body.get("red_picks", [])

    if (
        len(blue_picks) != 5
        or len(red_picks) != 5
        or any(p is None for p in blue_picks + red_picks)
    ):
        return jsonify({
            "blue_score": 50.0,
            "blue_win_prob": 0.5,
            "red_score": 50.0,
            "red_win_prob": 0.5,
            "prob_source": "score_heuristic_fallback",
            "model_version": wide_deep_adapter.model_version,
            "warnings": ["Incomplete draft — returning neutral win prob."],
        })

    # ---------- Compute ALL three probability sources for the toggle ----------
    ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
    blue_roles_valid = _can_assign_roles(blue_picks, ROLES, champ_can_play) if champ_can_play else True
    red_roles_valid = _can_assign_roles(red_picks, ROLES, champ_can_play) if champ_can_play else True
    ROLE_COVERAGE_PENALTY = 1.0  # -1.0 logit ≈ -26pp

    def _apply_coverage(logit):
        if not blue_roles_valid:
            logit -= ROLE_COVERAGE_PENALTY
        if not red_roles_valid:
            logit += ROLE_COVERAGE_PENALTY
        return logit

    # ---- 1. Per-pick model leave-one-out average (heuristic source) ----
    probs = []
    for i, candidate in enumerate(blue_picks):
        allies = [p for j, p in enumerate(blue_picks) if j != i]
        feats = build_candidate_features(candidate, allies, red_picks, embed_dict, champ_scores)
        probs.append(float(draft_model.predict_proba(feats.reshape(1, -1))[0, 1]))
    blue_prob_pick_avg = sum(probs) / len(probs)
    blue_logit_heuristic = _apply_coverage(_logit(blue_prob_pick_avg))
    blue_win_prob_heuristic = _sigmoid(blue_logit_heuristic)
    red_win_prob_heuristic = 1.0 - blue_win_prob_heuristic
    avg_blue_score = blue_prob_pick_avg * 100.0
    avg_red_score = 100.0 - avg_blue_score

    # ---- 2. Match-classifier (calibrated LR over composition deltas) ----
    if match_model is not None:
        match_feats = build_match_features(blue_picks, red_picks, embed_dict, champ_scores)
        blue_logit_mc = _apply_coverage(
            _logit(float(match_model.predict_proba(match_feats.reshape(1, -1))[0, 1]))
        )
        blue_win_prob_match = _sigmoid(blue_logit_mc)
        red_win_prob_match = 1.0 - blue_win_prob_match
    else:
        blue_win_prob_match = None
        red_win_prob_match = None

    # ---- 3. Wide & Deep neural net ----
    if wide_deep_adapter.available:
        wd_raw = wide_deep_adapter.predict_blue_win_prob(blue_picks, red_picks)
        blue_win_prob_wide_deep = _sigmoid(_apply_coverage(_logit(wd_raw)))
        red_win_prob_wide_deep = 1.0 - blue_win_prob_wide_deep
    else:
        blue_win_prob_wide_deep = None
        red_win_prob_wide_deep = None

    # ---- Pick the displayed win_prob based on body.prob_source toggle ----
    rank_source = (body.get("prob_source") or "").lower()
    if rank_source == "wide_deep" and blue_win_prob_wide_deep is not None:
        blue_win_prob, red_win_prob = blue_win_prob_wide_deep, red_win_prob_wide_deep
        active_source = "wide_deep"
    elif rank_source == "heuristic":
        blue_win_prob, red_win_prob = blue_win_prob_heuristic, red_win_prob_heuristic
        active_source = "heuristic"
    elif blue_win_prob_match is not None:
        blue_win_prob, red_win_prob = blue_win_prob_match, red_win_prob_match
        active_source = "match_classifier"
    elif blue_win_prob_wide_deep is not None:
        blue_win_prob, red_win_prob = blue_win_prob_wide_deep, red_win_prob_wide_deep
        active_source = "wide_deep"
    else:
        blue_win_prob, red_win_prob = blue_win_prob_heuristic, red_win_prob_heuristic
        active_source = "heuristic"

    _, model_version, base_warnings = _adapter_status()
    warnings = list(base_warnings)
    if not blue_roles_valid:
        warnings.append("Blue team has invalid role coverage — applying -1.0 logit penalty.")
    if not red_roles_valid:
        warnings.append("Red team has invalid role coverage — applying -1.0 logit penalty (favours blue).")

    return jsonify({
        "blue_score": round(avg_blue_score, 1),
        "blue_win_prob": round(blue_win_prob, 4),
        "red_score": round(avg_red_score, 1),
        "red_win_prob": round(red_win_prob, 4),
        "blue_win_prob_wide_deep": None if blue_win_prob_wide_deep is None else round(blue_win_prob_wide_deep, 4),
        "red_win_prob_wide_deep": None if red_win_prob_wide_deep is None else round(red_win_prob_wide_deep, 4),
        "blue_win_prob_heuristic": round(blue_win_prob_heuristic, 4),
        "red_win_prob_heuristic": round(red_win_prob_heuristic, 4),
        "blue_win_prob_match_classifier": None if blue_win_prob_match is None else round(blue_win_prob_match, 4),
        "red_win_prob_match_classifier": None if red_win_prob_match is None else round(red_win_prob_match, 4),
        "wide_deep_available": wide_deep_adapter.available,
        "match_classifier_available": match_model is not None,
        "prob_source": active_source,
        "model_version": model_version,
        "warnings": warnings,
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
    
    # Absolute MF prediction reconstructs the centered score deviation
    # for each pair using bias terms (when present in the artifact):
    #   pred(i, j) ~= mu + b_u[i] + b_v[j] + U[i] . V[j]
    # which represents (i's score when j is on relevant team) - 50.
    # We then average the two asymmetric directions and add 50 to land
    # on a 0-100 readable score where 50 = neutral.
    synergy_data = []
    counter_data = []

    if HAS_BIASES:
        bsu = embed_biases["b_syn_u"]; bsv = embed_biases["b_syn_v"]; mu_s = embed_biases["mu_syn"]
        bmu = embed_biases["b_match_u"]; bmv = embed_biases["b_match_v"]; mu_m = embed_biases["mu_match"]
    else:
        bsu = bsv = bmu = bmv = {}; mu_s = mu_m = 0.0

    for champ_b, b in embed_dict.items():
        if champ == champ_b: continue
        u_syn_b, v_syn_b = b[0:16], b[16:32]
        u_match_b, v_match_b = b[32:48], b[48:64]

        # Synergy: A's predicted score boost when B is ally, and reverse.
        a_with_b = mu_s + bsu.get(champ, 0.0) + bsv.get(champ_b, 0.0) + float(np.dot(u_syn_a, v_syn_b))
        b_with_a = mu_s + bsu.get(champ_b, 0.0) + bsv.get(champ, 0.0) + float(np.dot(u_syn_b, v_syn_a))
        synergy_avg = (a_with_b + b_with_a) / 2.0
        syn_score = max(0.0, min(100.0, 50.0 + synergy_avg))

        # Matchup: A's predicted score against B (positive = A favored).
        a_vs_b = mu_m + bmu.get(champ, 0.0) + bmv.get(champ_b, 0.0) + float(np.dot(u_match_a, v_match_b))
        ctr_score = max(0.0, min(100.0, 50.0 + a_vs_b))

        synergy_data.append({"champion": champ_b, "score": round(syn_score, 1)})
        counter_data.append({"champion": champ_b, "score": round(ctr_score, 1)})
        
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
    # For "countered_by", show "how strongly champion B counters A" by
    # mirroring around 50: low matchup score → high counter strength.
    response["countered_by"] = [
        {"champion": c["champion"], "score": round(100.0 - c["score"], 1)}
        for c in filtered_counters[-5:]
    ][::-1]

    if compare and compare in embed_dict:
        syn_val = next(s["score"] for s in synergy_data if s["champion"] == compare)
        match_val = next(s["score"] for s in counter_data if s["champion"] == compare)
        response["comparison"] = {
            "champion": compare,
            "synergy_score": syn_val,    # 0-100, 50 = neutral chemistry
            "matchup_score": match_val,  # 0-100, 50 = even matchup, >50 = champ favored
            "status": f"{champ} vs {compare}",
        }
        
    return jsonify(response)


@app.route("/api/role_analysis", methods=["GET"])
def api_role_analysis():
    """
    Role-aware analysis. Each unit is (champion, role) so Akali-MID
    has a different profile than Akali-TOP.

    Query: /api/role_analysis?champ=Akali&role=MIDDLE
    Optional: &compare=Ahri  (specific head-to-head; assumes same role
              unless &compare_role=... is supplied).

    Returns:
        - same_lane_matchups: matchups vs every other (champ, role=role)
            split into best (this champ favored) and worst (countered_by)
        - same_lane_synergies: top allies in each *other* role
        - comparison (if compare given)
    """
    if not role_embed_dict:
        return jsonify({"error": "role-aware embedding not available"}), 503

    champ = (request.args.get("champ") or "").strip()
    role = (request.args.get("role") or "").strip().upper()
    compare = (request.args.get("compare") or "").strip()
    compare_role = (request.args.get("compare_role") or role).strip().upper()

    if not champ or not role:
        return jsonify({"error": "champ and role are required"}), 400

    key = f"{champ}|{role}"
    if key not in role_embed_dict:
        return jsonify({
            "error": f"no role-aware data for {champ} in {role}",
            "hint": "champion may not have ≥10% pick share or ≥20 games in that role",
        }), 404

    a_vec = role_embed_dict[key]
    u_syn_a, v_syn_a = a_vec[0:16], a_vec[16:32]
    u_match_a, v_match_a = a_vec[32:48], a_vec[48:64]

    # Bias dicts (default 0 if absent)
    bsu = role_embed_biases.get("b_syn_u", {}) if role_embed_biases else {}
    bsv = role_embed_biases.get("b_syn_v", {}) if role_embed_biases else {}
    bmu = role_embed_biases.get("b_match_u", {}) if role_embed_biases else {}
    bmv = role_embed_biases.get("b_match_v", {}) if role_embed_biases else {}
    mu_s = role_embed_biases.get("mu_syn", 0.0) if role_embed_biases else 0.0
    mu_m = role_embed_biases.get("mu_match", 0.0) if role_embed_biases else 0.0

    same_lane = []
    cross_lane_matchups = {r: [] for r in ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]}
    role_synergies = {r: [] for r in ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]}

    for k, b_vec in role_embed_dict.items():
        if k == key:
            continue
        b_champ, b_role = k.split("|", 1)
        u_syn_b, v_syn_b = b_vec[0:16], b_vec[16:32]
        u_match_b, v_match_b = b_vec[32:48], b_vec[48:64]

        # Both directions of the matchup. delta drives the win probability;
        # each side's lane_score is shown as a sublabel so users can see the
        # underlying signal.
        a_vs_b = mu_m + bmu.get(key, 0.0) + bmv.get(k, 0.0) + float(np.dot(u_match_a, v_match_b))
        b_vs_a = mu_m + bmu.get(k, 0.0) + bmv.get(key, 0.0) + float(np.dot(u_match_b, v_match_a))
        delta = a_vs_b - b_vs_a
        a_win_prob = float(1.0 / (1.0 + np.exp(-delta / 5.0)))

        entry = {
            "champion": b_champ,
            "win_pct": round(a_win_prob * 100, 1),
            "lane_score": round(50.0 + a_vs_b, 1),
            "their_lane_score": round(50.0 + b_vs_a, 1),
        }

        if b_role == role:
            same_lane.append(entry)
        else:
            cross_lane_matchups[b_role].append(entry)

        # Synergy only with allies in *other* roles (you can't have two mids on one team)
        if b_role != role:
            a_with_b = mu_s + bsu.get(key, 0.0) + bsv.get(k, 0.0) + float(np.dot(u_syn_a, v_syn_b))
            b_with_a = mu_s + bsu.get(k, 0.0) + bsv.get(key, 0.0) + float(np.dot(u_syn_b, v_syn_a))
            syn_score = max(0.0, min(100.0, 50.0 + (a_with_b + b_with_a) / 2.0))
            role_synergies[b_role].append({"champion": b_champ, "score": round(syn_score, 1)})

    # Rank by win probability — the same metric the comparison panel uses.
    same_lane.sort(key=lambda x: x["win_pct"], reverse=True)
    for r in cross_lane_matchups:
        cross_lane_matchups[r].sort(key=lambda x: x["win_pct"], reverse=True)
    for r in role_synergies:
        role_synergies[r].sort(key=lambda x: x["score"], reverse=True)
        role_synergies[r] = role_synergies[r][:5]

    response = {
        "champion": champ,
        "role": role,
        "win_rate": champ_win_rates.get(champ, 50.0),
        "roles": champ_role_stats.get(champ, {}),
        "playable_roles": champ_roles_loose.get(champ, []),
        # Top 5 enemies the queried champion wins most against in this lane,
        # and the 5 they lose to most. `win_pct` is Akali's predicted win %
        # vs that enemy (sigmoid of lane-score delta), identical to what the
        # comparison panel shows.
        "same_lane_best": same_lane[:5],
        "same_lane_worst": same_lane[-5:][::-1],
        "cross_lane_matchups": {r: lst[:5] for r, lst in cross_lane_matchups.items()},
        "role_synergies": role_synergies,
    }

    if compare:
        ckey = f"{compare}|{compare_role}"
        if ckey in role_embed_dict:
            b = role_embed_dict[ckey]
            u_match_b, v_match_b = b[32:48], b[48:64]
            u_syn_b, v_syn_b = b[0:16], b[16:32]
            a_vs_b = mu_m + bmu.get(key, 0.0) + bmv.get(ckey, 0.0) + float(np.dot(u_match_a, v_match_b))
            b_vs_a = mu_m + bmu.get(ckey, 0.0) + bmv.get(key, 0.0) + float(np.dot(u_match_b, v_match_a))

            # Score deltas → win probability via sigmoid. The MF predicts
            # champ_score deviation from the 50-baseline, where champ_score
            # is dominated by the +3 win bonus. So a 5-point spread roughly
            # corresponds to a strong-favorite matchup; we use k=5 to map
            # delta=±5 → ~73%/27%, delta=±2 → ~60%/40%, delta=0 → 50%.
            delta = a_vs_b - b_vs_a
            a_win_prob = float(1.0 / (1.0 + np.exp(-delta / 5.0)))

            response["comparison"] = {
                "vs": compare,
                "vs_role": compare_role,
                "matchup_score": round(max(0.0, min(100.0, 50.0 + a_vs_b)), 1),
                "their_matchup_score": round(max(0.0, min(100.0, 50.0 + b_vs_a)), 1),
                "win_pct": round(a_win_prob * 100, 1),
                "their_win_pct": round((1.0 - a_win_prob) * 100, 1),
            }
            if compare_role != role:
                a_with_b = mu_s + bsu.get(key, 0.0) + bsv.get(ckey, 0.0) + float(np.dot(u_syn_a, v_syn_b))
                b_with_a = mu_s + bsu.get(ckey, 0.0) + bsv.get(key, 0.0) + float(np.dot(u_syn_b, v_syn_a))
                response["comparison"]["synergy_score"] = round(
                    max(0.0, min(100.0, 50.0 + (a_with_b + b_with_a) / 2.0)), 1)
        else:
            response["comparison_error"] = f"no data for {compare} in {compare_role}"

    return jsonify(response)


if __name__ == "__main__":
    print(f"Champions loaded: {len(champion_list)}")
    print(f"Draft model: {type(draft_model).__name__}")
    print(f"Embeddings: {len(embed_dict)} champions")
    app.run(debug=True, port=8080)
