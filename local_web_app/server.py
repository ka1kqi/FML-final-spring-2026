"""
Flask backend for the polished web draft simulator.

Loads the hand-rolled `lol_draft_pipeline` artifacts (Set Transformer
wide_deep + LightGBM hybrid + champion vocabulary + handcrafted stats +
isotonic calibrators) once at startup, then serves four endpoints to the
Vanilla-JS frontend.

Endpoints
---------
GET  /                  - index.html
GET  /api/meta          - dataset/run metadata (test AUC, recall@k, vocab size)
GET  /api/champions     - champion list with (best-guess) DDragon images
                          and role membership counts
POST /api/recommend     - top-K legal recommendations for the current
                          draft slot.  Honours model + algorithm selection
                          (greedy / beam / MCTS).
POST /api/evaluate      - blue-side win probability for the full draft

Run
---
    pip install flask
    python local_web_app/server.py
    open http://127.0.0.1:8090

The whole `local_web_app/` directory is tracked in git but artifacts /
caches / venvs inside it are ignored - same pattern as our other tools.
"""
from __future__ import annotations

import json
import logging
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask import Flask, jsonify, request, send_from_directory  # noqa: E402

import lol_draft_pipeline as P  # noqa: E402


# --------------------------------------------------------------------------- #
# One-time loaders
# --------------------------------------------------------------------------- #


ARTIFACTS_DIR = (ROOT / "artifacts").resolve()
DATA_CSV = (ROOT / "data" / "processed" / "matches.csv").resolve()
DDRAGON_VERSION = "16.8.1"

# The dataset uses some champion-name spellings that don't match Riot's
# DDragon exactly. Map data-name -> DDragon key here so the frontend can
# show portraits without 404s.
DDRAGON_KEY_MAP = {
    "FiddleSticks": "Fiddlesticks",
    "Wukong": "MonkeyKing",
    "Nunu": "Nunu",
    "Renata": "Renata",
}


_CFG = P.PipelineConfig(artifacts_dir=str(ARTIFACTS_DIR))


def _load_vocab() -> Dict[str, int]:
    return json.loads((ARTIFACTS_DIR / "champion_to_idx.json").read_text())


def _load_handcrafted():
    return P.load_pickle(ARTIFACTS_DIR / "handcrafted_stats.pkl")


def _list_available_models() -> List[str]:
    """Whichever model artefacts actually exist on disk."""
    out: List[str] = []
    for name, fn in (
        ("wide_deep", "wide_deep.pt"),
        ("hybrid", "lightgbm_with_embeddings.pkl"),
        ("baseline", "lightgbm_baseline.pkl"),
        ("teamcompnet", "teamcompnet.pt"),
        ("stacker", "stacker.pkl"),
    ):
        if (ARTIFACTS_DIR / fn).exists():
            out.append(name)
    return out


def _compute_role_membership(min_games: int = 5) -> Dict[str, List[str]]:
    """Per-champion list of roles played at least ``min_games`` times in the
    raw dataset. Used so the role filter in the UI hides inappropriate
    champions per role."""
    roles_for: Dict[str, set] = {}
    if not DATA_CSV.is_file():
        return {}
    try:
        import pandas as pd

        # Read just the four columns we need - the raw CSV is small enough
        # to load fully (~80k rows) but we keep the projection minimal.
        df = pd.read_csv(
            DATA_CSV,
            usecols=["championName", "teamPosition"],
            low_memory=False,
        ).dropna()
        df["role"] = df["teamPosition"].astype(str).str.upper().map(
            P.RIOT_POSITION_TO_ROLE
        )
        df = df.dropna(subset=["role"])
        counts = (
            df.groupby(["championName", "role"]).size().reset_index(name="n")
        )
        for _, row in counts.iterrows():
            if row["n"] >= min_games:
                roles_for.setdefault(row["championName"], set()).add(row["role"])
    except Exception as exc:
        app.logger.warning("Role membership compute failed: %s", exc)
    return {k: sorted(v) for k, v in roles_for.items()}


def _summary_metrics() -> Dict[str, object]:
    """Best-effort: surface the most recent run's metrics_summary.json."""
    runs = sorted(
        (ARTIFACTS_DIR / "runs").glob("*"), key=lambda p: p.stat().st_mtime
    ) if (ARTIFACTS_DIR / "runs").is_dir() else []
    runs = [r for r in runs if r.is_dir() and not r.name.startswith("_")]
    if not runs:
        return {}
    latest = runs[-1]
    out: Dict[str, object] = {"run_id": latest.name}
    summary_path = latest / "metrics_summary.json"
    if summary_path.is_file():
        out.update(json.loads(summary_path.read_text()))
    rec_metrics = latest / "metrics_recommender.json"
    if not rec_metrics.is_file():
        rec_metrics = ARTIFACTS_DIR / "metrics_recommender.json"
    if rec_metrics.is_file():
        try:
            out["recommender"] = json.loads(rec_metrics.read_text())
        except Exception:
            pass
    schema_path = latest / "schema_report.json"
    if schema_path.is_file():
        try:
            out["schema"] = json.loads(schema_path.read_text())
        except Exception:
            pass
    return out


# --------------------------------------------------------------------------- #
# App init
# --------------------------------------------------------------------------- #


app = Flask(__name__, static_folder="static", static_url_path="/static")
log = logging.getLogger("draft_web_app")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# ---------------------------------------------------------------------------
# Global error handler: NEVER let Flask render an HTML 500 page. The frontend
# parses every response as JSON, so an HTML body causes a cryptic
# "Unexpected token <" SyntaxError and hides the real error. By converting
# every uncaught exception (and HTTPException) into a JSON envelope with the
# traceback, the user / browser console always sees the actual problem.
# ---------------------------------------------------------------------------
from werkzeug.exceptions import HTTPException  # noqa: E402


@app.errorhandler(HTTPException)
def _handle_http_exception(e: HTTPException):
    log.warning("HTTP %s on %s: %s", e.code, request.path if request else "?", e.description)
    return jsonify({
        "error": e.description or e.name,
        "type": "HTTPException",
        "status": e.code,
    }), e.code or 500


@app.errorhandler(Exception)
def _handle_any_exception(e: Exception):
    tb = traceback.format_exc()
    log.error("Unhandled %s on %s\n%s", e.__class__.__name__,
              request.path if request else "?", tb)
    return jsonify({
        "error": str(e) or e.__class__.__name__,
        "type": e.__class__.__name__,
        "status": 500,
        "traceback": tb.splitlines(),
    }), 500


def _safe_endpoint(fn):
    """Decorator: any exception raised inside a route handler is caught and
    returned as a structured JSON 500. (The global @app.errorhandler above
    catches uncaught errors too, but explicit wrapping gives tighter control
    over per-endpoint metadata.)
    """
    from functools import wraps

    @wraps(fn)
    def _wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except HTTPException:
            raise  # let _handle_http_exception render it
        except Exception as exc:
            tb = traceback.format_exc()
            log.error("[%s] crashed: %s\n%s", fn.__name__, exc, tb)
            return jsonify({
                "error": str(exc) or exc.__class__.__name__,
                "type": exc.__class__.__name__,
                "endpoint": fn.__name__,
                "status": 500,
                "traceback": tb.splitlines(),
            }), 500
    return _wrapped


if not (ARTIFACTS_DIR / "champion_to_idx.json").exists():
    raise SystemExit(
        f"\nArtifacts not found at {ARTIFACTS_DIR}.\n"
        f"Train at least one model first:\n"
        f"  python lol_draft_pipeline.py train --fast-dev-run\n"
    )

_VOCAB = _load_vocab()
_HC = _load_handcrafted()
_ROLE_MEMBERSHIP = _compute_role_membership()
_AVAILABLE_MODELS = _list_available_models()
_SCORE_FN_CACHE: Dict[str, tuple] = {}
_RAW_SCORE_FN_CACHE: Dict[str, tuple] = {}


def _get_score_fns(model_name: str):
    """Calibrated score-fn pair (used by /api/evaluate for honest absolute prob)."""
    if model_name not in _SCORE_FN_CACHE:
        _SCORE_FN_CACHE[model_name] = P._build_score_fns(
            model_name, _CFG, _VOCAB, _HC
        )
    return _SCORE_FN_CACHE[model_name]


def _get_raw_score_fns(model_name: str):
    """Uncalibrated score-fn pair (used by /api/recommend for ranking).

    Isotonic regression fit on a small (~1k row) val set quantises the
    output to 2-3 unique values, which collapses recommendation
    rankings. For the recommender we want the model's full continuous
    output; calibration only matters for the absolute-value display in
    /api/evaluate.
    """
    if model_name in _RAW_SCORE_FN_CACHE:
        return _RAW_SCORE_FN_CACHE[model_name]

    art = Path(ARTIFACTS_DIR)
    extra_vocabs = P._load_extra_vocabs(_CFG)
    if model_name == "baseline":
        bundle = P.load_pickle(art / "lightgbm_baseline.pkl")
        feats = json.loads((art / "lightgbm_baseline_features.json").read_text())
        s = P.make_lgb_score_fn(bundle["model"], bundle["backend"], _VOCAB, _HC,
                                feats, None, calibrator=None,
                                extra_vocabs=extra_vocabs, cfg=_CFG)
        b = P.make_lgb_batch_score_fn(bundle["model"], bundle["backend"], _VOCAB, _HC,
                                      feats, None, calibrator=None,
                                      extra_vocabs=extra_vocabs, cfg=_CFG)
    elif model_name == "hybrid":
        bundle = P.load_pickle(art / "lightgbm_with_embeddings.pkl")
        feats = json.loads((art / "lightgbm_with_embeddings_features.json").read_text())
        emb = np.load(art / "champion_embeddings.npy")
        s = P.make_lgb_score_fn(bundle["model"], bundle["backend"], _VOCAB, _HC,
                                feats, emb, calibrator=None,
                                extra_vocabs=extra_vocabs, cfg=_CFG)
        b = P.make_lgb_batch_score_fn(bundle["model"], bundle["backend"], _VOCAB, _HC,
                                      feats, emb, calibrator=None,
                                      extra_vocabs=extra_vocabs, cfg=_CFG)
    elif model_name in ("teamcompnet", "wide_deep"):
        if not P._HAS_TORCH:
            return _get_score_fns(model_name)
        device = P.torch_device()
        if model_name == "teamcompnet":
            model = P._instantiate_teamcompnet(_CFG, _VOCAB, device)
            model.load_state_dict(P.torch.load(art / "teamcompnet.pt", map_location=device))
        else:
            model = P.WideDeepDraftNet(
                num_champions=len(_VOCAB),
                embedding_dim=_CFG.embedding_dim,
                hidden_dim=_CFG.hidden_dim,
                dropout=_CFG.dropout,
                combine=_CFG.wide_deep_combine,
            ).to(device)
            model.load_state_dict(P.torch.load(art / "wide_deep.pt", map_location=device))
        s = P.make_torch_score_fn(model, _VOCAB, device, calibrator=None)
        b = P.make_torch_batch_score_fn(model, _VOCAB, device, calibrator=None)
    else:
        return _get_score_fns(model_name)
    _RAW_SCORE_FN_CACHE[model_name] = (s, b)
    return s, b


def _make_recommender(model_name: str, use_raw: bool = False) -> P.Recommender:
    score_fn, batch_fn = (
        _get_raw_score_fns(model_name) if use_raw else _get_score_fns(model_name)
    )
    return P.Recommender(
        score_fn=score_fn, batch_score_fn=batch_fn,
        vocab=_VOCAB, handcrafted=_HC,
    )


def _draft_state_from_payload(body: dict) -> P.DraftState:
    """Build a DraftState from the JSON the frontend sends.

    Expected shape::

        {
          "blue_picks": {"top": "Fiora", "jungle": "LeeSin", ...},
          "red_picks":  {...},
          "bans":       ["Yasuo", "Yone", ...]
        }
    """
    state = P.DraftState()
    for r, c in (body.get("blue_picks") or {}).items():
        if c:
            state.blue_picks[r] = c
    for r, c in (body.get("red_picks") or {}).items():
        if c:
            state.red_picks[r] = c
    for b in body.get("bans") or []:
        if b:
            state.bans.append(b)
    return state


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@app.route("/")
@_safe_endpoint
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/recommend")
@_safe_endpoint
def recommend_page():
    """Recommendation-analysis page (form + detailed breakdown)."""
    return send_from_directory(app.static_folder, "recommend.html")


@app.route("/api/meta")
@_safe_endpoint
def api_meta():
    """Surfaced once at page-load so the UI can show real metric provenance
    (test AUC / recall@k / dataset size / split type) instead of
    fabricated linear conversions."""
    summary = _summary_metrics()
    return jsonify({
        "available_models": _AVAILABLE_MODELS,
        "champion_count": len(_VOCAB) - 1,
        "summary": summary,
        "ddragon_version": DDRAGON_VERSION,
    })


@app.route("/api/champions")
@_safe_endpoint
def api_champions():
    """List of champions for the grid + their playable roles."""
    out = []
    for name in sorted(c for c in _VOCAB.keys() if c != P.UNKNOWN_TOKEN):
        ddragon_key = DDRAGON_KEY_MAP.get(name, name)
        out.append({
            "name": name,
            "img": f"https://ddragon.leagueoflegends.com/cdn/"
                   f"{DDRAGON_VERSION}/img/champion/{ddragon_key}.png",
            "roles": _ROLE_MEMBERSHIP.get(name, []),
        })
    return jsonify(out)


@app.route("/api/recommend", methods=["POST"])
@_safe_endpoint
def api_recommend():
    """Return top-K legal champion recommendations with calibrated win prob."""
    body = request.get_json(force=True, silent=True) or {}
    side = body.get("side", "blue")
    role = body.get("role", "top")
    top_k = int(body.get("top_k", 5))
    model_name = body.get("model", _AVAILABLE_MODELS[0])
    algorithm = body.get("algorithm", "greedy")  # greedy | beam | mcts
    beam_width = int(body.get("beam_width", 5))
    beam_depth = int(body.get("beam_depth", 2))
    mcts_simulations = int(body.get("mcts_simulations", 64))

    if model_name not in _AVAILABLE_MODELS:
        return jsonify({"error": f"unknown model {model_name!r}"}), 400
    state = _draft_state_from_payload(body)
    if role in state.picks_for(side):
        return jsonify({"error": f"{side} {role} already filled"}), 400

    # Use uncalibrated probs for ranking unless caller explicitly opts in.
    use_calibrated = bool(body.get("calibrated", False))
    rec = _make_recommender(model_name, use_raw=not use_calibrated)
    if algorithm == "mcts":
        results = rec.mcts(
            state, side, role,
            n_simulations=mcts_simulations, k=top_k, depth=max(2, beam_depth),
        )
    elif algorithm == "beam":
        results = rec.beam_search(
            state, side, role,
            beam_width=beam_width, depth=beam_depth, k=top_k,
        )
    else:
        results = rec.top_k(state, side, role, k=top_k)

    # The torch batch score-fn returns float32; isotonic calibration converts to
    # float64 implicitly. When we bypass the calibrator we have to coerce.
    def _scrub(v):
        if isinstance(v, (np.floating, np.integer)):
            return float(v)
        if isinstance(v, dict):
            return {k: _scrub(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [_scrub(x) for x in v]
        return v

    results = [_scrub(r) for r in results]

    # Optional per-pair breakdown for the recommendation-analysis UI:
    # synergy of (candidate, each ally) + counter of (candidate, each enemy).
    if body.get("include_breakdown"):
        ally_picks = list(state.picks_for(side).items())
        enemy_picks = list(state.picks_for("red" if side == "blue" else "blue").items())
        for r in results:
            cand = r["champion"]
            r["ally_breakdown"] = [
                {"role": role_, "champion": ch,
                 "synergy": float(_HC.synergy(cand, ch))}
                for role_, ch in ally_picks
            ]
            r["enemy_breakdown"] = [
                {"role": role_, "champion": ch,
                 "counter": float(_HC.counter(cand, ch))}
                for role_, ch in enemy_picks
            ]
            r["sample_count"] = int(_HC.champ_winrate.get(cand, _HC.base_winrate) * 0)
            r["historical_winrate"] = float(_HC.winrate(cand))

    return jsonify({
        "side": side, "role": role,
        "model": model_name, "algorithm": algorithm,
        "current_blue_winprob": float(rec.score_fn(state)),
        "recommendations": results,
    })


@app.route("/api/evaluate", methods=["POST"])
@_safe_endpoint
def api_evaluate():
    """Predict blue-side win probability for the full draft (calibrated)."""
    body = request.get_json(force=True, silent=True) or {}
    model_name = body.get("model", _AVAILABLE_MODELS[0])
    if model_name not in _AVAILABLE_MODELS:
        return jsonify({"error": f"unknown model {model_name!r}"}), 400
    state = _draft_state_from_payload(body)
    score_fn, _ = _get_score_fns(model_name)
    blue_wp = float(score_fn(state))
    return jsonify({
        "model": model_name,
        "blue_win_prob": blue_wp,
        "red_win_prob": 1.0 - blue_wp,
        "blue_complete": len(state.blue_picks) == 5,
        "red_complete": len(state.red_picks) == 5,
    })


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


if __name__ == "__main__":
    print(f"Champions loaded:   {len(_VOCAB)-1}")
    print(f"Available models:   {_AVAILABLE_MODELS}")
    print(f"Role membership:    {len(_ROLE_MEMBERSHIP)} champions classified")
    summary = _summary_metrics()
    if summary:
        best = (summary.get("models") or {})
        if best:
            top_name = max(best, key=lambda n: best[n].get("roc_auc") or 0)
            top_auc = best[top_name].get("roc_auc")
            print(f"Best model on test: {top_name}  AUC={top_auc:.4f}")
    print()
    print(f"Open http://127.0.0.1:8090 in your browser.")
    app.run(debug=False, host="127.0.0.1", port=8090)
