"""
Streamlit dashboard for the LoL draft pipeline.

Run with::

    streamlit run local_dashboard/app.py -- --artifacts-dir artifacts

The CLI flag is positional after the ``--`` so streamlit doesn't parse it.
Every tab is fault-tolerant: missing artifacts surface as info / warning
boxes instead of tracebacks.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Make the sibling utils module importable regardless of cwd.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import dashboard_utils as du

# Repo root (one up) - needed so we can call back into lol_draft_pipeline
# for the live recommendation playground.
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_argv() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--default-refresh", type=int, default=5)
    args, _ = parser.parse_known_args()
    return args


CLI = _parse_argv()


# --------------------------------------------------------------------------- #
# Page config + header
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="LoL Draft Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# A tiny CSS tweak so metric cards don't shrink awkwardly on narrow viewports
st.markdown(
    """
    <style>
      div[data-testid="stMetric"] { background: rgba(127, 127, 127, 0.06); padding: 0.6rem; border-radius: 8px; }
      .small-muted { color: #999; font-size: 0.85rem; }
      .ok-pill { background:#1a7f37; color:white; padding:2px 8px; border-radius:8px; font-size:0.75rem; }
      .warn-pill { background:#bf8700; color:white; padding:2px 8px; border-radius:8px; font-size:0.75rem; }
      .err-pill { background:#cf222e; color:white; padding:2px 8px; border-radius:8px; font-size:0.75rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🛡️  LoL Draft Pipeline Dashboard")
st.caption(
    "Local dashboard — live training, model comparison, calibration, "
    "embedding explorer, recommender playground, leakage audit."
)


# --------------------------------------------------------------------------- #
# Sidebar - run picker + auto refresh
# --------------------------------------------------------------------------- #


def _status_pill(status: Optional[str]) -> str:
    if not status:
        return ""
    if status == "success":
        return f"<span class='ok-pill'>{status}</span>"
    if status in ("running", "no_events"):
        return f"<span class='warn-pill'>{status}</span>"
    return f"<span class='err-pill'>{status}</span>"


with st.sidebar:
    st.subheader("Configuration")
    artifacts_dir = st.text_input("artifacts-dir", value=CLI.artifacts_dir)
    runs = du.list_runs(artifacts_dir)

    if not runs:
        st.warning("No runs found.")
        st.code(
            "python lol_draft_pipeline.py train \\\n"
            f"  --data-dir data --artifacts-dir {artifacts_dir}",
            language="bash",
        )
        run_choice: Optional[du.RunRef] = None
    else:
        labels = [r.display() for r in runs]
        idx = st.selectbox(
            "run", options=list(range(len(runs))),
            format_func=lambda i: labels[i], index=0,
        )
        run_choice = runs[idx]

    st.divider()
    auto_refresh = st.toggle("Auto refresh", value=False)
    refresh_interval = st.select_slider(
        "Interval (sec)", options=[2, 3, 5, 10, 15, 30], value=CLI.default_refresh
    )

    st.divider()
    st.caption(
        "All artifacts and dashboard files are git-ignored; data and weights stay local."
    )
    if run_choice is not None:
        st.markdown(f"**Run:** `{run_choice.run_id}`")
        st.markdown(f"**Status:** {_status_pill(run_choice.status)}", unsafe_allow_html=True)
        if run_choice.duration_seconds:
            st.markdown(f"**Duration:** {run_choice.duration_seconds:.1f} s")
        if run_choice.started_at:
            st.markdown(f"**Started:** {run_choice.started_at}")


if run_choice is None:
    st.info("Train a model to populate the dashboard.")
    st.stop()

run_dir = run_choice.path

# Cached loaders so flipping tabs doesn't re-read the same files.
@st.cache_data(show_spinner=False, ttl=2.0)
def _cached_summary(p: str): return du.load_metrics_summary(p)
@st.cache_data(show_spinner=False, ttl=2.0)
def _cached_comparison(p: str): return du.load_model_comparison(p)
@st.cache_data(show_spinner=False, ttl=2.0)
def _cached_calibration(p: str): return du.load_calibration(p)
@st.cache_data(show_spinner=False, ttl=2.0)
def _cached_predictions(p: str): return du.load_predictions(p)
@st.cache_data(show_spinner=False, ttl=2.0)
def _cached_feat_imp(p: str): return du.load_feature_importance(p)
@st.cache_data(show_spinner=False, ttl=2.0)
def _cached_emb(p: str): return du.load_embeddings(p)
@st.cache_data(show_spinner=False, ttl=2.0)
def _cached_examples(p: str): return du.load_recommendation_examples(p)
@st.cache_data(show_spinner=False, ttl=2.0)
def _cached_leakage(p: str): return du.load_leakage_audit(p)
@st.cache_data(show_spinner=False, ttl=2.0)
def _cached_schema(p: str): return du.load_schema_report(p)
@st.cache_data(show_spinner=False, ttl=2.0)
def _cached_events(p: str): return du.load_events(p)


# Pre-load everything we need to avoid repeated file IO across tabs.
events = _cached_events(str(run_dir))
summary = _cached_summary(str(run_dir))
comparison = _cached_comparison(str(run_dir))
calibration = _cached_calibration(str(run_dir))
predictions = _cached_predictions(str(run_dir))
feat_imp = _cached_feat_imp(str(run_dir))
embeddings = _cached_emb(str(run_dir))
examples = _cached_examples(str(run_dir))
leakage = _cached_leakage(str(run_dir))
schema = _cached_schema(str(run_dir))


# --------------------------------------------------------------------------- #
# Tab layout
# --------------------------------------------------------------------------- #

TAB_NAMES = [
    "1. Overview",
    "2. Live Training",
    "3. Model Comparison",
    "4. Calibration",
    "5. Confusion / Threshold",
    "6. Feature Importance",
    "7. Champion Embeddings",
    "8. Recommendation Playground",
    "9. Beam Search Visualizer",
    "10. Leakage & Schema",
    "11. Artifacts Browser",
]
tabs = st.tabs(TAB_NAMES)


# --------------------------------------------------------------------------- #
# Tab 1: Overview
# --------------------------------------------------------------------------- #


def _render_overview():
    st.subheader("Run overview")

    cols = st.columns(4)
    if summary:
        models = summary.get("models", {}) or {}
        best = summary.get("best_model")
        best_auc = summary.get("best_test_auc")
        cols[0].metric("Best model", best or "-")
        cols[1].metric("Best test AUC", f"{best_auc:.4f}" if best_auc else "-")
        if best and best in models:
            m = models[best]
            cols[2].metric("Best test accuracy", f"{m.get('accuracy', float('nan')):.4f}")
            cols[3].metric("Best test F1", f"{m.get('f1', float('nan')):.4f}")
    else:
        st.info("metrics_summary.json not yet written (run still in progress?)")

    # Recommender hit-rate, if recorded
    rec = (summary or {}).get("recommender")
    rec_path = run_dir / "metrics_recommender.json"
    if rec is None and rec_path.is_file():
        rec = json.loads(rec_path.read_text())
    if rec:
        rcols = st.columns(4)
        rcols[0].metric("Recall@1", f"{rec.get('recall@1', float('nan')):.3f}")
        rcols[1].metric("Recall@3", f"{rec.get('recall@3', float('nan')):.3f}")
        rcols[2].metric("Recall@5", f"{rec.get('recall@5', float('nan')):.3f}")
        rcols[3].metric("MRR",       f"{rec.get('mrr', float('nan')):.3f}")

    st.divider()

    # Schema + leakage at a glance
    sch_col, leak_col = st.columns(2)
    with sch_col:
        st.markdown("#### Dataset")
        if schema:
            st.write(
                {
                    "matches": schema.get("matches"),
                    "blue_win_rate": schema.get("blue_win_rate"),
                    "champion_vocab_size": schema.get("champion_vocab_size"),
                    "split": schema.get("split"),
                }
            )
        else:
            st.info("schema_report.json missing")
    with leak_col:
        st.markdown("#### Leakage")
        if leakage:
            risk = leakage.get("leakage_risk_detected", False)
            if risk:
                st.error("Leakage risk detected!")
            else:
                st.success("No leakage risk.")
            st.caption(
                f"Excluded post-game columns: "
                f"{', '.join(leakage.get('excluded_post_game_columns', [])) or '(none in source data)'}"
            )
        else:
            st.info("leakage_audit.json missing")

    st.divider()

    st.markdown("#### Model comparison")
    if not comparison.empty:
        styled = du.highlight_best(comparison)
        st.dataframe(styled, use_container_width=True)
        chart_cols = st.columns(3)
        for col, metric in zip(chart_cols, ("auc", "log_loss", "brier")):
            if metric not in comparison.columns:
                continue
            higher_better = metric == "auc"
            fig = px.bar(
                comparison.sort_values(metric, ascending=not higher_better),
                x="model", y=metric, color="model", text=metric,
                title=f"Test {metric}",
            )
            fig.update_layout(showlegend=False, height=320, margin=dict(t=40, b=10))
            fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            col.plotly_chart(fig, use_container_width=True)
    else:
        st.info("model_comparison.csv not found yet")


# --------------------------------------------------------------------------- #
# Tab 2: Live Training
# --------------------------------------------------------------------------- #


def _events_to_metric_frame(events: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for ev in events:
        if ev.get("event_type") != "train_metric":
            continue
        m = ev.get("metrics") or {}
        rows.append(
            {
                "timestamp": ev.get("timestamp"),
                "model": ev.get("model"),
                "step": ev.get("epoch") or ev.get("iteration") or 0,
                "split": ev.get("split"),
                **m,
            }
        )
    return pd.DataFrame(rows)


def _render_live_training():
    st.subheader("Live training")
    if not events:
        st.info("No events.jsonl yet. The dashboard will start streaming as soon as training emits its first event.")
        return

    df = _events_to_metric_frame(events)
    if df.empty:
        st.info("Events found, but no train_metric events recorded yet.")
        return

    models = sorted(df["model"].dropna().unique().tolist())
    sel_models = st.multiselect("models", models, default=models)
    df = df[df["model"].isin(sel_models)] if sel_models else df

    metric_cols = [
        c
        for c in ("auc", "binary_logloss", "val_loss", "train_loss", "val_auc",
                  "log_loss", "f1", "accuracy")
        if c in df.columns
    ]
    if not metric_cols:
        st.info("No recognised numeric metrics in event stream.")
        return

    metric = st.selectbox("metric", metric_cols, index=0)
    plot_df = df.dropna(subset=[metric]).copy()
    if plot_df.empty:
        st.info(f"No values for {metric} yet.")
    else:
        fig = px.line(
            plot_df,
            x="step", y=metric, color="model", line_dash="split",
            markers=False, title=f"{metric} (live)",
        )
        fig.update_layout(height=420, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Recent events (last 20)")
    tail = events[-20:][::-1]
    st.json(tail, expanded=False)


# --------------------------------------------------------------------------- #
# Tab 3: Model Comparison
# --------------------------------------------------------------------------- #


def _render_model_comparison():
    st.subheader("Model comparison (test split)")

    if comparison.empty:
        st.info("model_comparison.csv missing.")
        return

    styled = du.highlight_best(comparison)
    st.dataframe(styled, use_container_width=True)

    metric_cols = [c for c in ("accuracy", "f1", "auc", "log_loss", "brier") if c in comparison.columns]

    melted = comparison.melt(id_vars=["model"], value_vars=metric_cols,
                             var_name="metric", value_name="value")
    fig = px.bar(
        melted, x="metric", y="value", color="model",
        barmode="group", text="value",
        title="All metrics, side-by-side",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

    # Radar-style overview, normalised so higher==better
    radar_metrics = [m for m in ("accuracy", "f1", "auc") if m in comparison.columns]
    inverse_metrics = [m for m in ("log_loss", "brier") if m in comparison.columns]
    if radar_metrics or inverse_metrics:
        radar = go.Figure()
        for _, row in comparison.iterrows():
            values = []
            theta = []
            for m in radar_metrics:
                values.append(float(row[m]))
                theta.append(m)
            for m in inverse_metrics:
                # Invert so larger is "more is better".
                v = float(row[m])
                values.append(1.0 - v)
                theta.append(f"1-{m}")
            if not values:
                continue
            values.append(values[0]); theta.append(theta[0])
            radar.add_trace(go.Scatterpolar(r=values, theta=theta, fill="toself", name=row["model"]))
        radar.update_layout(height=420, polar=dict(radialaxis=dict(range=[0, 1])),
                            title="Composite radar (higher = better; log_loss & brier inverted)")
        st.plotly_chart(radar, use_container_width=True)

    st.download_button(
        "Download CSV",
        data=comparison.to_csv(index=False).encode(),
        file_name="model_comparison.csv",
        mime="text/csv",
    )

    st.markdown(
        """
        **Reading this tab**

        * `baseline` — strong tabular LightGBM with handcrafted synergy/counter
          features. Robust no matter how thin the deep models train.
        * `teamcompnet` — the embedding model whose champion vectors power the
          hybrid LightGBM and feed the recommendation explanations.
        * `lightgbm_with_embeddings` (a.k.a. `hybrid`) — the recommended scorer
          for the recommender; combines tabular features with team-level
          embedding statistics.
        * `wide_deep` — end-to-end PyTorch combining a wide one-hot branch with
          a deep TeamCompNet body.
        * `recommender` — top-k / beam search wrapper; evaluated separately on
          *hidden-pick* reconstruction (recall@k, MRR).
        """
    )


# --------------------------------------------------------------------------- #
# Tab 4: Calibration
# --------------------------------------------------------------------------- #


def _render_calibration():
    st.subheader("Reliability diagram")

    df = calibration.copy() if not calibration.empty else pd.DataFrame()
    fallback_used = False
    if df.empty and not predictions.empty:
        df = du.compute_calibration_from_predictions(predictions)
        fallback_used = True
    if df.empty:
        st.info("calibration.csv and predictions_test.csv both missing.")
        return
    if fallback_used:
        st.markdown(du.format_info_box("calibration.csv missing - recomputed from predictions."))

    df = df.copy()
    if "model" not in df.columns and "model_name" in df.columns:
        df = df.rename(columns={"model_name": "model"})
    df = df[df["n"].astype(float) > 0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(dash="dash", color="gray"),
                             name="perfect calibration"))
    for name, sub in df.groupby("model"):
        fig.add_trace(
            go.Scatter(
                x=sub["mean_pred"], y=sub["empirical_rate"],
                mode="lines+markers", name=name,
                hovertext=sub.get("bucket"),
            )
        )
    fig.update_layout(height=420, xaxis_title="mean predicted prob",
                      yaxis_title="empirical win rate", title="Reliability diagram")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.bar(df, x="bucket", y="n", color="model", barmode="group",
                  title="Bucket counts")
    fig2.update_layout(height=320)
    st.plotly_chart(fig2, use_container_width=True)

    if not comparison.empty and "brier" in comparison.columns:
        fig3 = px.bar(comparison.sort_values("brier"), x="model", y="brier",
                      title="Brier score (lower = better)", text="brier")
        fig3.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig3.update_layout(height=320)
        st.plotly_chart(fig3, use_container_width=True)

    st.caption(
        "If a model says 0.60 it should win ~60% of the time at that probability. "
        "Lines hugging the diagonal are well calibrated."
    )


# --------------------------------------------------------------------------- #
# Tab 5: Confusion / Threshold
# --------------------------------------------------------------------------- #


def _render_confusion_threshold():
    st.subheader("Threshold analysis")
    if predictions.empty:
        st.info("predictions_test.csv missing.")
        return

    models = sorted(predictions["model_name"].dropna().unique().tolist())
    if not models:
        st.info("No model_name column in predictions.")
        return
    model = st.selectbox("model", models, key="threshold_model")
    threshold = st.slider("threshold", 0.0, 1.0, 0.5, 0.01)

    sub = predictions[predictions["model_name"] == model]
    metrics = du.compute_metrics_from_predictions(sub, threshold=threshold).get(model, {})

    cols = st.columns(4)
    cols[0].metric("Accuracy", f"{metrics.get('accuracy', float('nan')):.4f}")
    cols[1].metric("Precision", f"{metrics.get('precision', float('nan')):.4f}")
    cols[2].metric("Recall", f"{metrics.get('recall', float('nan')):.4f}")
    cols[3].metric("F1", f"{metrics.get('f1', float('nan')):.4f}")

    cm = metrics.get("confusion_matrix")
    if cm:
        fig = go.Figure(
            data=go.Heatmap(
                z=cm, x=["pred 0", "pred 1"], y=["true 0", "true 1"],
                colorscale="Blues", text=cm, texttemplate="%{text}",
            )
        )
        fig.update_layout(title=f"Confusion matrix @ threshold={threshold:.2f}", height=360)
        st.plotly_chart(fig, use_container_width=True)

    fpr, tpr = metrics.get("roc_fpr") or [], metrics.get("roc_tpr") or []
    if fpr and tpr:
        roc = go.Figure()
        roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                 name=f"AUC={metrics.get('roc_auc', float('nan')):.3f}"))
        roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                 line=dict(dash="dash", color="gray"), name="random"))
        roc.update_layout(title="ROC curve", height=360,
                          xaxis_title="FPR", yaxis_title="TPR")
        st.plotly_chart(roc, use_container_width=True)

    pp, pr = metrics.get("pr_precision") or [], metrics.get("pr_recall") or []
    if pp and pr:
        pr_fig = go.Figure(go.Scatter(x=pr, y=pp, mode="lines"))
        pr_fig.update_layout(title="Precision-Recall curve", height=360,
                             xaxis_title="recall", yaxis_title="precision")
        st.plotly_chart(pr_fig, use_container_width=True)


# --------------------------------------------------------------------------- #
# Tab 6: Feature importance
# --------------------------------------------------------------------------- #


def _is_embedding_feature(name: str) -> bool:
    n = (name or "").lower()
    return any(tag in n for tag in (
        "emb_", "embedding", "blue_pair", "red_pair", "cross_pair",
        "blue_synergy", "red_synergy", "blue_counter", "red_counter",
    ))


def _render_feature_importance():
    st.subheader("Feature importance (LightGBM)")
    if feat_imp.empty:
        st.info("feature_importance.csv missing - LightGBM may not have run, or fallback to HistGB skipped importance.")
        return

    models = sorted(feat_imp["model_name"].dropna().unique().tolist())
    model = st.selectbox("model", models, key="featimp_model")
    types = sorted(feat_imp["importance_type"].dropna().unique().tolist())
    itype = st.selectbox("importance type", types, key="featimp_type")
    search = st.text_input("filter feature name (substring)", value="").strip().lower()
    top_n = st.slider("Top N", 5, 60, 30)

    sub = feat_imp[(feat_imp["model_name"] == model) & (feat_imp["importance_type"] == itype)].copy()
    if search:
        sub = sub[sub["feature"].str.lower().str.contains(search)]
    sub = sub.sort_values("importance", ascending=False).head(top_n)
    sub["kind"] = sub["feature"].apply(lambda c: "embedding" if _is_embedding_feature(c) else "tabular")

    if sub.empty:
        st.info("No matching features.")
        return

    fig = px.bar(sub.iloc[::-1], x="importance", y="feature", color="kind",
                 orientation="h", title=f"{model} / {itype} - top {len(sub)}")
    fig.update_layout(height=max(360, 22 * len(sub)))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(sub, use_container_width=True)


# --------------------------------------------------------------------------- #
# Tab 7: Champion embedding explorer
# --------------------------------------------------------------------------- #


def _render_embedding_explorer():
    st.subheader("Champion embedding explorer (TeamCompNet)")
    if embeddings.empty:
        st.info("embedding_champions.csv missing - train TeamCompNet first.")
        return

    method = "umap" if du._HAS_UMAP else "pca"
    method = st.radio("projection", ("pca", "umap"),
                      horizontal=True, index=0 if method == "pca" else 1,
                      help="UMAP appears only if the umap-learn package is installed.")
    coords = du.compute_pca_embeddings(embeddings, n_components=2, method=method)
    if coords.empty:
        st.info("Embedding projection failed.")
        return

    color_options = [c for c in ("most_common_role", "win_rate", "sample_count") if c in coords.columns]
    color_by = st.selectbox("colour by", ["(none)"] + color_options) if color_options else "(none)"
    size_col = "sample_count" if "sample_count" in coords.columns else None

    fig = px.scatter(
        coords,
        x="component_1", y="component_2",
        hover_name="champion",
        color=None if color_by == "(none)" else color_by,
        size=size_col,
        title=f"Champion embeddings ({method.upper()})",
    )
    fig.update_layout(height=540)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("#### Nearest neighbours")
    champ = st.selectbox("champion", sorted(embeddings["champion"].tolist()))
    neighbours = du.find_nearest_champions(embeddings, champ, top_k=10)
    if neighbours.empty:
        st.info("No neighbours computed.")
    else:
        st.dataframe(neighbours, use_container_width=True)
    st.caption(
        "Cosine similarity in the learned embedding space. "
        "Don't read these as 'is a tank/assassin' - they reflect the model's view of "
        "compositional similarity given the training data."
    )


# --------------------------------------------------------------------------- #
# Tab 8: Recommendation playground
# --------------------------------------------------------------------------- #


def _try_import_pipeline():
    """Lazy-import the pipeline so dashboard works even when the import fails."""
    try:
        import lol_draft_pipeline as P
        return P
    except Exception as exc:
        st.error(f"Could not import lol_draft_pipeline: {exc}")
        return None


def _render_recommendation_playground():
    st.subheader("Recommendation playground")
    P = _try_import_pipeline()
    if P is None:
        return

    # Champion list from the embedding CSV (includes the full vocab).
    champ_list = sorted(embeddings["champion"].tolist()) if not embeddings.empty else []
    if not champ_list:
        # Fall back to vocab JSON
        vocab = du.load_champion_vocab(run_dir)
        champ_list = sorted([c for c in vocab.keys() if c != "<UNK>"])
    if not champ_list:
        st.info("Champion vocab not yet written - train at least once.")
        return
    options = ["(empty)"] + champ_list

    col_b, col_r = st.columns(2)
    blue_picks: Dict[str, str] = {}
    red_picks: Dict[str, str] = {}
    for role in du.ROLES:
        with col_b:
            b = st.selectbox(f"blue {role}", options, key=f"blue_{role}")
            if b != "(empty)":
                blue_picks[role] = b
        with col_r:
            r_ = st.selectbox(f"red {role}", options, key=f"red_{role}")
            if r_ != "(empty)":
                red_picks[role] = r_
    bans_str = st.text_input("bans (comma-separated, optional)", value="")
    bans = du.parse_bans_list(bans_str)

    cfg_cols = st.columns(4)
    side = cfg_cols[0].radio("side to move", ("blue", "red"), horizontal=True)
    role_to_pick = cfg_cols[1].selectbox("role to pick", du.ROLES,
                                          index=du.ROLES.index("top"))
    top_k = cfg_cols[2].number_input("top-k", min_value=1, max_value=15, value=5)
    available_models = []
    art = Path(artifacts_dir)
    if (art / "lightgbm_with_embeddings.pkl").exists():
        available_models.append("hybrid")
    if (art / "lightgbm_baseline.pkl").exists():
        available_models.append("baseline")
    if (art / "wide_deep.pt").exists():
        available_models.append("wide_deep")
    if (art / "teamcompnet.pt").exists():
        available_models.append("teamcompnet")
    if not available_models:
        st.info("Train models first - no scorer artifact available at the top-level artifacts/ directory.")
        return
    model_name = cfg_cols[3].selectbox("model", available_models)

    beam_cols = st.columns(3)
    use_beam = beam_cols[0].toggle("Use beam search", value=False)
    beam_width = beam_cols[1].slider("beam_width", 2, 10, 5, disabled=not use_beam)
    beam_depth = beam_cols[2].slider("beam_depth", 1, 4, 2, disabled=not use_beam)

    if not st.button("Recommend", type="primary"):
        st.info("Set the draft and click Recommend.")
        return

    cfg = P.PipelineConfig(
        artifacts_dir=str(artifacts_dir), data_dir="data",
        beam_width=beam_width, beam_depth=beam_depth, top_k=int(top_k),
    )
    vocab = du.load_champion_vocab(run_dir) or P.json.loads(
        (Path(artifacts_dir) / "champion_to_idx.json").read_text()
    )
    handcrafted_path = Path(artifacts_dir) / "handcrafted_stats.pkl"
    if not handcrafted_path.is_file():
        st.error("handcrafted_stats.pkl missing - cannot build score function.")
        return
    handcrafted = P.load_pickle(handcrafted_path)

    try:
        score_fn, batch_fn = P._build_score_fns(model_name, cfg, vocab, handcrafted)
    except Exception as exc:
        st.error(f"Could not build score function: {exc}")
        return

    state = P.DraftState(blue_picks=blue_picks, red_picks=red_picks, bans=bans)
    if role_to_pick in state.picks_for(side):
        st.warning(f"{side} {role_to_pick} is already filled with "
                   f"{state.picks_for(side)[role_to_pick]} - clear it before recommending.")
        return

    rec = P.Recommender(score_fn=score_fn, vocab=vocab,
                        handcrafted=handcrafted, batch_score_fn=batch_fn)

    cur = float(score_fn(state))
    cur_for_side = cur if side == "blue" else 1 - cur
    st.metric(f"Current {side} win prob", f"{cur_for_side:.4f}")

    if use_beam:
        results = rec.beam_search(state, side, role_to_pick,
                                  beam_width=beam_width, depth=beam_depth, k=int(top_k))
    else:
        results = rec.top_k(state, side, role_to_pick, k=int(top_k))

    if not results:
        st.warning("No legal candidates produced.")
        return

    df = pd.DataFrame(results)
    df.insert(0, "rank", range(1, len(df) + 1))
    st.dataframe(df, use_container_width=True)

    fig = px.bar(df, x="champion", y="win_prob", color="champion",
                 text="win_prob", title=f"Top {len(df)} candidates")
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_layout(showlegend=False, height=380)
    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------- #
# Tab 9: Beam search visualizer
# --------------------------------------------------------------------------- #


def _render_beam_visualizer():
    st.subheader("Beam search trace")
    trace = du.safe_read_json(run_dir / "beam_search_trace.json")
    if trace is None and examples:
        # Reuse the second canned example, which is typically the beam search one
        trace = next(
            (ex for ex in examples if "beam" in (ex.get("name") or "").lower()), None
        )
    if trace is None:
        st.info("No beam_search_trace.json or canned beam example available. Run `train` to generate canned examples, or run the recommender with --beam-search.")
        return

    if "results" in trace:
        st.markdown("#### Final ranking")
        df = pd.DataFrame(trace["results"])
        df.insert(0, "rank", range(1, len(df) + 1))
        st.dataframe(df, use_container_width=True)
        fig = px.bar(df, x="champion", y="win_prob", color="champion",
                     text="win_prob", title="Beam-search top-k")
        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig, use_container_width=True)
        st.json(
            {
                "model": trace.get("model"),
                "side": trace.get("side"),
                "role": trace.get("role"),
                "beam_width": trace.get("beam_width"),
                "beam_depth": trace.get("beam_depth"),
                "state": trace.get("state"),
            },
            expanded=False,
        )
        st.caption(
            "Beam search keeps the top `beam_width` children at our turns and "
            "assumes the opponent picks the value-minimising child at theirs "
            "(minimax flavour). For depth=1 this collapses to greedy top-k."
        )
        return

    if "levels" in trace:
        st.markdown("Tree-style trace not yet rendered - showing raw JSON.")
        st.json(trace, expanded=False)
        return

    st.info("Trace format not recognised; showing raw JSON.")
    st.json(trace, expanded=False)


# --------------------------------------------------------------------------- #
# Tab 10: Leakage & schema
# --------------------------------------------------------------------------- #


def _render_leakage_schema():
    st.subheader("Leakage audit")
    if leakage:
        risk = leakage.get("leakage_risk_detected", False)
        if risk:
            st.error("Leakage risk detected!")
        else:
            st.success("No leakage detected. Post-game columns are excluded; train-only stats; documented audit.")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Draft-time included**")
            for col in leakage.get("included_columns", []):
                st.markdown(f"- {col}")
        with c2:
            st.markdown("**Excluded post-game columns**")
            excluded = leakage.get("excluded_post_game_columns", [])
            if excluded:
                for col in excluded:
                    st.markdown(f"- ~~{col}~~")
            else:
                st.caption("(none of the blocklisted columns are present in the source data)")
        st.markdown("**Blocklist (always excluded by policy)**")
        st.code(", ".join(leakage.get("post_game_blocklist", [])), language="text")
        susp = leakage.get("suspicious_columns") or []
        if susp:
            st.markdown(du.format_warning_box(
                f"Suspicious columns ignored by main models: {', '.join(susp)}"
            ))
        st.caption(leakage.get("notes", ""))
    else:
        st.info("leakage_audit.json missing.")

    st.divider()
    st.subheader("Schema report")
    if schema:
        st.json(schema)
    else:
        st.info("schema_report.json missing.")


# --------------------------------------------------------------------------- #
# Tab 11: Artifacts browser
# --------------------------------------------------------------------------- #


def _render_artifact_browser():
    st.subheader(f"Artifacts in {run_dir}")
    files = du.list_run_files(run_dir)
    if files.empty:
        st.info("Run directory is empty.")
        return

    st.dataframe(files, use_container_width=True)

    name = st.selectbox("preview file", files["name"].tolist())
    target = run_dir / name
    ext = target.suffix.lower()

    if ext == ".json":
        payload = du.safe_read_json(target)
        st.json(payload if payload is not None else {})
    elif ext == ".csv":
        df = du.safe_read_csv(target)
        st.dataframe(df.head(500), use_container_width=True)
        st.caption(f"Showing up to 500 of {len(df)} rows.")
    elif ext in (".txt", ".log", ".jsonl", ".md"):
        st.code(du.preview_text_file(target), language="text")
    else:
        try:
            size_kb = round(target.stat().st_size / 1024, 2)
        except OSError:
            size_kb = "?"
        st.markdown(f"Binary file (`{ext}`, {size_kb} KB) - download to inspect.")

    if target.is_file():
        try:
            data = target.read_bytes()
            st.download_button("Download", data=data, file_name=target.name)
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Render tabs
# --------------------------------------------------------------------------- #

with tabs[0]:
    _render_overview()
with tabs[1]:
    _render_live_training()
with tabs[2]:
    _render_model_comparison()
with tabs[3]:
    _render_calibration()
with tabs[4]:
    _render_confusion_threshold()
with tabs[5]:
    _render_feature_importance()
with tabs[6]:
    _render_embedding_explorer()
with tabs[7]:
    _render_recommendation_playground()
with tabs[8]:
    _render_beam_visualizer()
with tabs[9]:
    _render_leakage_schema()
with tabs[10]:
    _render_artifact_browser()


# --------------------------------------------------------------------------- #
# Auto-refresh (kept at the bottom so it doesn't trip half-rendered widgets)
# --------------------------------------------------------------------------- #

if auto_refresh:
    time.sleep(int(refresh_interval))
    rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if rerun is not None:
        rerun()
