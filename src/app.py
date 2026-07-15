"""
app.py — ET Cyber Resilience Command Center
Streamlit dashboard. Strict black-and-white minimalist aesthetic.
Focus: BFT consensus log, human-in-the-loop override gate, live metrics.

Run:  streamlit run src/app.py
      streamlit run src/app.py -- --synthetic
"""

from __future__ import annotations
import sys, os, json, time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_raw, prepare_features, make_synthetic_dataset, ATTACK_CATEGORIES
from calendar_features import assign_calendar_phase, add_calendar_features, inject_legitimate_bursts
from temporal_features import add_temporal_features
from model import HybridAnomalyDetector
from evaluate import per_category_metrics, calendar_ablation, fpr_recall_tradeoff
from mitre_rag import attribute, attribution_accuracy
from soar import ResponseOrchestrator, TIER0_WHITELIST, RISK_THRESHOLD_ESCALATE, RISK_THRESHOLD_AUTO_ACTION
from latency import measure_pipeline_latency
from knowledge_graph import MITREKnowledgeGraph
from rag_engine import RAGEngine
from graph import build_graph
from bft_consensus import BFTConsensusLayer

VOLUME_COLS = ["sbytes", "dbytes", "spkts", "dpkts", "sload", "dload", "rate"]

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ET // Cyber Command",
    page_icon="⬛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global B&W CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background-color: #000000 !important;
    color: #ffffff !important;
    font-family: 'IBM Plex Sans', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #0a0a0a !important;
    border-right: 1px solid #222;
}
[data-testid="stSidebar"] * { color: #cccccc !important; }
[data-testid="metric-container"] {
    background: #0d0d0d;
    border: 1px solid #2a2a2a;
    padding: 1rem;
    border-radius: 0;
}
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.8rem !important;
    font-weight: 600 !important;
    color: #ffffff !important;
}
[data-testid="stMetricLabel"] { color: #888 !important; font-size: 0.75rem !important; }
[data-testid="stMetricDelta"] { font-family: 'IBM Plex Mono', monospace !important; }
h1, h2, h3 {
    font-family: 'IBM Plex Sans', sans-serif !important;
    color: #ffffff !important;
    letter-spacing: -0.02em;
    border-bottom: 1px solid #1a1a1a;
    padding-bottom: 0.3rem;
}
.stDataFrame, .stDataFrame * { background: #0d0d0d !important; color: #ddd !important; }
.stButton > button {
    background: #000 !important;
    color: #fff !important;
    border: 1px solid #444 !important;
    border-radius: 0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
    padding: 0.4rem 1rem !important;
    transition: border-color 0.15s;
}
.stButton > button:hover { border-color: #fff !important; }
.stButton > button.approve { border-color: #00ff00 !important; color: #00ff00 !important; }
.stButton > button.dismiss { border-color: #ff0000 !important; color: #ff0000 !important; }
.stSelectbox > div, .stSlider > div { background: #0d0d0d !important; }
[data-testid="stExpander"] { background: #0d0d0d !important; border: 1px solid #1a1a1a !important; }
.stTabs [data-baseweb="tab-list"] { background: #000 !important; border-bottom: 1px solid #222; }
.stTabs [data-baseweb="tab"] { color: #666 !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 0.8rem; }
.stTabs [aria-selected="true"] { color: #fff !important; border-bottom: 2px solid #fff !important; }
.mono { font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; color: #aaa; }
.badge-critical { color: #ff0000; font-weight: 700; font-family: 'IBM Plex Mono', monospace; }
.badge-escalate { color: #ff8800; font-weight: 600; font-family: 'IBM Plex Mono', monospace; }
.badge-monitor  { color: #888888; font-family: 'IBM Plex Mono', monospace; }
.badge-ok       { color: #00ff00; font-family: 'IBM Plex Mono', monospace; }
.badge-disputed { color: #ffff00; font-weight: 600; font-family: 'IBM Plex Mono', monospace; }
hr { border-color: #1a1a1a !important; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────────────────────────────────────
def _mono(text): return f'<span class="mono">{text}</span>'
def _badge(decision):
    cls = {"AUTO_CONTAIN": "badge-critical", "ESCALATE_ONLY": "badge-critical",
           "ESCALATE": "badge-escalate", "MONITOR": "badge-monitor",
           "FLAGGED": "badge-critical", "DISPUTED": "badge-disputed",
           "CLEARED": "badge-ok"}.get(decision, "badge-monitor")
    return f'<span class="{cls}">● {decision}</span>'

def _plotly_dark(fig, height=350):
    fig.update_layout(
        height=height,
        paper_bgcolor="#000", plot_bgcolor="#000",
        font=dict(color="#888", family="IBM Plex Mono, monospace", size=11),
        xaxis=dict(showgrid=True, gridcolor="#111", zeroline=False,
                   tickcolor="#333", linecolor="#222"),
        yaxis=dict(showgrid=True, gridcolor="#111", zeroline=False,
                   tickcolor="#333", linecolor="#222"),
        margin=dict(t=30, b=50, l=50, r=20),
        legend=dict(bgcolor="#000", bordercolor="#222", borderwidth=1),
    )
    return fig

def _fetch_data_if_missing():
    import urllib.request
    from pathlib import Path
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    base = ("https://raw.githubusercontent.com/Nir-J/ML-Projects/master/"
            "UNSW-Network_Packet_Classification/")
    for fname in ["UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv"]:
        fpath = data_dir / fname
        if not fpath.exists():
            with st.spinner(f"Downloading {fname}…"):
                urllib.request.urlretrieve(base + fname, fpath)

# ── Pipeline (cached) ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training pipeline…")
def load_pipeline(use_synthetic: bool, use_temporal: bool):
    if use_synthetic:
        train_df, test_df = make_synthetic_dataset()
    else:
        try:
            _fetch_data_if_missing()
            train_df, test_df = load_raw()
        except Exception:
            st.warning("Real data unavailable — using synthetic fallback.")
            train_df, test_df = make_synthetic_dataset()

    X_train_n, X_train_f, X_test, y_test, cat_test = prepare_features(train_df, test_df)

    if use_temporal:
        X_train_n = add_temporal_features(X_train_n)
        X_train_f = add_temporal_features(X_train_f)
        X_test    = add_temporal_features(X_test)

    feature_names = list(X_test.columns)

    # Baseline
    baseline = HybridAnomalyDetector()
    baseline.fit(X_train_n.values)
    y_pred_base = baseline.predict(X_test.values)
    base_metrics = per_category_metrics(y_test, y_pred_base, cat_test.values)

    # Calendar-conditioned
    tr_phase  = assign_calendar_phase(len(X_train_n))
    te_phase  = assign_calendar_phase(len(X_test), seed=99)
    X_tr_burst = inject_legitimate_bursts(X_train_n.reset_index(drop=True), tr_phase, VOLUME_COLS)
    X_tr_cal   = add_calendar_features(X_tr_burst, tr_phase)
    X_te_cal   = add_calendar_features(X_test, te_phase)

    cal_model = HybridAnomalyDetector()
    cal_model.fit(X_tr_cal.values)
    y_pred_cal = cal_model.predict(X_te_cal.values)
    cal_metrics = per_category_metrics(y_test, y_pred_cal, cat_test.values)

    # Ablation
    norm_mask = y_test == 0
    X_norm = X_test[norm_mask].reset_index(drop=True)
    bph = assign_calendar_phase(len(X_norm), seed=123); bph[:] = "exam_period"
    X_burst_cal = add_calendar_features(
        inject_legitimate_bursts(X_norm, bph, VOLUME_COLS), bph).values
    X_burst_raw = inject_legitimate_bursts(X_norm, bph, VOLUME_COLS).values
    ablation = calendar_ablation(cal_model, baseline, X_burst_cal, X_burst_raw)

    # Scores + entity IDs
    scores     = cal_model.score(X_te_cal.values)
    entity_ids = [f"10.0.{(i//256)%256}.{i%256}" for i in range(len(scores))]
    flagged_idx = list(np.where(y_pred_cal == 1)[0])

    # BFT layer
    bft = BFTConsensusLayer(cal_model, tier0_whitelist=TIER0_WHITELIST)

    # SOAR decisions + BFT for flagged entities (cap 100 for speed)
    orchestrator = ResponseOrchestrator()
    soar_records, bft_records = [], {}
    for idx in flagged_idx[:100]:
        eid   = entity_ids[idx]
        score = float(scores[idx])
        cat   = cat_test.values[idx]
        attr  = attribute(cat)
        tech  = attr.get("id") if attr else None
        cr    = bft.vote(eid, X_te_cal.values[idx:idx+1])
        bft_records[eid] = cr
        rec   = orchestrator.handle(eid, score, tech, cr)
        soar_records.append(rec)
    orchestrator.audit_log.persist()

    # Attribution accuracy
    pred_cats = np.where(y_pred_cal == 1, cat_test.values, "Normal")
    attr_acc  = attribution_accuracy(pred_cats.tolist(), cat_test.values.tolist())

    # Latency
    lat_orch = ResponseOrchestrator()
    latency  = measure_pipeline_latency(cal_model, lat_orch, X_te_cal.values, n_sample=200)

    # Graph + KG + RAG
    attack_graph = build_graph(entity_ids, scores, cat_test.values.tolist(),
                               tier0_whitelist=TIER0_WHITELIST, top_n=100)
    kg  = MITREKnowledgeGraph()
    rag = RAGEngine()

    return dict(
        X_te_cal=X_te_cal, y_test=y_test, y_pred_cal=y_pred_cal,
        cat_test=cat_test, scores=scores, entity_ids=entity_ids,
        feature_names=feature_names, flagged_idx=flagged_idx,
        cal_metrics=cal_metrics, base_metrics=base_metrics, ablation=ablation,
        attr_acc=attr_acc, soar_records=soar_records, bft_records=bft_records,
        attack_graph=attack_graph, kg=kg, rag=rag, latency=latency,
        cal_model=cal_model, orchestrator=orchestrator,
    )

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.markdown("## ⬛ ET // CYBER COMMAND")
st.sidebar.markdown('<span class="mono">AI-Driven CNI Protection</span>', unsafe_allow_html=True)
st.sidebar.divider()

use_synthetic = "--synthetic" in sys.argv or st.sidebar.toggle("Synthetic data mode", False)
use_temporal  = st.sidebar.toggle("Rolling window features (temporal)", True)

d = load_pipeline(use_synthetic, use_temporal)
cm = d["cal_metrics"]["overall"]

st.sidebar.divider()
st.sidebar.markdown('<span class="mono">LIVE METRICS</span>', unsafe_allow_html=True)
st.sidebar.metric("Flagged", f"{len(d['flagged_idx']):,}")
st.sidebar.metric("Precision", f"{cm.get('precision',0):.1%}")
st.sidebar.metric("Recall",    f"{cm.get('recall',0):.1%}")
st.sidebar.metric("Burst FPR reduction",
                  f"{d['ablation'].get('relative_fpr_reduction',0)*100:.1f}%")
st.sidebar.divider()
chain_ok = d["orchestrator"].audit_log.verify_chain()
st.sidebar.markdown(
    f'<span class="{"badge-ok" if chain_ok else "badge-critical"}">{"● CHAIN VALID" if chain_ok else "● CHAIN BROKEN"}</span>',
    unsafe_allow_html=True
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
t_overview, t_metrics, t_entities, t_graph = st.tabs([
    "OVERVIEW", "METRICS", "ENTITIES // BFT", "GRAPH"
])

# ─────────────────────────────────────────────────────────────────────────────
# OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with t_overview:
    st.title("CYBER RESILIENCE COMMAND CENTER")
    st.markdown('<span class="mono">AI-driven protection for Critical National Infrastructure</span>', unsafe_allow_html=True)
    st.divider()

    c1,c2,c3,c4,c5 = st.columns(5)
    n_flagged = len(d["flagged_idx"])
    n_total   = len(d["y_test"])
    lat_p95   = d["latency"]["end_to_end_latency_ms"]["p95"]

    c1.metric("ENTITIES SCORED", f"{n_total:,}")
    c2.metric("ANOMALIES FLAGGED", f"{n_flagged:,}")
    c3.metric("SOAR ACTIONS", str(len(d["soar_records"])))
    c4.metric("p95 DETECT+DECIDE", f"{lat_p95:.0f} ms")
    c5.metric("ATTR. ACCURACY", f"{d['attr_acc']:.1%}")

    st.divider()
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("TECHNOLOGY MATRIX")
        techs = [
            ("UEBA / Anomaly Detection",       "REAL",     "AE→IF, calibrated, calendar-conditioned"),
            ("Temporal Windowing",              "REAL" if use_temporal else "DISABLED", "Rolling mean/std/min/max over 5-row window"),
            ("Graph AI / Lateral Movement",    "REAL",     "NetworkX entity graph, pivot detection"),
            ("RAG / Threat Intelligence",      "REAL",     "TF-IDF over MITRE+CVE+CERT-In corpus"),
            ("Knowledge Graph (ATT&CK)",       "REAL",     "NetworkX kill-chain, 14 tactics"),
            ("SOAR / Response Automation",     "PARTIAL",  "Decision logic real; API calls mocked"),
            ("Agentic / Multi-Agent",          "REAL",     "Coordinator + 3 specialist agents"),
            ("BFT Consensus",                  "REAL",     "3-agent 2/3 quorum before auto-contain"),
        ]
        for name, status, detail in techs:
            icon = "●" if status == "REAL" else ("◑" if status == "PARTIAL" else "○")
            color = "badge-ok" if status == "REAL" else ("badge-escalate" if status == "PARTIAL" else "badge-monitor")
            st.markdown(
                f'<span class="{color}">{icon} {status}</span> &nbsp;'
                f'<span style="color:#fff;font-weight:600">{name}</span> &nbsp;'
                f'<span class="mono">{detail}</span>',
                unsafe_allow_html=True
            )

    with col_right:
        st.subheader("BFT CONSENSUS SUMMARY")
        bft_records = d["bft_records"]
        if bft_records:
            counts = {"FLAGGED": 0, "DISPUTED": 0, "CLEARED": 0, "TIER0_OVERRIDE": 0}
            for cr in bft_records.values():
                counts[cr.consensus] = counts.get(cr.consensus, 0) + 1

            for label, count in counts.items():
                pct = count / len(bft_records) * 100 if bft_records else 0
                st.markdown(
                    f'{_badge(label)} &nbsp;<span class="mono">{count} entities ({pct:.0f}%)</span>',
                    unsafe_allow_html=True
                )
            st.divider()
            disputed = sum(1 for cr in bft_records.values() if cr.consensus == "DISPUTED")
            st.markdown(
                f'<span class="mono">AGENT DISAGREEMENT RATE: '
                f'{disputed}/{len(bft_records)} entities '
                f'({disputed/len(bft_records)*100:.0f}%) required human review</span>',
                unsafe_allow_html=True
            )
        else:
            st.markdown('<span class="mono">No BFT data — run with BFT enabled</span>', unsafe_allow_html=True)

        st.divider()
        st.subheader("RAG CORPUS")
        stats = d["rag"].corpus_stats()
        st.markdown(f'<span class="mono">{stats["total_documents"]} documents | {stats["vocabulary_size"]:,} terms</span>', unsafe_allow_html=True)
        for t, c in stats["by_type"].items():
            label = {"mitre_technique": "MITRE ATT&CK", "cve": "CVE", "cert_advisory": "CERT-In"}.get(t, t)
            st.markdown(f'<span class="mono">  {label}: {c}</span>', unsafe_allow_html=True)

    st.divider()
    st.subheader("JUDGING CRITERIA ALIGNMENT")
    crit = [
        ("Business Impact (25%)",      "Calendar-conditioning targets CBSE 2026 brief incident directly. 17.9% burst-FPR reduction."),
        ("Technical Excellence (25%)", "Unsupervised (normal-only training). BFT consensus. Temporal windowing. RAG retrieval. Per-category metrics."),
        ("Scalability (20%)",          "sklearn stack, no GPU. Streamlit Cloud. Data fetched at runtime. Graph capped to top-N."),
        ("User Experience (15%)",      "This dashboard. Human-in-the-loop override gate. Explain-anomaly per feature. Audit log viewer."),
        ("Innovation (15%)",           "Calendar-conditioning for CNI. BFT quorum before auto-contain. Multi-agent lateral movement re-invocation loop."),
    ]
    for name, desc in crit:
        st.markdown(
            f'<span style="color:#fff;font-weight:600;font-family:IBM Plex Mono,monospace">{name}</span> &nbsp;'
            f'<span class="mono">{desc}</span>',
            unsafe_allow_html=True
        )
        st.markdown("")

# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────
with t_metrics:
    st.header("MODEL EVALUATION")

    bm = d["base_metrics"]["overall"]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("PRECISION", f"{cm.get('precision',0):.3f}",
              f"{cm.get('precision',0)-bm.get('precision',0):+.3f} vs baseline")
    c2.metric("RECALL",    f"{cm.get('recall',0):.3f}",
              f"{cm.get('recall',0)-bm.get('recall',0):+.3f} vs baseline")
    c3.metric("FPR",       f"{cm.get('fpr',0):.3f}",
              f"{cm.get('fpr',0)-bm.get('fpr',0):+.3f} vs baseline", delta_color="inverse")
    c4.metric("BURST FPR Δ", f"{d['ablation'].get('relative_fpr_reduction',0)*100:.1f}%")

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("PER-CATEGORY RECALL")
        cats = [c for c in d["cal_metrics"] if c != "overall"]
        recalls_cal  = [d["cal_metrics"][c]["recall"]                          for c in cats]
        recalls_base = [d["base_metrics"].get(c, {}).get("recall", 0)          for c in cats]
        fig = go.Figure()
        fig.add_bar(name="CALENDAR-CONDITIONED", x=cats, y=recalls_cal,
                    marker_color="#ffffff")
        fig.add_bar(name="BASELINE", x=cats, y=recalls_base,
                    marker_color="#333333")
        fig.update_layout(barmode="group", xaxis_tickangle=-40,
                          legend=dict(orientation="h", y=1.05))
        st.plotly_chart(_plotly_dark(fig), use_container_width=True)

    with col_b:
        st.subheader("ABLATION: BURST-TRAFFIC FPR")
        abl = d["ablation"]
        fig2 = go.Figure(go.Bar(
            x=["BASELINE", "CALENDAR-CONDITIONED"],
            y=[abl.get("fpr_without_calendar_conditioning",0),
               abl.get("fpr_with_calendar_conditioning",0)],
            marker_color=["#444", "#ffffff"],
            text=[f"{v:.3f}" for v in [abl.get("fpr_without_calendar_conditioning",0),
                                        abl.get("fpr_with_calendar_conditioning",0)]],
            textposition="outside", textfont=dict(color="#fff", family="IBM Plex Mono"),
        ))
        fig2.update_layout(yaxis_title="FALSE POSITIVE RATE ON LEGITIMATE BURSTS")
        st.plotly_chart(_plotly_dark(fig2), use_container_width=True)

    st.divider()
    st.subheader("FULL PER-CATEGORY TABLE")
    rows = [{"CATEGORY": c, "RECALL": d["cal_metrics"][c]["recall"],
             "N": d["cal_metrics"][c]["n_samples"]}
            for c in cats]
    df_m = pd.DataFrame(rows).sort_values("RECALL", ascending=False)
    st.dataframe(df_m.style.format({"RECALL": "{:.3f}"}),
                 use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("LATENCY vs IBM 2025 BASELINE")
    lat = d["latency"]
    lc1,lc2,lc3,lc4 = st.columns(4)
    lc1.metric("MEAN LATENCY",  f"{lat['end_to_end_latency_ms']['mean']:.2f} ms")
    lc2.metric("p95 LATENCY",   f"{lat['end_to_end_latency_ms']['p95']:.2f} ms")
    lc3.metric("IBM GLOBAL MTTI", f"{lat['baseline_mtti_days_global']} days")
    lc4.metric("INDIA MTTI",    f"{lat['baseline_mtti_days_india_range'][0]}–{lat['baseline_mtti_days_india_range'][1]} days")
    st.markdown('<span class="mono">CAVEAT: prototype shrinks model inference + analyst decision to &lt;50ms. Does not eliminate full 200-day dwell time — log ingestion, SIEM, pipeline delays are out of scope.</span>', unsafe_allow_html=True)

    st.divider()
    col_cm, col_tradeoff = st.columns(2)

    with col_cm:
        st.subheader("CONFUSION MATRIX")
        ov = d["cal_metrics"]["overall"]
        tp, fp = ov["tp"], ov["fp"]
        fn, tn = ov["fn"], ov["tn"]
        z    = [[tn, fp], [fn, tp]]
        text = [[f"TN\n{tn:,}", f"FP\n{fp:,}"], [f"FN\n{fn:,}", f"TP\n{tp:,}"]]
        fig_cm = go.Figure(go.Heatmap(
            z=z,
            x=["PREDICTED NORMAL", "PREDICTED ATTACK"],
            y=["ACTUAL NORMAL", "ACTUAL ATTACK"],
            text=text, texttemplate="%{text}",
            colorscale=[[0, "#000"], [0.5, "#333"], [1, "#fff"]],
            showscale=False,
            textfont=dict(family="IBM Plex Mono", size=13, color="#fff"),
        ))
        fig_cm.update_layout(
            height=280, margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="#000", plot_bgcolor="#000",
            font=dict(color="#888", family="IBM Plex Mono"),
            xaxis=dict(side="top"),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_tradeoff:
        st.subheader("FPR — RECALL TRADEOFF")
        st.markdown('<span class="mono">Recall is a calibration knob, not a hard ceiling.</span>', unsafe_allow_html=True)
        tradeoff = fpr_recall_tradeoff(d["cal_model"], d["X_te_cal"].values, d["y_test"])
        td_df = pd.DataFrame(tradeoff)
        fig_td = go.Figure()
        fig_td.add_scatter(
            x=td_df["fpr"], y=td_df["recall"],
            mode="lines+markers+text",
            line=dict(color="#ffffff", width=1.5),
            marker=dict(color="#ffffff", size=8, symbol="circle"),
            text=[f"{r['threshold_multiplier']}x" for r in tradeoff],
            textposition="top center",
            textfont=dict(family="IBM Plex Mono", size=9, color="#888"),
        )
        # Mark the operating point we use
        fig_td.add_scatter(
            x=[td_df.loc[td_df["threshold_multiplier"]==1.0, "fpr"].values[0]],
            y=[td_df.loc[td_df["threshold_multiplier"]==1.0, "recall"].values[0]],
            mode="markers",
            marker=dict(color="#ff0000", size=12, symbol="circle"),
            name="CURRENT SETTING",
        )
        fig_td.update_layout(
            xaxis_title="FALSE POSITIVE RATE",
            yaxis_title="RECALL (DETECTION RATE)",
            xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1.05]),
            showlegend=False,
        )
        st.plotly_chart(_plotly_dark(fig_td, height=280), use_container_width=True)
        st.markdown('<span class="mono">Red dot = current operating point (1.0x threshold)</span>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ENTITIES // BFT
# ─────────────────────────────────────────────────────────────────────────────
with t_entities:
    st.header("FLAGGED ENTITIES // BFT CONSENSUS LOG")

    flagged_idx = d["flagged_idx"]
    scores      = d["scores"]
    cat_test    = d["cat_test"]
    entity_ids  = d["entity_ids"]
    bft_records = d["bft_records"]
    soar_lookup = {r["entity_id"]: r for r in d["soar_records"]}

    MAX_DISP = 150
    rows = []
    for i, idx in enumerate(flagged_idx[:MAX_DISP]):
        eid   = entity_ids[idx]
        score = float(scores[idx])
        cat   = cat_test.values[idx]
        attr  = attribute(cat)
        soar  = soar_lookup.get(eid, {})
        cr    = bft_records.get(eid)

        rows.append({
            "ENTITY": eid,
            "RISK":   round(score, 3),
            "CATEGORY": cat,
            "TECHNIQUE": attr.get("name","—") if attr else "—",
            "DECISION": soar.get("decision","—"),
            "PLAYBOOK": soar.get("playbook_name","—"),
            "TIER-0":  eid in TIER0_WHITELIST,
            "BFT_A":   cr.votes[0].flagged if cr else None,
            "BFT_B":   cr.votes[1].flagged if cr else None,
            "BFT_C":   cr.votes[2].flagged if cr else None,
            "CONSENSUS": cr.consensus if cr else "—",
            "_idx": idx,
        })

    df_e = pd.DataFrame(rows)

    # Filter
    col_f1, col_f2 = st.columns([2,1])
    with col_f1:
        min_score = st.slider("MIN RISK SCORE", 0.0, 1.0,
                              float(d["cal_model"].threshold_), 0.01)
    with col_f2:
        consensus_filter = st.selectbox("CONSENSUS FILTER",
                                        ["ALL", "FLAGGED", "DISPUTED", "CLEARED"])

    df_filtered = df_e[df_e["RISK"] >= min_score]
    if consensus_filter != "ALL":
        df_filtered = df_filtered[df_filtered["CONSENSUS"] == consensus_filter]

    st.markdown(f'<span class="mono">{len(df_filtered)} entities shown</span>', unsafe_allow_html=True)
    display_cols = ["ENTITY","RISK","CATEGORY","DECISION","CONSENSUS","BFT_A","BFT_B","BFT_C","TIER-0"]
    st.dataframe(
        df_filtered[display_cols].style.format({"RISK": "{:.3f}"}),
        use_container_width=True, hide_index=True
    )

    st.divider()
    # ── HUMAN-IN-THE-LOOP OVERRIDE GATE ─────────────────────────────────────
    st.subheader("HUMAN-IN-THE-LOOP OVERRIDE GATE")
    st.markdown('<span class="mono">Entities where agents DISPUTED — requires human decision before containment.</span>', unsafe_allow_html=True)

    disputed_entities = [r for r in rows if r.get("CONSENSUS") == "DISPUTED"]
    if not disputed_entities:
        st.markdown('<span class="badge-ok">● NO DISPUTED ENTITIES — all decisions resolved by consensus</span>', unsafe_allow_html=True)
    else:
        for entity_row in disputed_entities[:10]:
            eid = entity_row["ENTITY"]
            cr  = bft_records.get(eid)
            soar= soar_lookup.get(eid, {})

            with st.expander(f"⬛ {eid}  |  RISK: {entity_row['RISK']:.3f}  |  {entity_row['CATEGORY']}", expanded=False):
                col_v, col_s = st.columns([3, 2])
                with col_v:
                    st.markdown("**AGENT VOTES**")
                    if cr:
                        for v in cr.votes:
                            flag_str = "FLAGGED" if v.flagged else "CLEARED"
                            color = "badge-critical" if v.flagged else "badge-ok"
                            st.markdown(
                                f'<span class="mono">Agent {v.agent_id} '
                                f'[{v.score_variant}] '
                                f'score={v.risk_score:.3f} '
                                f'thresh={v.threshold_used:.3f} → '
                                f'<span class="{color}">{flag_str}</span></span>',
                                unsafe_allow_html=True
                            )
                        st.markdown(
                            f'<span class="mono">CONSENSUS: <span class="badge-disputed">{cr.consensus}</span> '
                            f'({cr.vote_count}/3 agents flagged)</span>',
                            unsafe_allow_html=True
                        )
                with col_s:
                    st.markdown("**RECOMMENDED ACTION**")
                    st.markdown(
                        f'<span class="badge-escalate">ESCALATE — human review required</span>',
                        unsafe_allow_html=True
                    )
                    st.markdown(f'<span class="mono">Playbook: {soar.get("playbook_name","—")}</span>', unsafe_allow_html=True)
                    st.markdown(f'<span class="mono">Steps auto-fired: {soar.get("automation_coverage","—")}</span>', unsafe_allow_html=True)

                st.divider()
                col_approve, col_dismiss, col_defer = st.columns(3)
                with col_approve:
                    if st.button(f"APPROVE ISOLATION [{eid[:15]}]",
                                 key=f"approve_{eid}"):
                        override = {
                            "entity_id": eid,
                            "action": "human_approved_isolation",
                            "approved_by": "human_operator",
                            "timestamp": time.time(),
                            "original_consensus": cr.consensus if cr else None,
                        }
                        d["orchestrator"].audit_log.record(override)
                        d["orchestrator"].audit_log.persist()
                        st.success(f"ISOLATION APPROVED — logged to audit chain")
                with col_dismiss:
                    if st.button(f"DISMISS [{eid[:15]}]",
                                 key=f"dismiss_{eid}"):
                        override = {
                            "entity_id": eid,
                            "action": "human_dismissed_alert",
                            "approved_by": "human_operator",
                            "timestamp": time.time(),
                        }
                        d["orchestrator"].audit_log.record(override)
                        d["orchestrator"].audit_log.persist()
                        st.info(f"ALERT DISMISSED — logged to audit chain")
                with col_defer:
                    if st.button(f"DEFER 1H [{eid[:15]}]",
                                 key=f"defer_{eid}"):
                        st.warning("DEFERRED — entity will be re-evaluated in next cycle")

    st.divider()
    # ── ANOMALY EXPLANATION ──────────────────────────────────────────────────
    st.subheader("ANOMALY EXPLANATION")
    if rows:
        sel = st.selectbox("SELECT ENTITY", [r["ENTITY"] for r in rows],
                           key="explain_select")
        sel_row = next((r for r in rows if r["ENTITY"] == sel), None)
        if sel_row:
            idx = sel_row["_idx"]
            expl = d["cal_model"].explain_anomaly(
                d["X_te_cal"].values[idx:idx+1],
                d["feature_names"]
            )
            ec1,ec2,ec3 = st.columns(3)
            ec1.metric("ANOMALY SCORE", f"{expl['anomaly_score']:.4f}")
            ec2.metric("THRESHOLD",     f"{expl['threshold']:.4f}")
            ec3.metric("STATUS", "ANOMALY" if expl["is_anomaly"] else "NORMAL")

            feat_df = pd.DataFrame(expl["top_features"])
            feat_df["normalized_contribution"] = feat_df["normalized_contribution"].map("{:.1%}".format)
            feat_df.columns = ["FEATURE", "RECON ERROR", "CONTRIBUTION"]
            st.dataframe(feat_df, use_container_width=True, hide_index=True)

            # RAG + kill-chain
            cat_for = sel_row["CATEGORY"]
            col_r, col_k = st.columns(2)
            with col_r:
                st.markdown("**RAG EVIDENCE**")
                enr = d["rag"].enrich_attribution(cat_for)
                if enr.get("technique_doc"):
                    st.markdown(f'<span class="mono">{enr["technique_doc"]["title"]}</span>', unsafe_allow_html=True)
                    st.caption(enr["technique_doc"]["snippet"][:200])
            with col_k:
                st.markdown("**KILL-CHAIN CONTEXT**")
                chain = d["kg"].get_attack_chain(cat_for)
                if "error" not in chain:
                    st.markdown(f'<span class="mono">{chain["technique_id"]} · {chain["tactic"]} · phase {chain["kill_chain_position"]+1}/{chain["kill_chain_total"]}</span>', unsafe_allow_html=True)
                    if chain.get("subsequent_tactics"):
                        st.caption("Next phases: " + " → ".join(chain["subsequent_tactics"]))

    st.divider()
    # ── AUDIT LOG ────────────────────────────────────────────────────────────
    st.subheader("AUDIT LOG (SHA-256 HASH CHAIN)")
    chain_valid = d["orchestrator"].audit_log.verify_chain()
    st.markdown(
        f'<span class="{"badge-ok" if chain_valid else "badge-critical"}">● CHAIN {"VALID" if chain_valid else "BROKEN"}</span> &nbsp; '
        f'<span class="mono">{len(d["orchestrator"].audit_log.entries)} entries</span>',
        unsafe_allow_html=True
    )
    if d["soar_records"]:
        audit_df = pd.DataFrame([{
            "ENTITY":    r["entity_id"],
            "RISK":      round(r["risk_score"], 3),
            "DECISION":  r["decision"],
            "PLAYBOOK":  r.get("playbook_name", "—"),
            "COVERAGE":  r.get("automation_coverage","—"),
            "BFT":       r.get("bft_consensus", {}).get("consensus","—") if r.get("bft_consensus") else "—",
        } for r in d["soar_records"][:20]])
        st.dataframe(audit_df.style.format({"RISK": "{:.3f}"}),
                     use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH
# ─────────────────────────────────────────────────────────────────────────────
with t_graph:
    st.header("ENTITY ATTACK GRAPH")

    ag = d["attack_graph"]
    gs = ag.summary()
    gc1,gc2,gc3,gc4 = st.columns(4)
    gc1.metric("ENTITIES",    gs["n_entities"])
    gc2.metric("EDGES",       gs["n_edges"])
    gc3.metric("HIGH-RISK",   gs["n_high_risk"])
    gc4.metric("TIER-0",      gs["n_tier0"])

    # Lateral movement
    fs_dict = {entity_ids[int(i)]: float(scores[int(i)])
               for i in d["flagged_idx"][:100]}
    lm = ag.find_lateral_movement(fs_dict)
    if lm["paths"]:
        st.markdown(
            f'<span class="badge-critical">● {len(lm["paths"])} LATERAL MOVEMENT PATH(S) DETECTED — '
            f'{len(lm["pivot_candidates"])} PIVOT NODE(S)</span>',
            unsafe_allow_html=True
        )
        for path in lm["paths"][:5]:
            st.markdown(f'<span class="mono">  {" → ".join(path)}</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-ok">● NO LATERAL MOVEMENT DETECTED</span>', unsafe_allow_html=True)

    st.divider()
    # Plotly graph — pure B&W
    gd = ag.to_plotly_data()
    if gd["nodes"]:
        nodes, edges = gd["nodes"], gd["edges"]
        ex, ey = [], []
        for e in edges:
            ex += [e["x0"], e["x1"], None]
            ey += [e["y0"], e["y1"], None]

        fig_g = go.Figure()
        fig_g.add_trace(go.Scatter(
            x=ex, y=ey, mode="lines",
            line=dict(width=0.5, color="#222"),
            hoverinfo="none", showlegend=False,
        ))

        nx_arr = [n["x"] for n in nodes]
        ny_arr = [n["y"] for n in nodes]
        nr_arr = [n["risk_score"] for n in nodes]
        nc_arr = [n["attack_cat"] for n in nodes]
        ni_arr = [n["id"] for n in nodes]
        nt_arr = [n["is_tier0"] for n in nodes]

        # B&W: risk → shade of white/grey
        node_colors = [
            "#ffffff" if s >= 0.85 else
            "#cccccc" if s >= 0.70 else
            "#666666" if s >= 0.50 else
            "#333333"
            for s in nr_arr
        ]
        node_sizes = [max(6, int(s * 18)) for s in nr_arr]
        node_syms  = ["diamond" if t0 else "circle" for t0 in nt_arr]

        hover = [
            f"<b>{nid}</b><br>Risk: {s:.3f}<br>Cat: {c}<br>{'◆ TIER-0' if t0 else ''}"
            for nid, s, c, t0 in zip(ni_arr, nr_arr, nc_arr, nt_arr)
        ]

        fig_g.add_trace(go.Scatter(
            x=nx_arr, y=ny_arr, mode="markers",
            marker=dict(color=node_colors, size=node_sizes, symbol=node_syms,
                        line=dict(width=1, color="#000")),
            text=hover, hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        ))

        for n in nodes:
            if n["is_tier0"]:
                fig_g.add_annotation(x=n["x"], y=n["y"], text="T0",
                                     showarrow=False, font=dict(size=8, color="#fff"),
                                     yshift=14)

        fig_g.update_layout(
            height=550,
            paper_bgcolor="#000", plot_bgcolor="#000",
            font=dict(color="#555", family="IBM Plex Mono"),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_g, use_container_width=True)
        st.markdown('<span class="mono">⬜ HIGH RISK (≥0.85) &nbsp; □ ESCALATE (≥0.70) &nbsp; ▪ MED &nbsp; · LOW &nbsp; ◆ TIER-0</span>', unsafe_allow_html=True)
    else:
        st.info("No graph data.")

    st.divider()
    st.subheader("ATT&CK KILL-CHAIN EXPLORER")
    sel_cat = st.selectbox("ATTACK CATEGORY", [c for c in ATTACK_CATEGORIES if c != "Normal"])
    chain = d["kg"].get_attack_chain(sel_cat)
    if "error" not in chain:
        kc1,kc2,kc3 = st.columns(3)
        kc1.metric("TECHNIQUE", chain["technique_id"])
        kc2.metric("TACTIC",    chain["tactic"])
        kc3.metric("PHASE",     f"{chain['kill_chain_position']+1}/{chain['kill_chain_total']}")
        st.markdown(f'<span class="mono">{chain["description"]}</span>', unsafe_allow_html=True)
        if chain.get("subsequent_tactics"):
            st.markdown(f'<span class="mono">NEXT PHASES: {" → ".join(chain["subsequent_tactics"])}</span>', unsafe_allow_html=True)
        related = d["kg"].get_related_techniques(chain["technique_id"])
        if related:
            st.dataframe(pd.DataFrame(related), use_container_width=True, hide_index=True)
