"""
Traffic Flow Control - Interactive Dashboard
METR-LA Dataset | 207 Sensors | Los Angeles Freeway Network

Reads only from pre-computed result files in results/.
Does NOT import or modify any src/ module.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Paths
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
BASELINES_DIR = RESULTS / "baselines"
GAP_DIR = RESULTS / "gap"

# Page config
st.set_page_config(
    page_title="Traffic Flow Control - METR-LA Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0d1117 0%, #0f1923 50%, #0d1117 100%); }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #131e2b 0%, #0d1117 100%);
    border-right: 1px solid rgba(42,120,214,0.2);
}
[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(42,120,214,0.12) 0%, rgba(42,120,214,0.06) 100%);
    border: 1px solid rgba(42,120,214,0.3);
    border-radius: 12px;
    padding: 18px 20px;
    backdrop-filter: blur(10px);
    transition: border-color 0.3s ease, transform 0.2s ease;
}
[data-testid="metric-container"]:hover { border-color: rgba(42,120,214,0.6); transform: translateY(-2px); }
[data-testid="metric-container"] label {
    color: #8b9ab5 !important; font-size: 0.78rem !important;
    font-weight: 500 !important; letter-spacing: 0.06em !important; text-transform: uppercase;
}
[data-testid="stMetricValue"] { color: #e8f4fd !important; font-weight: 700 !important; font-size: 1.8rem !important; }
[data-testid="stMetricDelta"] { font-size: 0.78rem !important; }
.section-header {
    background: linear-gradient(90deg, rgba(42,120,214,0.15) 0%, transparent 100%);
    border-left: 3px solid #2a78d6;
    padding: 10px 16px;
    border-radius: 0 8px 8px 0;
    margin: 24px 0 16px 0;
    color: #e8f4fd;
    font-size: 1.05rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.insight-card {
    background: linear-gradient(135deg, rgba(27,175,122,0.1) 0%, rgba(27,175,122,0.05) 100%);
    border: 1px solid rgba(27,175,122,0.3);
    border-radius: 10px;
    padding: 14px 18px;
    margin: 8px 0;
    color: #d0f5e8;
    font-size: 0.88rem;
    line-height: 1.6;
}
.insight-card-orange {
    background: linear-gradient(135deg, rgba(235,104,52,0.1) 0%, rgba(235,104,52,0.05) 100%);
    border: 1px solid rgba(235,104,52,0.3);
    border-radius: 10px;
    padding: 14px 18px;
    margin: 8px 0;
    color: #fde8d8;
    font-size: 0.88rem;
    line-height: 1.6;
}
.insight-card-blue {
    background: linear-gradient(135deg, rgba(42,120,214,0.1) 0%, rgba(42,120,214,0.05) 100%);
    border: 1px solid rgba(42,120,214,0.3);
    border-radius: 10px;
    padding: 14px 18px;
    margin: 8px 0;
    color: #d8eafd;
    font-size: 0.88rem;
    line-height: 1.6;
}
.hero-banner {
    background: linear-gradient(135deg, rgba(42,120,214,0.15) 0%, rgba(27,175,122,0.08) 50%, rgba(235,104,52,0.05) 100%);
    border: 1px solid rgba(42,120,214,0.25);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
}
.hero-title { font-size: 1.6rem; font-weight: 700; color: #e8f4fd; margin: 0; }
.hero-sub { font-size: 0.88rem; color: #8b9ab5; margin: 4px 0 0 0; }
.footer {
    text-align: center; color: #4a5568; font-size: 0.78rem;
    padding: 24px 0 8px 0; border-top: 1px solid rgba(255,255,255,0.05); margin-top: 40px;
}
.stTabs [data-baseweb="tab-list"] { background: rgba(13,17,23,0.5); border-radius: 10px; padding: 4px; gap: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 8px; color: #8b9ab5; font-weight: 500; font-size: 0.88rem; padding: 8px 18px; transition: all 0.2s; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, #2a78d6 0%, #1a5cb0 100%) !important; color: #ffffff !important; font-weight: 600 !important; }
</style>
""", unsafe_allow_html=True)

# Colour palette (mirrors src/viz/style.py)
BLUE    = "#2a78d6"
ORANGE  = "#eb6834"
AQUA    = "#1baf7a"
YELLOW  = "#eda100"
MUTED   = "#898781"
BLUE_L  = "#b7d3f6"
SURFACE = "#0d1117"
INK     = "#e8f4fd"
INK_2   = "#8b9ab5"

def apply_plotly_layout(fig: go.Figure, **kwargs) -> go.Figure:
    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,17,23,0.4)",
        font=dict(family="Inter", color=INK, size=11),
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", color=INK_2, linecolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", color=INK_2, linecolor="rgba(255,255,255,0.1)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=INK_2, size=10)),
        hoverlabel=dict(bgcolor="#1a2535", font_color=INK, bordercolor=BLUE),
    )
    for k, v in kwargs.items():
        if k in ("xaxis", "yaxis", "margin", "legend") and isinstance(v, dict) and isinstance(layout.get(k), dict):
            merged = dict(layout[k])
            merged.update(v)
            layout[k] = merged
        else:
            layout[k] = v
    fig.update_layout(**layout)
    return fig


# Data loaders (cached)
@st.cache_data
def load_baselines():
    f = RESULTS / "baselines.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame()

@st.cache_data
def load_baselines_raw():
    out = {}
    for name in ("persistence", "hist_avg", "ridge", "xgb", "rf", "lstm"):
        f = BASELINES_DIR / f"{name}.csv"
        if f.exists():
            out[name] = pd.read_csv(f, dtype={"horizon": str})
    return out

@st.cache_data
def load_peak():
    f = RESULTS / "baselines_peak.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame()

@st.cache_data
def load_gap_results():
    f = RESULTS / "gap_results.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame()

@st.cache_data
def load_onset():
    f = GAP_DIR / "onset_vs_normal.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame()

@st.cache_data
def load_residual_corr():
    f = GAP_DIR / "residual_correlation.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame()

@st.cache_data
def load_phase1():
    f = RESULTS / "phase1_findings.json"
    return json.loads(f.read_text()) if f.exists() else {}

@st.cache_data
def load_xgb_importance():
    f = BASELINES_DIR / "xgb_importance.csv"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f, index_col="feature")
    df.columns = df.columns.astype(int)
    return df

@st.cache_data
def load_phase4():
    f = RESULTS / "phase4_findings.json"
    return json.loads(f.read_text()) if f.exists() else {}


# Sidebar
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 12px 0 20px 0;'>
        <div style='font-size:2.2rem;'>🚦</div>
        <div style='color:#e8f4fd; font-weight:700; font-size:1.0rem; margin-top:6px;'>Traffic Flow Control</div>
        <div style='color:#8b9ab5; font-size:0.75rem; margin-top:2px;'>METR-LA · AIML-DL T036</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "**Navigation**",
        ["📊 Overview", "📈 Baseline Models", "🏆 Model Comparison",
         "⚡ Spatial Gap Analysis", "🔬 XGBoost Deep Dive"],
    )
    st.divider()

    p1_sb = load_phase1()
    if p1_sb:
        st.markdown("**Dataset Stats**")
        di = p1_sb.get("data", {})
        st.markdown(f"""
        <div style='font-size:0.78rem; color:#8b9ab5; line-height:1.9;'>
        📅 <b style='color:#d8eafd;'>{di.get('start','')[:10]}</b> to <b style='color:#d8eafd;'>{di.get('end','')[:10]}</b><br>
        🔵 <b style='color:#d8eafd;'>{di.get('sensors',207)}</b> sensors<br>
        ⏱ <b style='color:#d8eafd;'>{di.get('timestamps',34272):,}</b> timestamps<br>
        📐 <b style='color:#d8eafd;'>5-min</b> sampling interval<br>
        🗺 <b style='color:#d8eafd;'>Los Angeles</b> freeway network
        </div>
        """, unsafe_allow_html=True)


def metric_line(raw, metric, split, title):
    LABELS = {
        "persistence": "Persistence", "hist_avg": "Hist. Avg",
        "ridge": "Ridge", "xgb": "XGBoost", "rf": "Random Forest", "lstm": "LSTM",
    }
    COLORS = [MUTED, "#666460", YELLOW, BLUE, AQUA, ORANGE]
    fig = go.Figure()
    for (name, df), c in zip(raw.items(), COLORS):
        d = df[(df.split == split) & (df.horizon != "avg")].copy()
        d["min"] = d.horizon.astype(int) * 5
        lbl = LABELS.get(name, name)
        fig.add_trace(go.Scatter(
            x=d["min"], y=d[metric], mode="lines+markers",
            name=lbl, line=dict(color=c, width=2.2),
            marker=dict(size=5, color=c),
            hovertemplate=f"<b>{lbl}</b><br>%{{x}} min: %{{y:.3f}} mph<extra></extra>"
        ))
    apply_plotly_layout(fig,
        title=dict(text=title, font=dict(size=13, color=INK), x=0),
        xaxis_title="Forecast Horizon (min)",
        yaxis_title=f"{metric.upper()} (mph)",
        height=380,
    )
    return fig


# ============================================================
# PAGE: Overview
# ============================================================
if page == "📊 Overview":
    st.markdown("""
    <div class="hero-banner">
        <p class="hero-title">🚦 Traffic Flow Control — Results Dashboard</p>
        <p class="hero-sub">METR-LA Dataset · 207 freeway sensors · Los Angeles · 2012</p>
    </div>
    """, unsafe_allow_html=True)

    p1 = load_phase1()
    baselines = load_baselines()

    best_test_mae, best_model_name = None, "—"
    if not baselines.empty:
        test_df = baselines[baselines.split == "test"].sort_values("MAE_avg")
        if not test_df.empty:
            best_test_mae = test_df.iloc[0]["MAE_avg"]
            best_model_name = test_df.iloc[0]["model"]

    missingness = p1.get("missingness", {})
    graph_info = p1.get("graph", {})
    dist_info = p1.get("distribution", {})

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🔵 Sensors", f"{p1.get('data', {}).get('sensors', 207)}")
    c2.metric("📡 Data Points", f"{p1.get('data', {}).get('timestamps', 34272):,}")
    c3.metric("🕳 Outage Rate", f"{missingness.get('zero_pct', '—')}%",
              delta=f"Network: {missingness.get('network_outage_pct', '—')}%", delta_color="inverse")
    median_speed = dist_info.get('median')
    c4.metric("⚡ Median Speed", f"{median_speed:.0f} mph" if median_speed else "—")
    c5.metric("🏆 Best Avg-MAE",
              f"{best_test_mae:.3f} mph" if best_test_mae else "—",
              delta=best_model_name)

    st.markdown("---")
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="section-header">🗺 Dataset & Graph Summary</div>', unsafe_allow_html=True)
        if graph_info:
            g_data = {
                "Metric": ["Sensors (nodes)", "Road links (edges)", "Reciprocal edges", "Mean degree", "Isolated nodes"],
                "Value": [
                    str(graph_info.get("nodes", "—")),
                    str(graph_info.get("edges", "—")),
                    f"{graph_info.get('reciprocal_edge_pct', 0):.1f}%",
                    f"{graph_info.get('mean_degree', 0):.2f}",
                    str(graph_info.get("isolated_nodes", "—")),
                ]
            }
            st.dataframe(pd.DataFrame(g_data), hide_index=True, use_container_width=True)

        dist_vals = [
            ("≥ 60 mph (free flow)", dist_info.get("pct_ge_60", 0), BLUE),
            ("40–60 mph (slow)", 100 - dist_info.get("pct_ge_60", 0) - dist_info.get("pct_lt_40", 0), YELLOW),
            ("< 40 mph (congested)", dist_info.get("pct_lt_40", 0), ORANGE),
        ]
        fig_dist = go.Figure()
        for label, val, col in dist_vals:
            fig_dist.add_trace(go.Bar(
                name=label, x=[val], y=["Speed"], orientation="h",
                marker_color=col, text=[f"{val:.1f}%"], textposition="inside",
                insidetextanchor="middle",
                hovertemplate=f"<b>{label}</b>: {val:.1f}%<extra></extra>"
            ))
        apply_plotly_layout(fig_dist, barmode="stack",
            title=dict(text="Speed Distribution (non-outage readings)", font=dict(size=12, color=INK), x=0),
            height=160, showlegend=True,
            yaxis=dict(showticklabels=False, showgrid=False, gridcolor="rgba(0,0,0,0)"),
            legend=dict(orientation="h", y=-0.5, x=0),
            margin=dict(l=5, r=5, t=40, b=40),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">📉 Model Performance (Test MAE)</div>', unsafe_allow_html=True)
        if not baselines.empty:
            test_b = baselines[baselines.split == "test"].sort_values("rung")
            fig_bar = go.Figure(go.Bar(
                x=test_b["model"], y=test_b["MAE_avg"],
                marker=dict(
                    color=test_b["MAE_avg"],
                    colorscale=[[0, AQUA], [0.5, BLUE], [1.0, ORANGE]],
                    reversescale=True, showscale=False, line=dict(width=0),
                ),
                text=test_b["MAE_avg"].round(3),
                textposition="outside", textfont=dict(color=INK, size=10),
                hovertemplate="<b>%{x}</b><br>Avg MAE: %{y:.3f} mph<extra></extra>",
            ))
            apply_plotly_layout(fig_bar,
                title=dict(text="Average MAE across all 12 horizons (test block)", font=dict(size=12, color=INK), x=0),
                xaxis_title="", yaxis_title="MAE (mph)", height=340,
                xaxis=dict(tickangle=-20, gridcolor="rgba(255,255,255,0.06)", color=INK_2, linecolor="rgba(255,255,255,0.1)"),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown('<div class="section-header">💡 Key Findings</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="insight-card-blue">
        📡 <b>Dataset:</b> METR-LA covers 34,272 timestamps across 207 freeway sensors with 5-minute resolution.
        8.1% of readings are zero (sensor outage), of which 6.3% are whole-network blackouts.
        </div>""", unsafe_allow_html=True)
    with c2:
        shockwave = p1.get("propagation", {}).get("shockwave_kmh", "—")
        st.markdown(f"""<div class="insight-card">
        🌊 <b>Shockwave speed:</b> Congestion waves travel at ≈ <b>{shockwave} km/h</b> upstream (backward waves).
        156 of 499 clear road pairs show the downstream sensor slowing down first.
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="insight-card-orange">
        🏆 <b>Best baseline:</b> LSTM achieves the lowest test MAE (2.97 mph at 15-min) but ignores spatial structure.
        Adding XGBoost neighbour features reduces MAE by up to 3.5% — motivating GNN approaches.
        </div>""", unsafe_allow_html=True)


# ============================================================
# PAGE: Baseline Models
# ============================================================
elif page == "📈 Baseline Models":
    st.markdown('<div class="section-header">📈 Baseline Model — Error by Forecast Horizon</div>', unsafe_allow_html=True)

    raw = load_baselines_raw()
    if not raw:
        st.warning("No individual baseline CSV files found in results/baselines/.")
        st.stop()

    c1, c2 = st.columns(2)
    with c1:
        metric = st.selectbox("Metric", ["mae", "rmse", "mape"], format_func=str.upper)
    with c2:
        split = st.selectbox("Split", ["test", "val"])

    fig = metric_line(raw, metric, split, f"All baselines · {metric.upper()} by horizon ({split})")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">📋 Detailed Results at Key Horizons</div>', unsafe_allow_html=True)
    baselines_df = load_baselines()
    if not baselines_df.empty:
        test_df = baselines_df[baselines_df.split == split].copy().sort_values("rung")
        display_cols = ["model", "MAE_15min", "RMSE_15min", "MAPE_15min",
                        "MAE_30min", "RMSE_30min", "MAPE_30min",
                        "MAE_60min", "RMSE_60min", "MAPE_60min",
                        "MAE_avg", "RMSE_avg", "MAPE_avg"]
        display_cols = [c for c in display_cols if c in test_df.columns]
        styled = test_df[display_cols].reset_index(drop=True)
        styled.columns = [c.replace("_", " ") for c in styled.columns]
        st.dataframe(styled, hide_index=True, use_container_width=True)

    st.markdown('<div class="section-header">⏰ Peak vs Off-peak vs Weekend</div>', unsafe_allow_html=True)
    peak_df = load_peak()
    if not peak_df.empty:
        h_sel = st.select_slider("Forecast Horizon", options=[15, 30, 60], format_func=lambda x: f"{x} min")
        sub = peak_df[peak_df.minutes == h_sel]
        period_colors = {"peak": ORANGE, "offpeak": BLUE, "weekend": AQUA}
        period_labels = {"peak": "Peak (AM/PM rush)", "offpeak": "Off-peak weekday", "weekend": "Weekend"}

        fig_peak = go.Figure()
        for period in ["peak", "offpeak", "weekend"]:
            d = sub[sub.period == period]
            fig_peak.add_trace(go.Bar(
                name=period_labels[period], x=d["model"], y=d["mae"],
                marker_color=period_colors[period],
                hovertemplate="<b>%{x}</b><br>MAE: %{y:.3f} mph<extra></extra>",
                text=d["mae"].round(2), textposition="outside", textfont=dict(size=9, color=INK),
            ))
        apply_plotly_layout(fig_peak, barmode="group",
            title=dict(text=f"MAE by period · {h_sel}-min horizon", font=dict(size=12, color=INK), x=0),
            xaxis_title="", yaxis_title="MAE (mph)", height=380,
            xaxis=dict(tickangle=-15, gridcolor="rgba(255,255,255,0.06)", color=INK_2, linecolor="rgba(255,255,255,0.1)"),
        )
        st.plotly_chart(fig_peak, use_container_width=True)
        st.markdown("""<div class="insight-card-orange">
        ⚠️ <b>Peak vs Off-peak gap:</b> All models struggle significantly during rush hours — MAE is 1.5–2×
        higher during AM/PM peaks compared to off-peak times, reflecting the difficulty of modelling congestion onset.
        </div>""", unsafe_allow_html=True)


# ============================================================
# PAGE: Model Comparison
# ============================================================
elif page == "🏆 Model Comparison":
    st.markdown('<div class="section-header">🏆 Head-to-Head Model Comparison</div>', unsafe_allow_html=True)

    baselines = load_baselines()
    if baselines.empty:
        st.warning("results/baselines.csv not found.")
        st.stop()

    split = st.radio("Split", ["test", "val"], horizontal=True)
    df = baselines[baselines.split == split].sort_values("rung")

    # Radar chart
    all_m = ["MAE_15min", "RMSE_15min", "MAPE_15min", "MAE_60min", "RMSE_60min", "MAPE_60min"]
    theta_labels = ["MAE 15m", "RMSE 15m", "MAPE 15m", "MAE 60m", "RMSE 60m", "MAPE 60m"]
    df_norm = df.copy()
    for m in all_m:
        mn, mx = df[m].min(), df[m].max()
        df_norm[m] = 1 - (df[m] - mn) / (mx - mn + 1e-9)

    COLORS_RADAR = [MUTED, "#666460", YELLOW, BLUE, AQUA, ORANGE]
    fig_radar = go.Figure()
    for row, c in zip(df_norm.itertuples(), COLORS_RADAR):
        vals = [getattr(row, m) for m in all_m]
        vals_close = vals + [vals[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals_close, theta=theta_labels + [theta_labels[0]],
            fill="toself", name=row.model,
            line=dict(color=c, width=2),
            opacity=0.85,
            hovertemplate=f"<b>{row.model}</b><br>%{{theta}}: %{{r:.3f}}<extra></extra>",
        ))
    apply_plotly_layout(fig_radar,
        polar=dict(
            bgcolor="rgba(13,17,23,0.5)",
            angularaxis=dict(linecolor=MUTED, gridcolor="rgba(255,255,255,0.08)", tickfont=dict(color=INK_2, size=10)),
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color=INK_2, size=8)),
        ),
        title=dict(text="Model Quality Radar (higher = better, normalised)", font=dict(size=13, color=INK), x=0.5, xanchor="center"),
        height=440,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("---")

    # Heatmap
    st.markdown('<div class="section-header">🌡 Metric Heatmap</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        metric_group = st.selectbox("Metric group", ["MAE", "RMSE", "MAPE"])
    with c2:
        heat_split = st.selectbox("Split ", ["test", "val"])

    heat_df = baselines[baselines.split == heat_split].sort_values("rung")
    heat_cols = [f"{metric_group}_15min", f"{metric_group}_30min", f"{metric_group}_60min", f"{metric_group}_avg"]
    heat_cols = [c for c in heat_cols if c in heat_df.columns]
    heat_data = heat_df.set_index("model")[heat_cols]
    heat_data.columns = ["15 min", "30 min", "60 min", "Avg"]

    fig_heat = go.Figure(go.Heatmap(
        z=heat_data.values, x=heat_data.columns.tolist(), y=heat_data.index.tolist(),
        colorscale=[[0, AQUA], [0.5, BLUE], [1.0, ORANGE]],
        text=[[f"{v:.3f}" for v in row] for row in heat_data.values],
        texttemplate="%{text}", textfont=dict(size=11, color=INK),
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.3f}<extra></extra>",
        colorbar=dict(tickfont=dict(color=INK_2), title=dict(text=metric_group, font=dict(color=INK_2))),
    ))
    apply_plotly_layout(fig_heat,
        title=dict(text=f"{metric_group} Heatmap — Models vs Horizons ({heat_split})", font=dict(size=12, color=INK), x=0),
        height=320,
        xaxis=dict(side="top", gridcolor="rgba(255,255,255,0.06)", color=INK_2, linecolor="rgba(255,255,255,0.1)"),
        yaxis=dict(autorange="reversed", gridcolor="rgba(255,255,255,0.06)", color=INK_2, linecolor="rgba(255,255,255,0.1)"),
        margin=dict(l=10, r=10, t=70, b=10),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # Improvement over Persistence
    st.markdown('<div class="section-header">📉 MAE Improvement Over Persistence</div>', unsafe_allow_html=True)
    test_b = baselines[baselines.split == split].sort_values("rung")
    persist_mae = test_b[test_b.model.str.contains("Persistence")]["MAE_avg"].values
    if len(persist_mae) > 0:
        persist_val = persist_mae[0]
        non_persist = test_b[~test_b.model.str.contains("Persistence")].copy()
        non_persist["improvement_%"] = (1 - non_persist["MAE_avg"] / persist_val) * 100
        fig_imp = go.Figure(go.Bar(
            x=non_persist["improvement_%"], y=non_persist["model"], orientation="h",
            marker=dict(color=non_persist["improvement_%"], colorscale=[[0, ORANGE], [1, AQUA]], showscale=False),
            text=non_persist["improvement_%"].apply(lambda v: f"+{v:.1f}%"),
            textposition="outside", textfont=dict(color=INK, size=10),
            hovertemplate="<b>%{y}</b><br>Improvement: +%{x:.1f}%<extra></extra>",
        ))
        apply_plotly_layout(fig_imp,
            title=dict(text="% MAE reduction vs. Persistence", font=dict(size=12, color=INK), x=0),
            xaxis_title="MAE Improvement (%)", yaxis_title="", height=280,
            yaxis=dict(autorange="reversed", gridcolor="rgba(255,255,255,0.06)", color=INK_2, linecolor="rgba(255,255,255,0.1)"),
        )
        st.plotly_chart(fig_imp, use_container_width=True)


# ============================================================
# PAGE: Spatial Gap Analysis
# ============================================================
elif page == "⚡ Spatial Gap Analysis":
    st.markdown("""
    <div class="hero-banner">
        <p class="hero-title">⚡ Spatial Gap Analysis (Phase 4)</p>
        <p class="hero-sub">Evidence that per-sensor models miss spatial propagation — motivation for Graph Neural Networks</p>
    </div>
    """, unsafe_allow_html=True)

    p4 = load_phase4()
    if p4:
        c1, c2 = st.columns(2)
        onsets = p4.get("test_onsets")
        onset_pct = p4.get("onset_share_of_targets_pct")
        c1.metric("🔥 Test Congestion Onsets", f"{onsets:,}" if isinstance(onsets, int) else "—")
        c2.metric("📊 Onset Share of Test Targets", f"{onset_pct:.2f}%" if isinstance(onset_pct, float) else "—")

    tabs = st.tabs(["Gap 1: Horizon Degradation", "Gap 2: Onset Errors",
                    "Gap 3: Neighbour Features", "Gap 4: Residual Correlation"])

    with tabs[0]:
        raw = load_baselines_raw()
        if raw:
            fig_g1 = go.Figure()
            for name, label, col, dash in [
                ("xgb", "XGBoost MAE", BLUE, "solid"),
                ("xgb", "XGBoost RMSE", BLUE, "dash"),
                ("persistence", "Persistence MAE", MUTED, "solid"),
            ]:
                if name not in raw:
                    continue
                d = raw[name][(raw[name].split == "test") & (raw[name].horizon != "avg")].copy()
                d["min"] = d.horizon.astype(int) * 5
                metric_col = "rmse" if "RMSE" in label else "mae"
                fig_g1.add_trace(go.Scatter(
                    x=d["min"], y=d[metric_col], mode="lines+markers",
                    name=label, line=dict(color=col, dash=dash, width=2.3),
                    marker=dict(size=5),
                    hovertemplate=f"<b>{label}</b><br>%{{x}} min: %{{y:.3f}} mph<extra></extra>",
                ))
            apply_plotly_layout(fig_g1,
                title=dict(text="Gap 1 · Best per-sensor model degrades steadily with horizon", font=dict(size=12, color=INK), x=0),
                xaxis_title="Forecast Horizon (min)", yaxis_title="Error (mph)", height=360,
            )
            st.plotly_chart(fig_g1, use_container_width=True)
            st.markdown("""<div class="insight-card-blue">
            📈 <b>Gap 1:</b> XGBoost error grows monotonically from 15 min to 60 min. This is expected for a model
            that only sees its own sensor's current and past speed — the further into the future, the more it misses
            congestion waves travelling from neighbouring sensors.
            </div>""", unsafe_allow_html=True)

    with tabs[1]:
        onset = load_onset()
        if not onset.empty:
            fig_g2 = go.Figure()
            for model, label, col in [("xgb", "XGBoost", BLUE), ("persistence", "Persistence", MUTED)]:
                d = onset[onset.model == model].pivot(index="horizon", columns="group", values="mae")
                if d.empty:
                    continue
                fig_g2.add_trace(go.Scatter(
                    x=d.index * 5, y=d["onset"], mode="lines+markers",
                    name=f"{label}: after onset", line=dict(color=col, width=2.3),
                    marker=dict(size=5),
                    hovertemplate=f"<b>{label} (onset)</b><br>%{{x}} min: %{{y:.2f}} mph<extra></extra>",
                ))
                fig_g2.add_trace(go.Scatter(
                    x=d.index * 5, y=d["normal"], mode="lines+markers",
                    name=f"{label}: normal", line=dict(color=col, dash="dash", width=2),
                    marker=dict(size=4),
                    hovertemplate=f"<b>{label} (normal)</b><br>%{{x}} min: %{{y:.2f}} mph<extra></extra>",
                ))
            nb = onset[onset.model == "xgb_both"]
            if not nb.empty:
                d = nb.pivot(index="horizon", columns="group", values="mae")
                fig_g2.add_trace(go.Scatter(
                    x=d.index * 5, y=d["onset"], mode="markers",
                    name="XGB + neighbours: onset",
                    marker=dict(symbol="star", size=12, color=ORANGE, line=dict(width=1, color=INK)),
                    hovertemplate="<b>XGB+neighbours (onset)</b><br>%{x} min: %{y:.2f} mph<extra></extra>",
                ))
            apply_plotly_layout(fig_g2,
                title=dict(text="Gap 2 · Errors are several times larger right after congestion starts", font=dict(size=12, color=INK), x=0),
                xaxis_title="Forecast Horizon (min)", yaxis_title="MAE (mph)", height=380,
            )
            st.plotly_chart(fig_g2, use_container_width=True)
            st.markdown("""<div class="insight-card-orange">
            🔴 <b>Gap 2:</b> In the 30 minutes after a congestion onset event, XGBoost's MAE is 5–7x higher than
            at normal times. Adding neighbour features (XGB + both) partially closes this gap.
            </div>""", unsafe_allow_html=True)

    with tabs[2]:
        gap_res = load_gap_results()
        if not gap_res.empty:
            test_g = gap_res[gap_res.split == "test"]
            order = ["xgb (no neighbours)", "xgb + upstream", "xgb + downstream", "xgb + both"]
            colors_g3 = [MUTED, AQUA, YELLOW, ORANGE]
            mins_g3 = [15, 30, 60]
            fig_g3 = go.Figure()
            for model, col in zip(order, colors_g3):
                rows = test_g[test_g.model == model]
                if rows.empty:
                    continue
                r = rows.iloc[0]
                vals = [r.get(f"MAE_{m}min", np.nan) for m in mins_g3]
                gains = [r.get(f"MAE_gain_{m}min_%", np.nan) for m in mins_g3]
                fig_g3.add_trace(go.Bar(
                    name=model, x=[f"{m} min" for m in mins_g3], y=vals,
                    marker_color=col, width=0.18,
                    text=[f"{g:+.1f}%" if model != "xgb (no neighbours)" and not np.isnan(g) else "" for g in gains],
                    textposition="outside", textfont=dict(size=9, color=INK_2),
                    hovertemplate=f"<b>{model}</b><br>%{{x}}: %{{y:.3f}} mph<extra></extra>",
                ))
            apply_plotly_layout(fig_g3, barmode="group",
                title=dict(text="Gap 3 · Neighbour speeds lower XGBoost error at every horizon", font=dict(size=12, color=INK), x=0),
                xaxis_title="", yaxis_title="MAE (mph)", height=380,
            )
            st.plotly_chart(fig_g3, use_container_width=True)

            gain_data = []
            for model in order:
                rows = test_g[test_g.model == model]
                if rows.empty:
                    continue
                r = rows.iloc[0]
                gain_data.append({
                    "Model": model,
                    "MAE 15min": round(r.get("MAE_15min", np.nan), 3),
                    "MAE 30min": round(r.get("MAE_30min", np.nan), 3),
                    "MAE 60min": round(r.get("MAE_60min", np.nan), 3),
                    "Gain 15min %": round(r.get("MAE_gain_15min_%", np.nan), 1),
                    "Gain 30min %": round(r.get("MAE_gain_30min_%", np.nan), 1),
                    "Gain 60min %": round(r.get("MAE_gain_60min_%", np.nan), 1),
                })
            st.dataframe(pd.DataFrame(gain_data), hide_index=True, use_container_width=True)
            st.markdown("""<div class="insight-card">
            ✅ <b>Gap 3:</b> Adding both upstream and downstream neighbour speed features to XGBoost reduces MAE
            by 3.5% at 15 min. The gain is consistent across all horizons, motivating Graph Neural Networks.
            </div>""", unsafe_allow_html=True)

    with tabs[3]:
        res_corr = load_residual_corr()
        if not res_corr.empty:
            model_sel = st.selectbox(
                "Model", ["xgb", "xgb_both"],
                format_func=lambda x: "XGBoost" if x == "xgb" else "XGBoost + Both Neighbours"
            )
            d_rc = res_corr[res_corr.model == model_sel]
            groups = list(dict.fromkeys(d_rc.group))
            horizon_colors = {3: BLUE_L, 6: BLUE, 12: "#184f95"}
            fig_g4 = go.Figure()
            for h in sorted(d_rc.horizon.unique()):
                v = d_rc[d_rc.horizon == h].set_index("group").loc[groups, "mean_corr"]
                fig_g4.add_trace(go.Bar(
                    name=f"{5 * h}-min forecast errors",
                    x=groups, y=v.values,
                    marker_color=horizon_colors.get(h, BLUE),
                    hovertemplate=f"<b>%{{x}}</b><br>{5*h}-min corr: %{{y:.4f}}<extra></extra>",
                ))
            apply_plotly_layout(fig_g4, barmode="group",
                title=dict(text="Gap 4 · Errors of linked sensors move together", font=dict(size=12, color=INK), x=0),
                xaxis_title="Road-graph separation (hop count)", yaxis_title="Mean residual correlation",
                height=360,
            )
            st.plotly_chart(fig_g4, use_container_width=True)
            st.markdown("""<div class="insight-card-blue">
            🔗 <b>Gap 4:</b> XGBoost residuals of directly linked sensors (1 hop apart) are correlated at ~0.12–0.20,
            much higher than unconnected sensor pairs. This confirms correlated mistakes along the road network —
            a signature of missed spatial propagation that a GNN would learn to exploit.
            </div>""", unsafe_allow_html=True)


# ============================================================
# PAGE: XGBoost Deep Dive
# ============================================================
elif page == "🔬 XGBoost Deep Dive":
    st.markdown('<div class="section-header">🔬 XGBoost Feature Importance</div>', unsafe_allow_html=True)

    imp = load_xgb_importance()
    if imp.empty:
        st.warning("XGBoost importance file not found (results/baselines/xgb_importance.csv).")
        st.stop()

    c1, c2 = st.columns(2)
    with c1:
        horizon_sel = st.select_slider(
            "Forecast horizon (steps)", options=sorted(imp.columns.tolist()), value=3,
            format_func=lambda x: f"{x * 5} min"
        )
    with c2:
        top_n = st.slider("Top N features", 5, 30, 15)

    col_l, col_r = st.columns(2)
    for ax_col, h, col in zip([col_l, col_r], [3, 12], [BLUE, ORANGE]):
        top_f = imp[h].sort_values(ascending=False).head(top_n) * 100
        fig_imp = go.Figure(go.Bar(
            y=top_f.index.tolist(), x=top_f.values,
            orientation="h",
            marker=dict(color=col, opacity=0.85, line=dict(width=0)),
            hovertemplate="<b>%{y}</b><br>Gain share: %{x:.2f}%<extra></extra>",
        ))
        apply_plotly_layout(fig_imp,
            title=dict(text=f"Top {top_n} features · {h * 5}-min horizon", font=dict(size=12, color=INK), x=0),
            xaxis_title="Share of total gain (%)",
            yaxis=dict(autorange="reversed", gridcolor="rgba(255,255,255,0.06)", color=INK_2, linecolor="rgba(255,255,255,0.1)"),
            height=420, margin=dict(l=180, r=10, t=40, b=10),
        )
        ax_col.plotly_chart(fig_imp, use_container_width=True)

    st.markdown(f'<div class="section-header">📊 Feature Importance at {horizon_sel * 5}-min Horizon</div>',
                unsafe_allow_html=True)
    h_data = imp[horizon_sel].sort_values(ascending=False).head(top_n) * 100
    fig_sel = go.Figure(go.Bar(
        x=h_data.index.tolist(), y=h_data.values,
        marker=dict(
            color=h_data.values,
            colorscale=[[0, AQUA], [0.5, BLUE], [1.0, ORANGE]],
            reversescale=True, showscale=False,
        ),
        hovertemplate="<b>%{x}</b><br>Gain share: %{y:.2f}%<extra></extra>",
    ))
    apply_plotly_layout(fig_sel,
        title=dict(text=f"Feature gain share at {horizon_sel * 5}-min horizon (top {top_n})", font=dict(size=12, color=INK), x=0),
        xaxis_title="Feature", yaxis_title="% of total gain", height=340,
        xaxis=dict(tickangle=-35, gridcolor="rgba(255,255,255,0.06)", color=INK_2, linecolor="rgba(255,255,255,0.1)"),
    )
    st.plotly_chart(fig_sel, use_container_width=True)

    top_feature = imp[3].sort_values(ascending=False).index[0]
    st.markdown(f"""<div class="insight-card-blue">
    🔍 <b>XGBoost feature reliance:</b> At the 15-min horizon, the top feature is <b>"{top_feature}"</b>.
    The model leans heavily on each sensor's own recent speed and time-of-day signal — it does NOT use
    any information from neighbouring sensors. This is exactly the spatial gap that neighbour features
    and GNNs are designed to fill.
    </div>""", unsafe_allow_html=True)

    xgb_meta_file = BASELINES_DIR / "xgb.json"
    if xgb_meta_file.exists():
        xgb_meta = json.loads(xgb_meta_file.read_text())
        st.markdown('<div class="section-header">⚙️ XGBoost Hyperparameters</div>', unsafe_allow_html=True)
        params = xgb_meta.get("params", {})
        best_rounds = xgb_meta.get("best_rounds", {})
        param_df = pd.DataFrame([{"Parameter": k, "Value": str(v)} for k, v in params.items()])
        if not param_df.empty:
            c1, c2 = st.columns([1, 1])
            with c1:
                st.dataframe(param_df, hide_index=True, use_container_width=True)
            with c2:
                if best_rounds:
                    rounds_df = pd.DataFrame([{"Horizon (step)": k, "Best Round": v}
                                              for k, v in best_rounds.items()])
                    st.caption("Best boosting rounds per horizon")
                    st.dataframe(rounds_df, hide_index=True, use_container_width=True)


# Footer
st.markdown("""
<div class="footer">
    🚦 Traffic Flow Control · METR-LA Dashboard &nbsp;|&nbsp; AIML-DL T036 &nbsp;|&nbsp;
    Built with <a href="https://streamlit.io" target="_blank" style="color:#2a78d6;">Streamlit</a> &
    <a href="https://plotly.com" target="_blank" style="color:#2a78d6;">Plotly</a>
</div>
""", unsafe_allow_html=True)
