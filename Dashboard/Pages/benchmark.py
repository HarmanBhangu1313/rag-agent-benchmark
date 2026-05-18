import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ── Real benchmark data from notebooks ───────────────────────────────────────
RESULTS = {
    "with_chunking": pd.DataFrame([
        {"Agent": "Single-Agent", "Dataset": "SQuAD",    "EM": 32.0, "F1": 37.1},
        {"Agent": "Multi-Agent",  "Dataset": "SQuAD",    "EM": 40.0, "F1": 45.0},
        {"Agent": "Single-Agent", "Dataset": "HotpotQA", "EM": 30.0, "F1": 41.1},
        {"Agent": "Multi-Agent",  "Dataset": "HotpotQA", "EM": 37.0, "F1": 47.6},
    ]),
    "ablation": pd.DataFrame([
        {"Agent": "Single-Agent", "Chunking": "No Chunking",  "EM": 48.0, "F1": 55.3},
        {"Agent": "Multi-Agent",  "Chunking": "No Chunking",  "EM": 47.0, "F1": 52.4},
        {"Agent": "Single-Agent", "Chunking": "With Chunking","EM": 32.0, "F1": 37.1},
        {"Agent": "Multi-Agent",  "Chunking": "With Chunking","EM": 40.0, "F1": 45.0},
    ])
}

COLORS = {
    "Single-Agent": "#a371f7",
    "Multi-Agent":  "#3fb950",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Grotesk, sans-serif", color="#8b949e"),
    xaxis=dict(gridcolor="#1e1e2e", zerolinecolor="#2d2d3d", tickfont=dict(color="#8b949e")),
    yaxis=dict(gridcolor="#1e1e2e", zerolinecolor="#2d2d3d", tickfont=dict(color="#8b949e")),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#c9d1d9")),
    margin=dict(t=40, b=40, l=40, r=40),
)


def grouped_bar(df, x_col, metric, title):
    fig = go.Figure()
    for agent, color in COLORS.items():
        sub = df[df["Agent"] == agent]
        fig.add_trace(go.Bar(
            name=agent,
            x=sub[x_col],
            y=sub[metric],
            marker_color=color,
            marker_line_width=0,
            text=[f"{v:.1f}%" for v in sub[metric]],
            textposition="outside",
            textfont=dict(size=11, color="#c9d1d9"),
        ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=title, font=dict(color="#e6edf3", size=14)),
        barmode="group",
        bargap=0.25,
        bargroupgap=0.08,
        yaxis_range=[0, 75],
        yaxis_title=metric,
        height=320,
    )
    return fig


def show():
    st.markdown("""
    <div class="platform-header">
        <div class="platform-title">📊 Benchmark Comparison</div>
        <div class="platform-sub">Single-Agent RAG vs Multi-Agent RAG (LangGraph) — SQuAD & HotpotQA · 100 queries each · Groq gpt-oss-120b</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Top-level metrics ────────────────────────────────────────────────────
    st.markdown("#### Overall Results — With Chunking")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Multi-Agent EM (SQuAD)",    "40.0%", "+8.0% vs Single-Agent")
    c2.metric("Multi-Agent F1 (SQuAD)",    "45.0%", "+7.9% vs Single-Agent")
    c3.metric("Multi-Agent EM (HotpotQA)", "37.0%", "+7.0% vs Single-Agent")
    c4.metric("Multi-Agent F1 (HotpotQA)", "47.6%", "+6.5% vs Single-Agent")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── EM and F1 charts ─────────────────────────────────────────────────────
    col_em, col_f1 = st.columns(2)
    with col_em:
        fig_em = grouped_bar(RESULTS["with_chunking"], "Dataset", "EM", "Exact Match — With Chunking")
        st.plotly_chart(fig_em, use_container_width=True)
    with col_f1:
        fig_f1 = grouped_bar(RESULTS["with_chunking"], "Dataset", "F1", "F1 Score — With Chunking")
        st.plotly_chart(fig_f1, use_container_width=True)

    # ── Ablation study ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Ablation: Effect of Chunking on SQuAD")

    st.markdown("""
    <div class="insight-card">
        <div class="insight-title">⚡ Key Discovery — Chunking Flipped the Winner</div>
        <div class="insight-body">
            Without chunking, Single-Agent led on SQuAD (48% vs 47% EM). After introducing chunking,
            Multi-Agent overtook it (40% vs 32% EM). The winning architecture is not fixed —
            it depends entirely on preprocessing strategy.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_abl_em, col_abl_f1 = st.columns(2)
    with col_abl_em:
        fig_abl_em = grouped_bar(RESULTS["ablation"], "Chunking", "EM", "EM: Chunking Ablation (SQuAD)")
        st.plotly_chart(fig_abl_em, use_container_width=True)
    with col_abl_f1:
        fig_abl_f1 = grouped_bar(RESULTS["ablation"], "Chunking", "F1", "F1: Chunking Ablation (SQuAD)")
        st.plotly_chart(fig_abl_f1, use_container_width=True)

    # ── Delta table ──────────────────────────────────────────────────────────
    st.markdown("#### EM Drop After Chunking — Architecture Robustness")
    delta_df = pd.DataFrame([
        {"Agent": "Single-Agent", "EM (No Chunk)": "48.0%", "EM (Chunked)": "32.0%",
         "EM Drop": "−16.0%", "F1 Drop": "−18.2%", "Robustness": "⚠️ High drop"},
        {"Agent": "Multi-Agent",  "EM (No Chunk)": "47.0%", "EM (Chunked)": "40.0%",
         "EM Drop": "−7.0%",  "F1 Drop": "−7.4%",  "Robustness": "✅ Resilient"},
    ])
    st.dataframe(delta_df, use_container_width=True, hide_index=True)

    # ── Radar chart ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Architecture Comparison — Full Profile")

    categories = ["SQuAD EM", "SQuAD F1", "HotpotQA EM", "HotpotQA F1", "Chunking Robustness"]
    single_vals = [32, 37.1, 30, 41.1, 30]   # robustness = relative score
    multi_vals  = [40, 45.0, 37, 47.6, 70]

    fig_radar = go.Figure()
    for name, vals, color in [("Single-Agent", single_vals, "#a371f7"), ("Multi-Agent", multi_vals, "#3fb950")]:
        fig_radar.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=name,
            line=dict(color=color, width=2),
            fillcolor=color.replace(")", ", 0.12)").replace("rgb", "rgba") if "rgb" in color else color + "20",
        ))
    fig_radar.update_layout(
        **PLOTLY_LAYOUT,
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 80], tickfont=dict(color="#8b949e", size=9), gridcolor="#1e1e2e"),
            angularaxis=dict(tickfont=dict(color="#c9d1d9", size=11), gridcolor="#2d2d3d"),
        ),
        height=380,
        title=dict(text="Multi-dimensional performance profile", font=dict(color="#e6edf3", size=13)),
    )
    col_r, col_insight = st.columns([3, 2])
    with col_r:
        st.plotly_chart(fig_radar, use_container_width=True)
    with col_insight:
        st.markdown("<br>", unsafe_allow_html=True)
        for title, body in [
            ("Dataset complexity dominates", "Both agents score significantly lower on HotpotQA. Multi-hop reasoning across 10 docs is a stronger performance driver than architecture choice."),
            ("Query rewriting buffers degradation", "Multi-Agent EM dropped only −7% after chunking vs −16% for Single-Agent. The rewrite node compensates for noisier retrieval."),
            ("F1 > EM gap on HotpotQA", "Single-Agent F1 (41.1%) is far above EM (30%), suggesting partial matches. Multi-Agent closes this gap further.")
        ]:
            st.markdown(f"""
            <div class="insight-card">
                <div class="insight-title">{title}</div>
                <div class="insight-body">{body}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Full results table ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Full Results Table")
    full_df = pd.DataFrame([
        {"Agent": "Single-Agent", "Dataset": "SQuAD",    "Chunking": "Yes", "EM": "32.0%", "F1": "37.1%", "Winner": ""},
        {"Agent": "Multi-Agent",  "Dataset": "SQuAD",    "Chunking": "Yes", "EM": "40.0%", "F1": "45.0%", "Winner": "✅"},
        {"Agent": "Single-Agent", "Dataset": "HotpotQA", "Chunking": "Yes", "EM": "30.0%", "F1": "41.1%", "Winner": ""},
        {"Agent": "Multi-Agent",  "Dataset": "HotpotQA", "Chunking": "Yes", "EM": "37.0%", "F1": "47.6%", "Winner": "✅"},
        {"Agent": "Single-Agent", "Dataset": "SQuAD",    "Chunking": "No",  "EM": "48.0%", "F1": "55.3%", "Winner": "✅"},
        {"Agent": "Multi-Agent",  "Dataset": "SQuAD",    "Chunking": "No",  "EM": "47.0%", "F1": "52.4%", "Winner": ""},
    ])
    st.dataframe(full_df, use_container_width=True, hide_index=True)
