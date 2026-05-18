import streamlit as st

st.set_page_config(
    page_title="AI Agent Reliability Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0a0a0f;
    border-right: 1px solid #1e1e2e;
}
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label { color: #8b949e !important; }

/* Main background */
.main { background: #0d0d14; }
.block-container { padding-top: 2rem; max-width: 1400px; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #161622;
    border: 1px solid #2d2d3d;
    border-radius: 12px;
    padding: 1rem;
}
[data-testid="metric-container"] label { color: #8b949e !important; font-size: 0.75rem !important; letter-spacing: 0.1em; text-transform: uppercase; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #e6edf3 !important; font-size: 2rem !important; font-weight: 600; }
[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

/* Headers */
h1, h2, h3 { color: #e6edf3 !important; }
h1 { font-weight: 700; letter-spacing: -0.02em; }

/* Tables */
[data-testid="stDataFrame"] { border: 1px solid #2d2d3d; border-radius: 8px; overflow: hidden; }

/* Badges */
.badge-multi { background: #1a3a2a; color: #3fb950; border: 1px solid #3fb950; padding: 2px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.05em; }
.badge-single { background: #2a1a3a; color: #a371f7; border: 1px solid #a371f7; padding: 2px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.05em; }
.badge-squad { background: #1a2a3a; color: #58a6ff; border: 1px solid #58a6ff; padding: 2px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 600; }
.badge-hotpot { background: #3a2a1a; color: #f0883e; border: 1px solid #f0883e; padding: 2px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 600; }

/* Platform header */
.platform-header {
    background: linear-gradient(135deg, #161622 0%, #1a1a2e 100%);
    border: 1px solid #2d2d3d;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
}
.platform-title { font-size: 1.6rem; font-weight: 700; color: #e6edf3; margin: 0; letter-spacing: -0.02em; }
.platform-sub { color: #8b949e; font-size: 0.85rem; margin-top: 4px; }
.accent { color: #58a6ff; }

/* Insight card */
.insight-card {
    background: #161622;
    border: 1px solid #2d2d3d;
    border-left: 3px solid #58a6ff;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
}
.insight-title { color: #e6edf3; font-weight: 600; font-size: 0.9rem; margin-bottom: 4px; }
.insight-body { color: #8b949e; font-size: 0.82rem; line-height: 1.6; }

/* Nav pills in sidebar */
div[data-testid="stRadio"] > label { display: none; }
div[data-testid="stRadio"] div[role="radiogroup"] { gap: 4px; }
div[data-testid="stRadio"] label {
    display: flex !important;
    align-items: center;
    padding: 8px 14px;
    border-radius: 8px;
    cursor: pointer;
    transition: background 0.15s;
    font-size: 0.87rem;
    color: #8b949e !important;
}
div[data-testid="stRadio"] label:has(input:checked) {
    background: #1e1e2e;
    color: #e6edf3 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar Navigation ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem 0 1.5rem;">
        <div style="font-size: 1.3rem; font-weight: 700; color: #e6edf3; letter-spacing: -0.02em;">
            ⚡ 
        </div>
        <div style="color: #8b949e; font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; margin-top: 2px;">
            AI Agent Reliability Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div style="color:#8b949e;font-size:0.7rem;letter-spacing:0.1em;text-transform:uppercase;padding:0 0 8px;">Navigation</div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["📊  Benchmark Comparison",
         "🔍  Retrieval Trace Viewer",
         "🧩  Chunking Impact",
         "❌  Failure Analysis"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem; color:#8b949e; line-height:1.7;">
        <div style="color:#e6edf3; font-weight:600; margin-bottom:6px;">Experiment Config</div>
        <div>📦 Dataset: SQuAD + HotpotQA</div>
        <div>🔢 Queries: 100 per run</div>
        <div>🤖 LLM: gpt-oss-120b (Groq)</div>
        <div>📐 Embeddings: MiniLM-L6-v2</div>
        <div>🗃️ Vector DB: FAISS CPU</div>
    </div>
    """, unsafe_allow_html=True)

# ── Page routing ──────────────────────────────────────────────────────────────
if "Benchmark" in page:
    import pages.benchmark as benchmark
    benchmark.show()
elif "Trace" in page:
    import pages.trace_viewer as trace_viewer
    trace_viewer.show()
elif "Chunking" in page:
    import pages.chunking_impact as chunking_impact
    chunking_impact.show()
elif "Failure" in page:
    import pages.chunking_impact as chunking_impact
    failure_analysis.show()
