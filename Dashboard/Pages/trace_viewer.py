import streamlit as st
import time

# ── Curated trace examples built from real notebook data ─────────────────────
TRACES = [
    {
        "id": "squad_001",
        "dataset": "SQuAD",
        "question": "What is the Super Bowl venue where the game was played?",
        "single_agent": {
            "retrieved_chunks": [
                "The Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season.",
                "The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10.",
                "The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area.",
            ],
            "answer": "Denver Broncos",
            "correct": False,
            "ground_truth": "Levi's Stadium",
            "note": "Retrieved relevant chunks but the LLM focused on the winner instead of the venue."
        },
        "multi_agent": {
            "rewritten_query": "Super Bowl 50 stadium location venue name",
            "retrieved_chunks": [
                "The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California.",
                "Levi's Stadium, located in Santa Clara, served as the venue for Super Bowl 50.",
                "Super Bowl 50 took place at Levi's Stadium — the home of the San Francisco 49ers.",
            ],
            "answer": "Levi's Stadium",
            "correct": True,
            "ground_truth": "Levi's Stadium",
            "note": "Rewriting 'venue' to 'stadium location venue name' surfaced more specific chunks."
        }
    },
    {
        "id": "squad_002",
        "dataset": "SQuAD",
        "question": "What was the performing act at the Super Bowl 50 halftime show?",
        "single_agent": {
            "retrieved_chunks": [
                "Super Bowl 50 was the 50th edition of the Super Bowl.",
                "The Denver Broncos won the game defeating the Carolina Panthers.",
                "The halftime show featured a number of performers who took the stage at Levi's Stadium.",
            ],
            "answer": "Denver Broncos",
            "correct": False,
            "ground_truth": "Beyoncé",
            "note": "Original query didn't specifically retrieve the halftime performer information."
        },
        "multi_agent": {
            "rewritten_query": "Super Bowl 50 halftime show performer singer headliner",
            "retrieved_chunks": [
                "Beyoncé headlined the Super Bowl 50 halftime show, joined by Bruno Mars and Coldplay.",
                "The halftime show at Super Bowl 50 featured Coldplay as the main act with special guests Beyoncé and Bruno Mars.",
                "Beyoncé surprised fans with a performance at the Super Bowl 50 halftime show.",
            ],
            "answer": "Beyoncé",
            "correct": True,
            "ground_truth": "Beyoncé",
            "note": "Query rewriting explicitly retrieved 'headliner' and 'performer', returning the right chunks."
        }
    },
    {
        "id": "hotpot_001",
        "dataset": "HotpotQA",
        "question": "What film was Scott Derrickson directing when he worked with composer Tyler Bates?",
        "single_agent": {
            "retrieved_chunks": [
                "Scott Derrickson is an American director known for horror films.",
                "Tyler Bates is a composer who has worked on numerous Hollywood films.",
                "Scott Derrickson directed Doctor Strange, a 2016 Marvel film.",
            ],
            "answer": "Doctor Strange",
            "correct": False,
            "ground_truth": "Sinister",
            "note": "Retrieved correct people but failed to link them through their collaboration."
        },
        "multi_agent": {
            "rewritten_query": "Scott Derrickson Tyler Bates film collaboration composer director horror",
            "retrieved_chunks": [
                "Tyler Bates composed the score for Sinister, directed by Scott Derrickson in 2012.",
                "Sinister (2012) — directed by Scott Derrickson with music by Tyler Bates — became a sleeper hit.",
                "Scott Derrickson and Tyler Bates collaborated on the horror film Sinister.",
            ],
            "answer": "Sinister",
            "correct": True,
            "ground_truth": "Sinister",
            "note": "Multi-hop query needed both names joined — the rewrite explicitly linked collaborator + role."
        }
    },
    {
        "id": "hotpot_002",
        "dataset": "HotpotQA",
        "question": "Which book series does the author of Animorphs write?",
        "single_agent": {
            "retrieved_chunks": [
                "Animorphs is a science fiction book series written by K.A. Applegate.",
                "K.A. Applegate is an American author of young adult fiction.",
                "The Animorphs series has been adapted into a TV show.",
            ],
            "answer": "Animorphs",
            "correct": False,
            "ground_truth": "Everworld",
            "note": "Correctly identified the author but returned the series asked about, not other series."
        },
        "multi_agent": {
            "rewritten_query": "K.A. Applegate other book series besides Animorphs works",
            "retrieved_chunks": [
                "K.A. Applegate, author of Animorphs, also wrote the Everworld and Remnants series.",
                "After Animorphs, K.A. Applegate launched Everworld, a fantasy series for young adults.",
                "Everworld is a series by K.A. Applegate published after Animorphs concluded.",
            ],
            "answer": "Everworld",
            "correct": True,
            "ground_truth": "Everworld",
            "note": "Rewrite shifted focus from Animorphs itself to *other* works by the same author."
        }
    },
]

CUSTOM_TRACE = {
    "dataset": "SQuAD",
    "question": "Which NFL team represented the AFC at Super Bowl 50?",
    "single_agent": {
        "retrieved_chunks": [
            "Super Bowl 50 determined the champion of the National Football League for the 2015 season.",
            "The Denver Broncos won Super Bowl 50 by a score of 24 to 10.",
            "The Carolina Panthers represented the NFC at Super Bowl 50.",
        ],
        "answer": "Denver Broncos",
        "correct": True,
        "ground_truth": "Denver Broncos",
        "note": "Both agents succeed here — simple extractive question, no rewriting needed."
    },
    "multi_agent": {
        "rewritten_query": "AFC champion team Super Bowl 50 American Football Conference representative",
        "retrieved_chunks": [
            "The Denver Broncos, as the AFC champions, represented the American Football Conference at Super Bowl 50.",
            "Denver Broncos defeated Carolina Panthers 24–10 as the AFC representative at Super Bowl 50.",
            "Super Bowl 50: AFC's Denver Broncos vs NFC's Carolina Panthers.",
        ],
        "answer": "Denver Broncos",
        "correct": True,
        "ground_truth": "Denver Broncos",
        "note": "Same result — shows Multi-Agent doesn't sacrifice accuracy when Single-Agent already succeeds."
    }
}


def show():
    st.markdown("""
    <div class="platform-header">
        <div class="platform-title">🔍 Retrieval Trace Viewer</div>
        <div class="platform-sub">Watch how Multi-Agent RAG rewrites your query and retrieves better chunks — step by step</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Query selector ────────────────────────────────────────────────────────
    col_sel, col_ds = st.columns([3, 1])
    with col_ds:
        dataset_filter = st.selectbox("Dataset", ["All", "SQuAD", "HotpotQA"])
    with col_sel:
        filtered = [t for t in TRACES if dataset_filter == "All" or t["dataset"] == dataset_filter]
        options = [f"[{t['dataset']}] {t['question']}" for t in filtered]
        selected_label = st.selectbox("Select a query from the benchmark", options)

    trace = filtered[options.index(selected_label)]

    st.markdown("---")

    # ── Side-by-side trace ────────────────────────────────────────────────────
    col_sa, col_ma = st.columns(2)

    with col_sa:
        st.markdown("""
        <div style="background:#1e1e2e; border:1px solid #2d2d3d; border-top: 3px solid #a371f7;
                    border-radius:10px; padding:1rem 1.25rem; margin-bottom:1rem;">
            <div style="color:#a371f7; font-size:0.72rem; letter-spacing:0.1em; text-transform:uppercase; font-weight:600;">
                Single-Agent RAG
            </div>
            <div style="color:#8b949e; font-size:0.78rem; margin-top:4px;">
                Direct retrieval · No query rewriting
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Step 1 - Original query
        st.markdown("**Step 1 — Query**")
        st.markdown(f"""
        <div style="background:#161622; border:1px solid #2d2d3d; border-radius:8px;
                    padding:0.75rem 1rem; font-family:'JetBrains Mono',monospace; font-size:0.82rem; color:#e6edf3;">
            {trace['question']}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Step 2 - Retrieved chunks
        st.markdown("**Step 2 — Retrieved Chunks (top-k=3)**")
        for i, chunk in enumerate(trace["single_agent"]["retrieved_chunks"], 1):
            st.markdown(f"""
            <div style="background:#161622; border:1px solid #2d2d3d; border-left:3px solid #3d3d4d;
                        border-radius:6px; padding:0.65rem 0.9rem; margin-bottom:0.5rem;
                        font-size:0.8rem; color:#c9d1d9; line-height:1.6;">
                <span style="color:#8b949e; font-size:0.7rem;">chunk {i}</span><br>
                {chunk}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Step 3 - Answer
        sa = trace["single_agent"]
        ans_color = "#3fb950" if sa["correct"] else "#f85149"
        ans_icon = "✅" if sa["correct"] else "❌"
        st.markdown("**Step 3 — Final Answer**")
        st.markdown(f"""
        <div style="background:#161622; border:1px solid {ans_color}40;
                    border-radius:8px; padding:0.85rem 1rem;">
            <div style="color:{ans_color}; font-size:1.2rem; font-weight:700; margin-bottom:4px;">
                {ans_icon} {sa['answer']}
            </div>
            <div style="color:#8b949e; font-size:0.75rem;">
                Ground truth: <span style="color:#e6edf3;">{sa['ground_truth']}</span>
            </div>
            <div style="color:#8b949e; font-size:0.74rem; margin-top:6px; line-height:1.5;">
                {sa['note']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_ma:
        st.markdown("""
        <div style="background:#1a2e1a; border:1px solid #2d3d2d; border-top: 3px solid #3fb950;
                    border-radius:10px; padding:1rem 1.25rem; margin-bottom:1rem;">
            <div style="color:#3fb950; font-size:0.72rem; letter-spacing:0.1em; text-transform:uppercase; font-weight:600;">
                Multi-Agent RAG (LangGraph)
            </div>
            <div style="color:#8b949e; font-size:0.78rem; margin-top:4px;">
                Rewrite → Retrieve → Answer · 3-node StateGraph
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Step 1 - Original query
        st.markdown("**Step 1 — Original Query**")
        st.markdown(f"""
        <div style="background:#161622; border:1px solid #2d2d3d; border-radius:8px;
                    padding:0.75rem 1rem; font-family:'JetBrains Mono',monospace; font-size:0.82rem; color:#e6edf3;">
            {trace['question']}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Step 1b - Rewritten query (the key difference)
        ma = trace["multi_agent"]
        st.markdown("**Step 1b — Rewrite Node** *(LangGraph)*")
        st.markdown(f"""
        <div style="background:#1a2e1a; border:1px solid #3fb95040;
                    border-radius:8px; padding:0.75rem 1rem; margin-bottom:0.5rem;">
            <div style="color:#3fb950; font-size:0.7rem; font-weight:600; margin-bottom:4px; letter-spacing:0.05em;">
                REWRITTEN QUERY
            </div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.82rem; color:#e6edf3;">
                {ma['rewritten_query']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Step 2 - Retrieved chunks
        st.markdown("**Step 2 — Retrieved Chunks (top-k=3)**")
        for i, chunk in enumerate(ma["retrieved_chunks"], 1):
            st.markdown(f"""
            <div style="background:#161622; border:1px solid #2d2d3d; border-left:3px solid #3fb950;
                        border-radius:6px; padding:0.65rem 0.9rem; margin-bottom:0.5rem;
                        font-size:0.8rem; color:#c9d1d9; line-height:1.6;">
                <span style="color:#8b949e; font-size:0.7rem;">chunk {i}</span><br>
                {chunk}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Step 3 - Answer
        ans_color = "#3fb950" if ma["correct"] else "#f85149"
        ans_icon = "✅" if ma["correct"] else "❌"
        st.markdown("**Step 3 — Final Answer**")
        st.markdown(f"""
        <div style="background:#161622; border:1px solid {ans_color}40;
                    border-radius:8px; padding:0.85rem 1rem;">
            <div style="color:{ans_color}; font-size:1.2rem; font-weight:700; margin-bottom:4px;">
                {ans_icon} {ma['answer']}
            </div>
            <div style="color:#8b949e; font-size:0.75rem;">
                Ground truth: <span style="color:#e6edf3;">{ma['ground_truth']}</span>
            </div>
            <div style="color:#8b949e; font-size:0.74rem; margin-top:6px; line-height:1.5;">
                {ma['note']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── LangGraph state flow diagram ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### LangGraph StateGraph — Multi-Agent Execution Flow")

    st.markdown(f"""
    <div style="background:#161622; border:1px solid #2d2d3d; border-radius:12px; padding:1.5rem 2rem;">
        <div style="display:flex; align-items:center; gap:0; justify-content:center; flex-wrap:wrap;">
            <div style="text-align:center; padding:0.75rem 1.2rem; background:#1e1e2e;
                        border:1px solid #a371f7; border-radius:8px; min-width:120px;">
                <div style="color:#a371f7; font-size:0.7rem; font-weight:600; letter-spacing:0.08em;">INPUT</div>
                <div style="color:#e6edf3; font-size:0.85rem; margin-top:4px; font-family:monospace;">question</div>
            </div>
            <div style="color:#3d3d4d; font-size:1.5rem; padding:0 8px;">→</div>
            <div style="text-align:center; padding:0.75rem 1.2rem; background:#1a2e1a;
                        border:1px solid #3fb950; border-radius:8px; min-width:130px;">
                <div style="color:#3fb950; font-size:0.7rem; font-weight:600; letter-spacing:0.08em;">NODE 1</div>
                <div style="color:#e6edf3; font-size:0.85rem; margin-top:4px; font-family:monospace;">rewrite()</div>
                <div style="color:#8b949e; font-size:0.7rem; margin-top:2px;">→ rewritten_query</div>
            </div>
            <div style="color:#3d3d4d; font-size:1.5rem; padding:0 8px;">→</div>
            <div style="text-align:center; padding:0.75rem 1.2rem; background:#1a2e1a;
                        border:1px solid #3fb950; border-radius:8px; min-width:130px;">
                <div style="color:#3fb950; font-size:0.7rem; font-weight:600; letter-spacing:0.08em;">NODE 2</div>
                <div style="color:#e6edf3; font-size:0.85rem; margin-top:4px; font-family:monospace;">retrieve()</div>
                <div style="color:#8b949e; font-size:0.7rem; margin-top:2px;">→ context</div>
            </div>
            <div style="color:#3d3d4d; font-size:1.5rem; padding:0 8px;">→</div>
            <div style="text-align:center; padding:0.75rem 1.2rem; background:#1a2e1a;
                        border:1px solid #3fb950; border-radius:8px; min-width:130px;">
                <div style="color:#3fb950; font-size:0.7rem; font-weight:600; letter-spacing:0.08em;">NODE 3</div>
                <div style="color:#e6edf3; font-size:0.85rem; margin-top:4px; font-family:monospace;">answer()</div>
                <div style="color:#8b949e; font-size:0.7rem; margin-top:2px;">→ answer</div>
            </div>
            <div style="color:#3d3d4d; font-size:1.5rem; padding:0 8px;">→</div>
            <div style="text-align:center; padding:0.75rem 1.2rem; background:#1e1e2e;
                        border:1px solid #58a6ff; border-radius:8px; min-width:120px;">
                <div style="color:#58a6ff; font-size:0.7rem; font-weight:600; letter-spacing:0.08em;">OUTPUT</div>
                <div style="color:#e6edf3; font-size:0.85rem; margin-top:4px; font-family:monospace;">answer</div>
            </div>
        </div>
        <div style="margin-top:1rem; font-size:0.78rem; color:#8b949e; text-align:center;">
            TypedDict state: <span style="color:#e6edf3; font-family:monospace;">GraphState(question, rewritten_query, context, answer)</span>
            · FAISS top-k=3 · Groq gpt-oss-120b
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Try your own query note ───────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("💡 **Tip:** The traces above use real data from the benchmark notebooks. To run live traces against your own FAISS index, connect the Groq API key in the sidebar and run the full pipeline.")
