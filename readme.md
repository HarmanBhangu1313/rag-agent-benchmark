# AI Agent Reliability & Benchmarking Platform

> *Companies deploying AI agents lack tools to evaluate retrieval reliability and preprocessing robustness. Uptiq makes that visible.*

A research platform for understanding **when and why RAG systems fail** — built on real experimentation across two QA datasets, two architectures, and systematic ablation studies. This is not just a benchmark. It is an end-to-end reliability analysis tool with a live dashboard, retrieval trace viewer, and failure attribution engine.

---

##  Core Research Finding

> **Preprocessing strategy can completely flip the benchmark winner.**

On SQuAD without chunking, Single-Agent RAG leads (48% vs 47% EM). Apply chunking, and Multi-Agent takes over (40% vs 32% EM). The winning architecture is not fixed — it is conditional on how you preprocess your corpus. Any RAG evaluation that reports architecture comparisons without controlling for preprocessing is incomplete.

```
Performance hierarchy:
  Dataset Complexity  →  biggest impact on scores
  Preprocessing       →  second biggest (can flip the winner)
  Architecture        →  consistent but smaller impact
```

---

## 📊 Benchmark Results

### With Chunking — Cross-Dataset

| System | Dataset | EM | F1 |
|---|---|---|---|
| Single-Agent | SQuAD | 32.0% | 37.1% |
| **Multi-Agent** | **SQuAD** | **40.0%** | **45.0%** |
| Single-Agent | HotpotQA | 30.0% | 41.1% |
| **Multi-Agent** | **HotpotQA** | **37.0%** | **47.6%** |

### Chunking Ablation — SQuAD (The Key Experiment)

| System | Chunking | EM | F1 | EM Drop |
|---|---|---|---|---|
| Single-Agent | ❌ None | 48.0% | 55.3% | — |
| Multi-Agent | ❌ None | 47.0% | 52.4% | — |
| Single-Agent | ✅ 300/50 | 32.0% | 37.1% | **−16.0%** |
| Multi-Agent | ✅ 300/50 | 40.0% | 45.0% | **−7.0%** |

Multi-Agent drops only half as much — query rewriting buffers against retrieval degradation from chunking.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                                                                  │
│  ┌───────────────────┐     ┌───────────────────────────────┐    │
│  │  Single-Agent RAG │     │  Multi-Agent RAG (LangGraph)  │    │
│  │                   │     │                               │    │
│  │  Query            │     │  Query                        │    │
│  │    ↓              │     │    ↓                          │    │
│  │  FAISS Retriever  │     │  [rewrite node]  ← LLM        │    │
│  │  (top-k=3)        │     │    ↓                          │    │
│  │    ↓              │     │  FAISS Retriever (top-k=3)    │    │
│  │  LLM → Answer     │     │    ↓                          │    │
│  │                   │     │  [answer node] ← LLM          │    │
│  └───────────────────┘     └───────────────────────────────┘    │
│                                                                  │
│  Datasets: SQuAD (1000q) + HotpotQA (1000q)                     │
│  Embeddings: all-MiniLM-L6-v2 · Vector DB: FAISS CPU            │
│  LLM: gpt-oss-120b via Groq · Eval: EM + F1                     │
└─────────────────────────────────────────────────────────────────┘
```

### LangGraph StateGraph — Multi-Agent Pipeline

```python
class GraphState(TypedDict):
    question:        str   # original user query
    rewritten_query: str   # LLM-improved search query
    context:         str   # retrieved FAISS chunks
    answer:          str   # final generated answer

# 3-node DAG: rewrite → retrieve → answer
graph.add_node("rewrite",   rewrite_query)   # vocabulary expansion
graph.add_node("retrieve",  retrieve_docs)   # FAISS top-k search
graph.add_node("answer",    generate_answer) # span extraction
```

---

## 🖥️ Dashboard

Run the Streamlit platform locally:

```bash
pip install -r requirements_dashboard.txt
streamlit run dashboard/app.py
```

### Pages

| Page | What it shows |
|---|---|
| **Benchmark Comparison** | EM/F1 bar charts, ablation study, radar profile, full results table |
| **Retrieval Trace Viewer** | Query → rewrite → chunks → answer, side-by-side SA vs MA |
| **Chunking Impact** | Performance curves vs chunk size, the crossover visualization |
| **Failure Analysis** | 5 case studies where SA failed, MA succeeded, with root cause |

---

## 📁 Repository Structure

```
/
├── dashboard/
│   ├── app.py                     # Streamlit entry point + navigation
│   ├── pages/
│   │   ├── benchmark.py           # Benchmark comparison page
│   │   ├── trace_viewer.py        # Retrieval Trace Viewer
│   │   ├── chunking_impact.py     # Chunking impact explorer
│   │   └── failure_analysis.py    # Failure analysis cases
│   └── requirements_dashboard.txt
├── notebooks/
│   ├── Single_agent_Squad.ipynb
│   ├── Single_agent_Hotpot.ipynb
│   ├── Multi_agent_Squad.ipynb
│   └── Multi_agent_Hotpot.ipynb
├── data/
│   ├── squad_subset.json          # SQuAD: 1000 queries
│   └── hotpot_subset.json         # HotpotQA: 1000 queries
├── evaluation/
│   ├── results.json
│   └── benchmark_chart.png
├── ARCHITECTURE.md
└── requirements.txt
```

---

## ⚙️ Setup

```bash
# Install dependencies
pip install langchain langchain_community langchain_huggingface
pip install langchain_groq langgraph faiss-cpu
pip install datasets transformers sentence-transformers

# Or all at once
pip install -r requirements.txt
```

**API Key — Colab:**
```python
from google.colab import userdata
api_key = userdata.get("groq_api_key")
```

**API Key — Local:**
```bash
echo "GROQ_API_KEY=your_key_here" > .env
```

---

## 🔎 Datasets

### SQuAD
Single-hop extractive QA. Answers are short spans within one paragraph. 1000 queries.
Chunking config: `chunk_size=300, chunk_overlap=50` → embedded with MiniLM into FAISS.

### HotpotQA
Multi-hop reasoning. Each question has 10 supporting documents. 1000 queries.
Raw docs: 10,000 · After chunking (`chunk_size=500, overlap=100`): **16,809 chunks**
FAISS index build: ~12 min first run, ~2 sec on reload.

---

## 💡 Key Discoveries

**1. Chunking flipped the benchmark winner**
Without chunking, Single-Agent won on SQuAD. With chunking, Multi-Agent took the lead.
The winning architecture depends on preprocessing, not just design.

**2. Query rewriting is a reliability buffer**
After chunking, Single-Agent EM dropped −16%; Multi-Agent dropped only −7%.
The rewrite node partially compensates for noisier retrieval.

**3. F1 > EM gap reveals partial correctness**
On HotpotQA, Single-Agent F1 (41.1%) >> EM (30%) — the agent finds partial answers
but fails exact span extraction. Multi-Agent closes this gap with better chunk precision.

**4. Dataset complexity dominates everything**
Moving from SQuAD to HotpotQA is a larger performance drop than any architecture or
preprocessing choice. Task complexity is the primary bottleneck.

**5. Vocabulary mismatch is the #1 silent failure mode**
Every failure case traces to the same root: the original query vocabulary doesn't match
the source chunk vocabulary. Query rewriting is a low-cost fix with high reliability ROI.

---

## 🧰 Tech Stack

| Component | Tool |
|---|---|
| Agent Framework | LangChain + LangGraph |
| Orchestration | LangGraph `StateGraph` |
| Embedding Model | `all-MiniLM-L6-v2` (HuggingFace) |
| Vector Store | FAISS (CPU) |
| LLM | `gpt-oss-120b` via Groq |
| Dashboard | Streamlit + Plotly |
| Evaluation | Custom EM + F1 with text normalization |
| Environment | Google Colab / Local |

---

## 📌 Conclusion

> *Multi-Agent RAG with query rewriting consistently outperforms Single-Agent when chunking is applied (+8% EM on SQuAD, +7% EM on HotpotQA). However, this advantage is conditional on preprocessing strategy — without chunking, both architectures perform comparably. The primary bottleneck in RAG systems is retrieval quality, shaped by preprocessing and query formulation, not the generation step.*

---

## Author

**Harman Bhangu** — B.Tech, IIT BHU  
AI/ML Engineering · LangChain · LangGraph · FAISS · RAG Systems
