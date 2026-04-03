# RAG Agent Benchmarking — Single vs Multi-Agent

Benchmarking **Single-Agent RAG** vs **Multi-Agent RAG** across two QA datasets (SQuAD & HotpotQA) using LangChain, LangGraph, FAISS, and Groq-hosted LLMs.

---

## Repository Structure

```
repo/
├── Notebooks/
│   ├── Single_agent_Squad.ipynb       # Single Agent RAG on SQuAD
│   ├── Single_agent_Hotpot.ipynb      # Single Agent RAG on HotpotQA
│   ├── Multi_agent_Squad.ipynb        # Multi Agent RAG on SQuAD (LangGraph)
│   └── Multi_agent_Hotpot.ipynb       # Multi Agent RAG on HotpotQA (LangGraph)
├── Data/
│   ├── squad_subset.json              # SQuAD subset (1000 queries)
│   └── hotpot_subset.json             # HotpotQA subset (1000 queries)
├── evaluation/
│   ├── benchmark_chart.png                  
│   ├── results.json            
│   
├── README.md
├── ARCHITECTURE.md
├── DEMO_VIDEO
└── requirements.txt
```

---

## Agent Setup

### Single-Agent RAG

| | |
|---|---|
| **Input** | User question |
| **Output** | Answer string |
| **Tools** | FAISS retriever (top-k=3) |

**Pipeline:**
```
User Query → FAISS Retriever → Context → LLM → Answer
```
One LLM directly retrieves context from FAISS and generates the answer in a single step.

**Prompt used:**
```
You are a question answering system.
Use ONLY the context to answer.
Rules:
- Return ONLY the short answer span.
- Do NOT explain.
- Do NOT write full sentences.
- Answer must be a phrase from the context.
```

---

### Multi-Agent RAG (LangGraph)

| | |
|---|---|
| **Input** | User question |
| **Output** | Answer string |
| **Tools** | FAISS retriever (top-k=3) + Query Rewriter LLM |

**Pipeline:**
```
User Query → [Rewrite Node] → [Retrieve Node] → [Answer Node] → Answer
```

Built using **LangGraph `StateGraph`** with a typed state and three nodes:

```python
class GraphState(TypedDict):
    question: str
    rewritten_query: str
    context: str
    answer: str
```

| Node | Role |
|---|---|
| `rewrite` | Rewrites the user query to improve retrieval |
| `retrieve` | Semantic search over FAISS using rewritten query |
| `answer` | Generates final answer from retrieved context |

---

##  Datasets

### 1. SQuAD (Stanford Question Answering Dataset)
- **Source:** Stanford NLP via HuggingFace `datasets`
- **Size:** 1000 queries
- **Type:** Single hop extractive QA — answers are short spans within one paragraph
- **Fields:** `context`, `question`, `answers.text[0]`
- **Preprocessing:**
  - Contexts split using `RecursiveCharacterTextSplitter`
  - `chunk_size=300`, `chunk_overlap=50`
  - Embedded with `all-MiniLM-L6-V2` and stored in FAISS

### 2. HotpotQA
- **Source:** CMU via HuggingFace `datasets`
- **Size:** 1000 queries
- **Type:** Multi hop reasoning QA — each question has 10 supporting documents
- **Fields:** `context.title`, `context.sentences`, `question`, `answer`
- **Preprocessing:**
  - Titles prepended to sentences for each of the 10 documents per question
  - All documents flattened → 10,000 raw docs
  - Split: `chunk_size=500`, `chunk_overlap=100` → **16,809 chunks**
  - Embedded with `all-MiniLM-L6-V2` and stored in FAISS
  - FAISS index saved locally (~12 min first build, ~2 sec reload after)

---

## Benchmark Design

### Metrics

**Exact Match (EM)** — Returns 1 if normalized prediction exactly matches ground truth, else 0.

**F1 Score** — Token level overlap between prediction and ground truth. Harmonic mean of precision and recall. Handles partial matches.

Both use text normalization:
```python
def normalize_text(text):
    text = text.lower()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    return ' '.join(text.split())
```

### Reproducible Pipeline
```
Load Data → Preprocess → Chunk → Embed → FAISS → Run Agent → Collect Predictions → Evaluate → Compare
```

Each notebook follows this exact pipeline end to end, independently and reproducibly.

---

##  Setup & Installation


### Install Dependencies

```bash
pip install langchain langchain_community langchain_huggingface
pip install langchain_groq langgraph faiss-cpu
pip install datasets transformers sentence-transformers
```

Or:
```bash
pip install -r requirements.txt
```

### API Key Setup — Google Colab

Add your Groq API key to Colab Secrets with the name `groq_api_key`, then load it in the notebook:

```python
from google.colab import userdata
api_key = userdata.get("groq_api_key")
```

### API Key Setup — Local

Create a `.env` file in the root:
```
GROQ_API_KEY=your_key_here
```

---

## Running the Benchmark

### Step 1 — Add Data
Place `squad_subset.json` and `hotpot_subset.json` inside the `data/` directory.

### Step 2 — Run Notebooks
Open any notebook from `src/` in Google Colab and run all cells top to bottom:

| Notebook | Agent | Dataset |
|---|---|---|
| `Single_agent_Squad.ipynb` | Single-Agent | SQuAD |
| `Single_agent_Hotpot.ipynb` | Single-Agent | HotpotQA |
| `Multi_agent_Squad.ipynb` | Multi-Agent | SQuAD |
| `Multi_agent_Hotpot.ipynb` | Multi-Agent | HotpotQA |

> ⚠️ **HotpotQA notebooks only:** FAISS index takes ~12 minutes to build on first run. After that it saves automatically and loads in ~2 seconds on every subsequent run.

### Step 3 — Evaluate
EM and F1 scores are computed automatically at the end of each notebook using the `evaluate()` function.

---

## Tech Stack

| Component | Tool |
|---|---|
| Agent Framework | LangChain + LangGraph |
| Embedding Model | `all-MiniLM-L6-V2` (HuggingFace) |
| Vector Store | FAISS (CPU) |
| LLM | `openai/gpt-oss-120b` via Groq |
| Retrieval | Semantic top-k search (k=3) |
| Environment | Google Colab |




---

## Results

### With Chunking — Cross Dataset Comparison

| System | Dataset | Exact Match | F1 Score |
|---|---|---|---|
| Single-Agent | SQuAD | 32% | 37.1% |
| **Multi-Agent** | **SQuAD** | **40%** | **45.0%** |
| Single-Agent | HotpotQA | 30% | 41.1% |
| **Multi-Agent** | **HotpotQA** | **37%** | **47.6%** |

### Effect of Chunking on SQuAD (Ablation)

| System | Chunking | Exact Match | F1 Score |
|---|---|---|---|
| Single-Agent | ❌ No | 48% | 55.3% |
| Multi-Agent | ❌ No | 47% | 52.4% |
| Single-Agent | ✅ Yes | 32% | 37.1% |
| Multi-Agent | ✅ Yes | 40% | 45.0% |

---

##  Key Discoveries

### Discovery 1 — Chunking flipped the winner
Without chunking, **Single Agent won on SQuAD** (48% vs 47% EM). After introducing chunking, **Multi Agent took the lead** (40% vs 32% EM). The winner between architectures is not fixed — it depends entirely on the preprocessing strategy.

### Discovery 2 — Chunking hurts SQuAD but is necessary for HotpotQA
SQuAD paragraphs are pre curated short spans where the answer always lives in one paragraph. Chunking fragments these natural boundaries, causing both agents to drop significantly. For HotpotQA, chunking is essential without it, 10 long documents per question would overwhelm the retriever.

### Discovery 3 — Query rewriting buffers against retrieval degradation
After chunking, Single Agent EM dropped **-16%** while Multi-Agent only dropped **-7%**. The query rewriting step partially compensates for noisier retrieval, making the multi-agent architecture more robust to preprocessing changes.

### Discovery 4 — F1 tells a different story than EM on HotpotQA
On HotpotQA, Single Agent F1 (41.1%) is notably higher than its EM (30%), suggesting the agent is finding partially correct answers getting some words right but missing exact span boundaries. Multi Agent F1 (47.6%) improves further, confirming query rewriting helps surface more relevant content even when exact extraction fails.

### Discovery 5 — Dataset complexity dominates everything
Both agents score significantly lower on HotpotQA than SQuAD across all configurations. The dataset's inherent complexity (multi hop reasoning across 10 documents vs single paragraph extraction) is a stronger performance driver than either architecture or preprocessing choice.

---

## Insights & Conclusions

### Insight 1 — Architecture choice must match task complexity
Multi agent with query rewriting is justified for complex multi-hop tasks like HotpotQA. For simple extractive tasks like SQuAD without chunking, a single-agent is equally effective and more efficient.

### Insight 2 — Preprocessing is as important as architecture
The chunking ablation study reveals that preprocessing decisions can flip benchmark winners entirely. Any RAG evaluation that reports architecture comparisons without controlling for preprocessing is incomplete.

### Insight 3 — Performance hierarchy
```
Dataset Complexity     →  biggest impact on scores
Preprocessing          →  second biggest impact
Architecture           →  consistent but smaller impact
```

### Final Conclusion
> *Multi-agent RAG with query rewriting consistently outperforms single agent RAG when chunking is applied (+8% EM on SQuAD, +7% EM on HotpotQA). However, this advantage is conditional on preprocessing strategy without chunking, both architectures perform comparably. The key takeaway is that retrieval quality, shaped by both preprocessing and query formulation, is the primary bottleneck in RAG systems, not the generation step.*

-- 

##  Author

Harman Bhangu
IIT BHU
B.Tech Undergraduate  