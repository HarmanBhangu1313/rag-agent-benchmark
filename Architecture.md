# 🏗️ Architecture

This document describes the architecture of both RAG systems benchmarked in this project.

---

## 1. Single-Agent RAG

### Overview
A straightforward RAG pipeline where a single LLM is responsible for both understanding the question and generating the answer using retrieved context. No orchestration layer exists  the pipeline runs linearly in one pass.

### Pipeline
```
User Query
    │
    ▼
┌─────────────────────────┐
│     FAISS Retriever     │  ← Semantic search, top-k=3 chunks
│   (all-MiniLM-L6-V2)   │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│    Context Assembly     │  ← Top 3 chunks joined as context
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│   LLM (gpt-oss-120b)    │  ← Prompt + Context → Answer
└─────────────────────────┘
    │
    ▼
  Answer
```

### Components

| Component | Details |
|---|---|
| Embedding Model | `all-MiniLM-L6-V2` (HuggingFace) |
| Vector Store | FAISS (CPU, flat index) |
| Retrieval | Semantic similarity, top-k=3 |
| LLM | `openai/gpt-oss-120b` via Groq |
| Temperature | 0 (deterministic) |

### Prompt Design
```
You are a question answering system.
Use ONLY the context to answer.
Rules:
- Return ONLY the short answer span.
- Do NOT explain.
- Do NOT write full sentences.
- Answer must be a phrase from the context.

Context: {retrieved chunks}
Question: {user question}
Answer:
```

### Data Flow
1. User question is sent directly to the FAISS retriever
2. Top 3 most similar chunks are retrieved
3. Chunks are joined into a single context string
4. Context + question is passed to the LLM via a structured prompt
5. LLM returns a short answer span

### Strengths
- Simple and fast — single LLM call per query
- Low latency, low cost
- Works well on simple, well-formed questions

### Weaknesses
- Query is used as-is — no optimization for retrieval
- Struggles when question phrasing doesn't match document vocabulary
- No reasoning step for complex multi-hop questions

---

## 2. Multi-Agent RAG (LangGraph)

### Overview
An orchestrated RAG pipeline built using **LangGraph `StateGraph`**. The system splits the QA task across three specialized nodes — a query rewriter, a retriever, and an answer generator — each responsible for a distinct step. State is passed through the graph explicitly via a typed dict.

### Pipeline
```
User Query
    │
    ▼
┌─────────────────────────┐
│     Rewrite Node        │  ← LLM rewrites query for better retrieval
│   (gpt-oss-120b)        │
└─────────────────────────┘
    │
    ▼  rewritten_query
┌─────────────────────────┐
│     Retrieve Node       │  ← FAISS semantic search, top-k=3
│   (all-MiniLM-L6-V2)   │
└─────────────────────────┘
    │
    ▼  context
┌─────────────────────────┐
│      Answer Node        │  ← LLM generates answer from context
│   (gpt-oss-120b)        │
└─────────────────────────┘
    │
    ▼
  Answer
```

### LangGraph State
```python
class GraphState(TypedDict):
    question: str           # original user question
    rewritten_query: str    # improved query from rewrite node
    context: str            # retrieved chunks from FAISS
    answer: str             # final generated answer
```

### Graph Definition
```python
graph = StateGraph(GraphState)

graph.add_node("rewrite", rewrite_query)
graph.add_node("retrieve", retrieve_docs)
graph.add_node("answer", generate_answer)

graph.set_entry_point("rewrite")

graph.add_edge("rewrite", "retrieve")
graph.add_edge("retrieve", "answer")
graph.add_edge("answer", END)

app = graph.compile()
```

### Node Details

#### Node 1 — Rewrite
- **Input:** `question`
- **Output:** `rewritten_query`
- **Role:** LLM rewrites the original question into a more retrieval-friendly search query
- **Prompt:**
```
Rewrite the question to improve document retrieval.
Return ONLY the improved search query.

Question: {question}
Rewritten query:
```

#### Node 2 — Retrieve
- **Input:** `rewritten_query`
- **Output:** `context`
- **Role:** Performs semantic similarity search over FAISS using the rewritten query, retrieves top-3 chunks and assembles them into context

#### Node 3 — Answer
- **Input:** `question` + `context`
- **Output:** `answer`
- **Role:** LLM generates final answer using original question and retrieved context
- **Prompt:** Same structured QA prompt as Single-Agent

### Components

| Component | Details |
|---|---|
| Orchestration | LangGraph `StateGraph` |
| Embedding Model | `all-MiniLM-L6-V2` (HuggingFace) |
| Vector Store | FAISS (CPU, flat index) |
| Retrieval | Semantic similarity on rewritten query, top-k=3 |
| LLM (Rewrite) | `openai/gpt-oss-120b` via Groq |
| LLM (Answer) | `openai/gpt-oss-120b` via Groq |
| Temperature | 0 (deterministic) |

### Data Flow
1. User question enters the graph at the `rewrite` node
2. LLM rewrites the question into an optimized retrieval query
3. Rewritten query is passed to the `retrieve` node
4. FAISS returns top 3 semantically similar chunks
5. Original question + assembled context passed to `answer` node
6. LLM generates and returns the final answer span

### Strengths
- Query rewriting improves retrieval precision, especially for vague or complex questions
- Modular — each node can be independently improved or swapped
- More robust to preprocessing changes (chunking ablation showed only -7% EM drop vs -16% for single-agent)
- LangGraph state is explicit and traceable

### Weaknesses
- 2x LLM calls per query → higher latency and cost
- Query rewriting can hurt retrieval on already well-formed questions
- Adds orchestration complexity for simple tasks

---

## Architecture Comparison

| | Single-Agent RAG | Multi-Agent RAG |
|---|---|---|
| **LLM Calls per Query** | 1 | 2 |
| **Query Optimization** | ❌ None | ✅ Rewrite node |
| **Orchestration** | ❌ None | ✅ LangGraph StateGraph |
| **Latency** | Lower | Higher |
| **Cost** | Lower | Higher |
| **Best Dataset** | Simple extractive QA | Multi-hop reasoning QA |
| **EM on SQuAD** | 32% | 40% |
| **EM on HotpotQA** | 30% | 37% |

---

## Vector Store — FAISS Index

Both architectures share the same FAISS index setup:

### SQuAD Index
- **Chunks:** ~1000 (300 char chunks, 50 overlap)
- **Build time:** ~1-2 minutes
- **Index type:** Flat L2

### HotpotQA Index
- **Chunks:** 16,809 (500 char chunks, 100 overlap)
- **Build time:** ~12 minutes (first run only)
- **Index type:** Flat L2
- **Saved locally** and reloaded in ~2 seconds on subsequent runs:
```python
vectorstore.save_local("hotpot_faiss_index")
vectorstore = FAISS.load_local("hotpot_faiss_index", embeddings,
                allow_dangerous_deserialization=True)
```