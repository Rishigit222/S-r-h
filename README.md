# 🌟 s@r@h — Self-Adaptive Reasoning & Retrieval Autonomous Host

A production-grade, CPU-optimized, enterprise AI system combining:
1. **Self-Healing RAG Engine** with Hybrid Retrieval (ChromaDB + BM25Okapi + Cross-Encoder Reranking)
2. **Knowledge Graph RAG (GraphRAG)** for multi-hop entity relationship traversal
3. **Persistent Memory Agent** with automatic user preference and fact extraction
4. **Real-Time Quality & Hallucination Drift Platform** with live telemetry
5. **Tri-Guard Rail Architecture** with pre-relevance filtering, NLI faithfulness scoring, and citation enforcement
6. **Enterprise Security Shield** with prompt injection blocker and PII redactor
7. **Semantic Response Cache** delivering instant (<10ms) responses

---

## 🏛️ System Architecture

```
                                    User Query
                                        │
    ┌───────────────────────────────────┴───────────────────────────────────┐
    ▼                                                                       ▼
[Pillar 6: Security Guard]                                    [Pillar 5: Safety & Anti-Sycophancy]
 (Jailbreak Detection + PII Redactor)                          (Toxic intent & bias filter)
    │                                                                       │
    └───────────────────────────────────┬───────────────────────────────────┘
                                        ▼
                       [Pillar 7: Semantic Response Cache]
                      ⚡ Hit (sim >= 0.94) → Response in <10ms
                                        │ (Miss)
                                        ▼
                       [Pillar 4: Persistent Memory Agent]
                      (SQLite Episodic & User Fact Store)
                                        │
                                        ▼
                     [Pillar 2: Multi-Hop Query Decomposer]
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
      [Pillar 10: Hybrid Search]               [Pillar 10: GraphRAG Traversal]
   (ChromaDB Dense + BM25 Sparse)                 (Entity-Relation Triples)
                    │                                       │
                    └───────────────────┬───────────────────┘
                                        ▼
                        [Pillar 10: Cross-Encoder Reranker]
                             (ms-marco-MiniLM-L6-v2)
                                        │
                                        ▼
                        [Pillar 1: Pre-Relevance Gate]
                                        │ (If low relevance)
                                        ├──────► [Pillar 3: Dynamic Web Grounding]
                                        │        (Wikipedia REST API live summary)
                                        ▼
                            [Pillar 1: LLM Generation]
                          (Groq / Google / Ollama / Local)
                                        │
                                        ▼
                      [Pillar 1: Faithfulness & Citations]
                         (NLI Cross-Encoder + [1], [2])
                                        │
                                        ▼
                     [Pillar 8: Decision Trace & Observability]
                                        │
                                        ▼
                      [Pillar 9: Real-Time Telemetry & Drift]
```

---

## 🚀 Quickstart

### 1. Installation

```bash
cd c:\Users\rishi\OneDrive\Desktop\self-healing-rag
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start the Backend API (FastAPI)

```bash
python -m uvicorn src.api.main:app --reload --port 8000
```
Interactive Swagger Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 3. Start the Web Dashboard (Streamlit)

```bash
streamlit run src/ui/app.py --server.port 8501
```
Open Dashboard: [http://127.0.0.1:8501](http://127.0.0.1:8501)

---

## 🧪 Automated Testing & Evaluation

Run unit and integration tests across all 10 pillars:
```bash
pytest tests/ -v
```

Run automated Golden Dataset quality benchmark:
```bash
python -m src.evaluation.eval_runner
```
