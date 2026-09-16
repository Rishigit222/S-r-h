# Architecture Audit — Self-Healing RAG Engine Transformation

> **Date**: 2026-09-15
> **Repository**: https://github.com/Rishigit222/S-r-h
> **Current Product**: s@r@h — Enterprise RAG Platform with 3D GraphRAG, Voice Assistant, Meta-Learning & Settings Studio
> **Target Product**: Self-Healing RAG Engine — Autonomous control plane for diagnosing and repairing RAG systems

---

## A. Current Architecture

The system is a **Modular Monolith** using a **Pipeline / Chain Orchestration** pattern.

`api/main.py` (609 LOC) acts as the master orchestrator, routing queries through sequential interceptor gates:

```
Security Audit → Semantic Cache → Query Decomposition
→ Graph + Hybrid Retrieval → Relevance Gate
→ Generation → Guardrails (Faithfulness & Citations)
→ Self-Heal Retry Loop
```

### Module Map

| Directory | Files | Total LOC (approx) | Domain |
|---|---|---|---|
| `src/api/` | `main.py`, `auth.py`, `settings.py` | ~913 | API / Orchestration |
| `src/ingestion/` | `loader.py`, `chunker.py`, `embedder.py` | ~230 | Document Ingestion |
| `src/retrieval/` | `hybrid.py`, `vector_store.py`, `bm25_store.py`, `reranker.py`, `web_grounding.py` | ~331 | Retrieval Pipeline |
| `src/generation/` | `generator.py`, `llm.py`, `prompts.py` | ~215 | LLM Generation |
| `src/guardrails/` | `relevance_gate.py`, `faithfulness_checker.py`, `citation_verifier.py`, `safety_gate.py`, `self_healer.py` | ~299 | Quality Gates & Self-Healing |
| `src/graph/` | `knowledge_graph.py`, `voice_assistant.py` | ~608 | GraphRAG & Voice |
| `src/evolution/` | `self_optimizer.py`, `meta_modifier.py`, `synthetic_trainer.py` | ~426 | Self-Optimization |
| `src/evaluation/` | `eval_runner.py`, `metrics.py` | ~147 | Evaluation |
| `src/observability/` | `drift_monitor.py`, `tracer.py` | ~211 | Telemetry & Monitoring |
| `src/optimization/` | `semantic_cache.py` | ~97 | Performance |
| `src/memory/` | `episodic_memory.py` | ~131 | Conversational Memory |
| `src/security/` | `sanitizer.py` | ~90 | Input Security |
| `src/reasoning/` | `decomposer.py` | ~74 | Query Decomposition |
| `src/models/` | `manager.py` | ~90 | ML Model Management |
| `src/ui/` | `app.py` | ~250 | Streamlit Frontend |
| `src/` | `config.py` | ~73 | Configuration |

**Total**: ~4,185 LOC across ~30 Python files

### Data Layer
- `data/chroma_db/` — Persistent vector database
- `data/sample_docs/` — Raw ingestion documents
- `data/auth_vault.sqlite` — API key/session store
- `data/evolution_history.sqlite` — Mutation/tuning history
- `data/knowledge_graph.sqlite` — Entity-relation triples
- `data/memory.sqlite` — Episodic conversation memory
- `data/telemetry.sqlite` — Observability logs
- `data/user_settings.sqlite` — User preferences

### Frontend
- `static/index.html` — 3D Glassmorphic SPA
- `static/css/styles.css` — Glassmorphic styling
- `static/js/app.js` — Chat UI logic
- `static/js/graph3d.js` — 3D Knowledge Graph Canvas
- `static/js/particles.js` — Decorative particle effects

---

## B. Existing Reusable Components

These components directly align with the Self-Healing RAG Engine architecture:

| Component | File | Relevance | Reuse Quality |
|---|---|---|---|
| Self-Healer | `guardrails/self_healer.py` | **Core** — query rewrite + retry loop | ⭐⭐⭐ High — needs expansion |
| Self-Optimizer | `evolution/self_optimizer.py` | **Core** — autonomous threshold tuning | ⭐⭐⭐ High — needs expansion |
| Meta-Modifier | `evolution/meta_modifier.py` | **Core** — external RAG config auditor | ⭐⭐⭐ High — unique differentiator |
| Faithfulness Checker | `guardrails/faithfulness_checker.py` | **Core** — HHEM entailment evaluation | ⭐⭐⭐ High — direct reuse |
| Relevance Gate | `guardrails/relevance_gate.py` | **Core** — pre-generation quality gate | ⭐⭐⭐ High — direct reuse |
| Citation Verifier | `guardrails/citation_verifier.py` | **Core** — citation grounding check | ⭐⭐⭐ High — direct reuse |
| Drift Monitor | `observability/drift_monitor.py` | **Core** — telemetry + sliding window metrics | ⭐⭐⭐ High — needs expansion |
| Tracer | `observability/tracer.py` | **Core** — step-level execution traces | ⭐⭐⭐ High — needs expansion |
| Eval Runner | `evaluation/eval_runner.py` | **Core** — golden dataset benchmarking | ⭐⭐⭐ High — needs major expansion |
| Eval Metrics | `evaluation/metrics.py` | **Core** — refusal & citation metrics | ⭐⭐⭐ High — needs major expansion |
| Hybrid Retrieval | `retrieval/hybrid.py` | RAG Pipeline — RRF fusion | ⭐⭐⭐ High — direct reuse |
| Vector Store | `retrieval/vector_store.py` | RAG Pipeline — dense retrieval | ⭐⭐⭐ High — direct reuse |
| BM25 Store | `retrieval/bm25_store.py` | RAG Pipeline — sparse retrieval | ⭐⭐⭐ High — direct reuse |
| Reranker | `retrieval/reranker.py` | RAG Pipeline — cross-encoder reranking | ⭐⭐⭐ High — direct reuse |
| Document Loader | `ingestion/loader.py` | RAG Pipeline — multi-format loader | ⭐⭐⭐ High — direct reuse |
| Chunker | `ingestion/chunker.py` | RAG Pipeline — semantic splitting | ⭐⭐⭐ High — direct reuse |
| Embedder | `ingestion/embedder.py` | RAG Pipeline — CPU-optimized embeddings | ⭐⭐⭐ High — direct reuse |
| LLM Client | `generation/llm.py` | RAG Pipeline — multi-provider abstraction | ⭐⭐⭐ High — direct reuse |
| Generator | `generation/generator.py` | RAG Pipeline — context-aware generation | ⭐⭐⭐ High — direct reuse |
| Prompts | `generation/prompts.py` | RAG Pipeline — system prompts | ⭐⭐ Medium — needs revision |
| Config | `config.py` | Infrastructure — centralized settings | ⭐⭐⭐ High — needs expansion |
| Model Manager | `models/manager.py` | Infrastructure — lazy model loading | ⭐⭐⭐ High — direct reuse |
| Semantic Cache | `optimization/semantic_cache.py` | Performance — sub-5ms cache | ⭐⭐ Medium — optional |
| Synthetic Trainer | `evolution/synthetic_trainer.py` | Evaluation — QA pair generation | ⭐⭐⭐ High — feeds eval |
| Web Grounding | `retrieval/web_grounding.py` | Fallback — Wikipedia context | ⭐⭐ Medium — optional |

---

## C. Components to Keep (As-Is or Minor Tweaks)

1. `src/guardrails/faithfulness_checker.py` — Direct reuse
2. `src/guardrails/relevance_gate.py` — Direct reuse
3. `src/guardrails/citation_verifier.py` — Direct reuse
4. `src/retrieval/hybrid.py` — Direct reuse
5. `src/retrieval/vector_store.py` — Direct reuse
6. `src/retrieval/bm25_store.py` — Direct reuse
7. `src/retrieval/reranker.py` — Direct reuse
8. `src/ingestion/loader.py` — Direct reuse
9. `src/ingestion/chunker.py` — Direct reuse
10. `src/ingestion/embedder.py` — Direct reuse
11. `src/generation/llm.py` — Direct reuse
12. `src/generation/generator.py` — Direct reuse
13. `src/models/manager.py` — Direct reuse
14. `src/config.py` — Extend with new settings
15. `scripts/setup_models.py` — Direct reuse

---

## D. Components to Refactor

| Component | Current State | Refactor Target |
|---|---|---|
| `guardrails/self_healer.py` | Simple query rewrite retry (49 LOC) | Full diagnosis → repair planning → sandbox → regression engine |
| `evolution/self_optimizer.py` | Threshold tuner only (215 LOC) | Comprehensive repair strategy selector with rollback |
| `evolution/meta_modifier.py` | External RAG config auditor (127 LOC) | RAG Adapter Layer for external pipeline integration |
| `observability/drift_monitor.py` | Basic SQLite telemetry (148 LOC) | Full observability layer with structured events, metrics, alerts |
| `observability/tracer.py` | Simple step trace (63 LOC) | OpenTelemetry-compatible distributed tracing |
| `evaluation/eval_runner.py` | Single golden-dataset benchmark (104 LOC) | Multi-dimensional evaluation engine (retrieval, answer, faithfulness, citation, latency) |
| `evaluation/metrics.py` | Refusal + citation only (43 LOC) | Comprehensive metric suite with before/after comparison |
| `api/main.py` | 609 LOC monolithic orchestrator | Decompose into Heal Loop Controller + RAG Pipeline Runner |
| `generation/prompts.py` | Chatbot-oriented prompts | Diagnosis/repair-oriented prompts |
| `evolution/synthetic_trainer.py` | Basic QA generator (84 LOC) | Regression test generator for sandbox evaluation |

---

## E. Components to Delete

| Component | File(s) | Reason |
|---|---|---|
| 3D Graph Voice Assistant | `src/graph/voice_assistant.py` | Not core to self-healing engine |
| 3D Graph UI Frontend | `static/js/graph3d.js` | Consumer chatbot feature |
| Particle Effects | `static/js/particles.js` | Decorative, non-functional |
| Glassmorphic Chat UI | `static/index.html`, `static/css/styles.css`, `static/js/app.js` | Consumer chatbot UI |
| Episodic Memory | `src/memory/episodic_memory.py` | Conversational state, not needed for engine |
| User Settings/Personas | `src/api/settings.py` | Subscription tiers, custom personas |
| Auth Vault | `src/api/auth.py` | SaaS auth, not core |
| Safety/Sycophancy Gate | `src/guardrails/safety_gate.py` | Consumer chatbot feature |
| Prompt Injection Defense | `src/security/sanitizer.py` | Consumer-facing security, not core to engine |
| Query Decomposer | `src/reasoning/decomposer.py` | Voice decomposition feature |
| Streamlit UI | `src/ui/app.py` | Will be replaced by ops dashboard |
| Knowledge Graph | `src/graph/knowledge_graph.py` | GraphRAG visualization, not core |

### Data Files to Delete
- `data/auth_vault.sqlite`
- `data/knowledge_graph.sqlite`
- `data/memory.sqlite`
- `data/user_settings.sqlite`

### Tests to Delete
- `tests/test_voice_assistant.py`
- `tests/test_settings.py`
- Parts of `tests/test_sarah.py` (auth, GraphRAG seeds, memory agent)

---

## F. Components to Replace

| Current | Replacement | Reason |
|---|---|---|
| Monolithic `api/main.py` orchestrator | `src/engine/heal_loop.py` — dedicated heal loop controller | Separation of concerns |
| Single self-heal retry | Multi-strategy repair planner with sandbox | Core product requirement |
| SQLite-only telemetry | Structured event bus + metrics store | Production observability |
| Single `eval_runner` | Before/after comparison engine with regression detection | Core product requirement |
| Static frontend | Minimal ops dashboard (FastAPI + Jinja2 or API-only) | Infrastructure tool, not chatbot |

---

## G. Dependency Analysis

### `requirements.txt` — Current Dependencies

| Package | Category | Keep? |
|---|---|---|
| `fastapi` | Backend | ✅ Keep |
| `uvicorn` | Server | ✅ Keep |
| `pydantic` | Validation | ✅ Keep |
| `pydantic-settings` | Config | ✅ Keep |
| `python-dotenv` | Config | ✅ Keep |
| `pypdf` | Ingestion | ✅ Keep |
| `sentence-transformers` | Embeddings | ✅ Keep |
| `torch` | ML Runtime | ✅ Keep |
| `chromadb` | Vector Store | ✅ Keep |
| `rank-bm25` | Sparse Retrieval | ✅ Keep |
| `openai` | LLM Client | ✅ Keep |
| `ragas` | Evaluation | ✅ Keep |
| `deepeval` | Evaluation | ✅ Keep |
| `streamlit` | UI | ❌ Remove (replaced by ops dashboard or API-only) |
| `rich` | CLI Output | ✅ Keep |
| `numpy` | Numerics | ✅ Keep |

### New Dependencies Required

| Package | Purpose |
|---|---|
| `jinja2` | Ops dashboard templates (if HTML dashboard) |
| `httpx` | External RAG adapter HTTP calls |
| `structlog` | Structured logging |
| `tabulate` | CLI metric tables |

---

## H. Technical Debt

1. **Monolithic orchestrator** — `api/main.py` at 609 LOC handles everything from security to generation to self-healing. Must be decomposed.
2. **No rollback mechanism** — Self-optimizer tunes thresholds but cannot revert if results worsen.
3. **No sandbox evaluation** — Repairs are applied directly with no candidate testing.
4. **No before/after comparison** — Evaluation is one-shot, no delta tracking.
5. **Primitive self-healing** — Only query rewrite; no chunking strategy changes, no embedding model swaps, no retrieval weight adjustments.
6. **No structured failure taxonomy** — Failures are not categorized by type (retrieval, grounding, generation, citation, latency, config).
7. **SQLite-only telemetry** — No event streaming, no alerting, no dashboarding.
8. **No external RAG adapter protocol** — `meta_modifier.py` audits configs but has no runtime adapter for external pipelines.
9. **Missing regression detection** — No mechanism to detect if a repair degrades other queries.
10. **No continuous monitoring loop** — System is request-driven, not continuously monitoring.

---

## I. Existing Tests That Remain Useful

| Test File | Relevant Tests | Maps To |
|---|---|---|
| `tests/test_guardrails.py` | Relevance gate pass/block, citation verification, self-healing logic | Core guardrail evaluation |
| `tests/test_retrieval.py` | Document loading, metadata chunking, hybrid retrieval integration | RAG pipeline integrity |
| `tests/test_evolution.py` | Self-optimizer mutation, synthetic QA generation, meta-modifier audit | Self-healing / repair logic |
| `tests/test_super_rag.py` | Semantic caching, observability tracer, query decomposition | Pipeline robustness |
| `tests/test_sarah.py` | Telemetry drift monitors (partial) | Observability (partial keep) |

### Tests to Delete
- `tests/test_voice_assistant.py` — Voice decomposer not needed
- `tests/test_settings.py` — User settings/subscriptions not needed
- Parts of `tests/test_sarah.py` — Auth vault, GraphRAG seeds, memory extraction

---

## J. Missing Functionality Required for Self-Healing RAG Engine

| # | Missing Component | Priority | Description |
|---|---|---|---|
| 1 | **Failure Taxonomy** | P0 | Structured classification of failure types: retrieval, grounding, generation, citation, latency, configuration |
| 2 | **Diagnosis Engine** | P0 | Root cause analyzer that maps observed failures to probable causes |
| 3 | **Repair Planner** | P0 | Strategy selector that picks appropriate repair action based on diagnosis |
| 4 | **Repair Strategy Library** | P0 | Collection of repair strategies: query rewrite, re-chunk, re-embed, adjust weights, swap models, expand context |
| 5 | **Sandbox Executor** | P0 | Safe candidate environment that applies repair without affecting production |
| 6 | **Regression Evaluator** | P0 | Before/after metric comparison engine with improvement thresholds |
| 7 | **Rollback Controller** | P0 | Automatic revert mechanism when repair degrades performance |
| 8 | **Heal Loop Controller** | P0 | Orchestrator that drives the detect → diagnose → repair → evaluate → deploy/rollback cycle |
| 9 | **RAG Adapter Layer** | P1 | Protocol for connecting to external RAG pipelines (not just own pipeline) |
| 10 | **Continuous Monitor** | P1 | Background loop that continuously evaluates RAG health and triggers healing |
| 11 | **Event Bus** | P1 | Structured event system for emitting failures, repairs, rollbacks |
| 12 | **Ops Dashboard API** | P2 | API endpoints exposing system health, repair history, metric trends |
| 13 | **CLI Interface** | P2 | Command-line interface for running audits, triggering heals, viewing status |
| 14 | **Alerting System** | P3 | Configurable alerts when failure rates exceed thresholds |
