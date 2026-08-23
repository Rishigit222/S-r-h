"""FastAPI application — s@r@h: Self-Adaptive Reasoning & Retrieval Autonomous Host with Meta-RAG & Settings Studio."""

import time
import uuid
import shutil
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from src.config import settings
from src.ingestion.loader import load_directory, load_text, load_pdf
from src.ingestion.chunker import chunk_documents
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid import reciprocal_rank_fusion
from src.retrieval.reranker import rerank
from src.generation.generator import generate_answer, RAGResponse
from src.guardrails.relevance_gate import check_relevance
from src.guardrails.faithfulness_checker import check_faithfulness
from src.guardrails.citation_verifier import verify_citations
from src.guardrails.self_healer import rewrite_query, should_heal, build_refusal_response

# All Integrated Extensions
from src.security.sanitizer import security_sanitizer
from src.guardrails.safety_gate import safety_gate
from src.optimization.semantic_cache import semantic_cache
from src.memory.episodic_memory import memory_store
from src.reasoning.decomposer import query_decomposer
from src.retrieval.web_grounding import web_grounding
from src.graph.knowledge_graph import knowledge_graph
from src.observability.drift_monitor import drift_monitor
from src.observability.tracer import QueryTracer
from src.api.auth import auth_manager, UserLoginRequest, ApiKeyVaultRequest

# Meta-RAG & Self-Evolution Subsystems
from src.evolution.self_optimizer import self_optimizer
from src.evolution.synthetic_trainer import synthetic_trainer
from src.evolution.meta_modifier import rag_meta_modifier

# Settings & Subscription Suite
from src.api.settings import settings_manager, UserProfileUpdateRequest, TierUpgradeRequest


app = FastAPI(
    title="s@r@h — Autonomous Knowledge & Meta-RAG Host",
    description="s@r@h: Self-Adaptive Reasoning & Retrieval Autonomous Host with 3D GraphRAG, Meta-Learning, Settings Suite & External RAG Modifier.",
    version="4.1.0",
)

# Mount Static Files for Modern UI
STATIC_DIR = Path("./static")
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Stores
_vector_store: VectorStore | None = None
_bm25_store: BM25Store | None = None


def _get_stores() -> tuple[VectorStore, BM25Store]:
    global _vector_store, _bm25_store
    if _vector_store is None:
        _vector_store = VectorStore()
    if _bm25_store is None:
        _bm25_store = BM25Store()
    return _vector_store, _bm25_store


# --- Models ---

class QueryRequest(BaseModel):
    question: str = Field(..., description="The question to ask s@r@h")
    session_id: str = Field(default="user_session_01", description="Session ID for long-term memory")
    top_k: int = Field(default=5, description="Number of results to use")
    enable_web_fallback: bool = Field(default=True, description="Enable real-world live grounding fallback")
    enable_graph_rag: bool = Field(default=True, description="Enable Knowledge Graph relational traversal")


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict] = []
    graph_relations: list[str] = []
    confidence: float = 0.0
    faithfulness_score: float = 0.0
    hallucination_risk: float = 0.0
    citations_valid: bool = False
    guardrail_passed: bool = False
    cache_hit: bool = False
    multi_hop_used: bool = False
    web_grounded: bool = False
    user_facts: dict = {}
    heal_attempts: int = 0
    provider: str = ""
    trace: dict = {}
    error: str | None = None


class IngestResponse(BaseModel):
    documents_loaded: int
    chunks_created: int
    chunks_indexed: int
    graph_triples_indexed: int


class HealthResponse(BaseModel):
    status: str
    engine_name: str
    generation_version: int
    subscription_tier: str
    vector_store_count: int
    graph_triples_count: int
    llm_provider: str
    cache_stats: dict
    telemetry_summary: dict
    dynamic_hyperparameters: dict


class ExternalRAGAuditRequest(BaseModel):
    config_or_code: str = Field(..., description="External RAG configuration, YAML, JSON, or Python code to audit and modify")


# --- Endpoints ---

@app.get("/")
async def serve_index():
    """Serves the modern Framer/Motion-grade Glassmorphic UI."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "s@r@h API is running. Open /docs for Swagger documentation."}


@app.get("/health", response_model=HealthResponse)
async def health():
    vector_store, _ = _get_stores()
    user_s = settings_manager.get_settings("user_session_01")
    return HealthResponse(
        status="healthy",
        engine_name="s@r@h",
        generation_version=self_optimizer.params.generation_version,
        subscription_tier=user_s.subscription_tier,
        vector_store_count=vector_store.count(),
        graph_triples_count=knowledge_graph.count(),
        llm_provider=settings.get_effective_provider(),
        cache_stats=semantic_cache.stats(),
        telemetry_summary=drift_monitor.get_summary_metrics(),
        dynamic_hyperparameters={
            "vector_weight": self_optimizer.params.vector_weight,
            "bm25_weight": self_optimizer.params.bm25_weight,
            "relevance_threshold": self_optimizer.params.relevance_threshold,
            "rrf_k": self_optimizer.params.rrf_k,
        },
    )


# --- Settings & Pro Tier Endpoints ---

@app.get("/settings/profile")
async def get_user_settings(session_id: str = "user_session_01"):
    """Fetch current user profile, subscription tier, custom instructions, and preferences."""
    return settings_manager.get_settings(session_id)


@app.post("/settings/profile")
async def update_user_settings(req: UserProfileUpdateRequest, session_id: str = "user_session_01"):
    """Update custom persona instructions, reasoning effort, auto-refine preference, or theme."""
    return settings_manager.update_profile(session_id, req)


@app.post("/settings/tier/upgrade")
async def upgrade_tier(req: TierUpgradeRequest, session_id: str = "user_session_01"):
    """Upgrade or switch subscription tier (e.g. Free ➔ Pro ($20/mo) ➔ Enterprise)."""
    return settings_manager.upgrade_tier(session_id, req.target_tier)


@app.post("/settings/cache/clear")
async def clear_cache():
    """Purge in-memory and persistent semantic cache."""
    semantic_cache.clear()
    return {"status": "cleared", "message": "Semantic cache purged successfully."}


@app.post("/settings/memory/clear")
async def clear_memory(session_id: str = "user_session_01"):
    """Clear episodic chat turns and session history."""
    memory_store.clear_history(session_id)
    return {"status": "cleared", "session_id": session_id, "message": "Episodic memory cleared."}


@app.get("/settings/export")
async def export_data(session_id: str = "user_session_01"):
    """Download comprehensive JSON archive of user memory, telemetry logs, and graph triples."""
    data = settings_manager.export_full_data_archive(session_id)
    return JSONResponse(
        content=data,
        headers={"Content-Disposition": f"attachment; filename=sarah_export_{session_id}.json"}
    )


# --- Meta-RAG & Self-Evolution Endpoints ---

@app.post("/evolution/auto-tune")
async def trigger_auto_tune():
    """Triggers autonomous hyperparameter tuning based on real-time telemetry drift."""
    return self_optimizer.auto_tune()


@app.post("/evolution/synthetic-train")
async def run_synthetic_training():
    """Generates synthetic QA pairs from knowledge documents and executes self-training benchmark."""
    qa_pairs = synthetic_trainer.generate_synthetic_dataset(max_pairs=8)
    vector_store, bm25_store = _get_stores()
    results = []

    for pair in qa_pairs:
        v_res = vector_store.search(pair.question, top_k=5)
        b_res = bm25_store.search(pair.question, top_k=5)
        fused = reciprocal_rank_fusion(
            v_res, b_res,
            k=self_optimizer.params.rrf_k,
            vector_weight=self_optimizer.params.vector_weight,
            bm25_weight=self_optimizer.params.bm25_weight
        )
        reranked = rerank(pair.question, fused, top_k=3)
        relevance = check_relevance(reranked, threshold=self_optimizer.params.relevance_threshold)
        results.append({
            "question": pair.question,
            "target_entity": pair.target_entity,
            "difficulty": pair.difficulty,
            "retrieval_passed": relevance.passed,
            "best_score": relevance.best_score,
        })

    passed_count = sum(1 for r in results if r["retrieval_passed"])
    accuracy_pct = round((passed_count / len(results) * 100), 1) if results else 100.0

    return {
        "dataset_size": len(results),
        "self_trained_accuracy": f"{accuracy_pct}%",
        "passed": passed_count,
        "failed": len(results) - passed_count,
        "evaluations": results,
    }


@app.post("/evolution/audit-rag")
async def audit_external_rag(req: ExternalRAGAuditRequest):
    """Audits external RAG model configurations and generates an optimized drop-in architecture."""
    report = rag_meta_modifier.audit_and_modify(req.config_or_code)
    return report


@app.get("/evolution/status")
async def get_evolution_status():
    """Returns current self-evolved hyperparameters and mutation history."""
    return {
        "current_parameters": {
            "version": f"v{self_optimizer.params.generation_version}",
            "vector_weight": self_optimizer.params.vector_weight,
            "bm25_weight": self_optimizer.params.bm25_weight,
            "relevance_threshold": self_optimizer.params.relevance_threshold,
            "rrf_k": self_optimizer.params.rrf_k,
        },
        "mutation_history": self_optimizer.get_mutation_history(15),
    }


# Auth Endpoints
@app.post("/auth/login")
async def login(req: UserLoginRequest):
    return auth_manager.create_session(req.username, req.role)


@app.post("/auth/keys")
async def save_keys(req: ApiKeyVaultRequest):
    success = auth_manager.save_api_keys(req)
    return {"success": success}


@app.get("/metrics/telemetry")
async def get_telemetry():
    """Real-time LLM quality and drift telemetry logs."""
    return {
        "summary": drift_monitor.get_summary_metrics(),
        "recent_logs": drift_monitor.get_recent_logs(20),
    }


@app.get("/graph/triples")
async def get_graph_triples(limit: int = 100):
    """Retrieve Knowledge Graph triples for 3D visualization."""
    return {"triples": knowledge_graph.get_all_triples(limit)}


@app.get("/memory/facts/{session_id}")
async def get_memory_facts(session_id: str):
    """Retrieve remembered user facts and conversation history."""
    return {
        "session_id": session_id,
        "facts": memory_store.get_user_facts(session_id),
        "history": memory_store.get_recent_history(session_id, limit=10),
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(directory: str = settings.documents_dir):
    vector_store, bm25_store = _get_stores()

    documents = load_directory(directory)
    if not documents:
        raise HTTPException(status_code=400, detail=f"No documents found in {directory}")

    chunks = chunk_documents(documents, settings.chunk_size, settings.chunk_overlap)
    vector_store.add_chunks(chunks)
    bm25_store.add_chunks(chunks)

    # Knowledge Graph Ingestion
    knowledge_graph.extract_and_index_chunks(chunks)

    return IngestResponse(
        documents_loaded=len(documents),
        chunks_created=len(chunks),
        chunks_indexed=vector_store.count(),
        graph_triples_indexed=knowledge_graph.count(),
    )


@app.post("/ingest/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload custom document to knowledge corpus and index on the fly."""
    save_path = Path(settings.documents_dir) / file.filename
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return await ingest_documents()


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    tracer = QueryTracer(trace_id=str(uuid.uuid4()), query=request.question)
    vector_store, bm25_store = _get_stores()
    user_settings = settings_manager.get_settings(request.session_id)

    # 1. Security Audit
    sec_audit = security_sanitizer.audit_input_query(request.question)
    tracer.record_step("Security Audit", "passed" if sec_audit.is_safe else "blocked", {
        "injections": sec_audit.injections_detected,
        "pii_redacted": sec_audit.pii_redacted,
    })
    if not sec_audit.is_safe:
        return QueryResponse(
            answer="⚠️ Request blocked by s@r@h security guard: Input contains suspected prompt injection or unsafe patterns.",
            guardrail_passed=False,
            trace=tracer.finalize(),
        )

    sanitized_query = sec_audit.sanitized_text

    # Record to Memory Agent immediately for user entity extraction
    memory_store.add_message(request.session_id, "user", request.question)
    user_facts = memory_store.get_user_facts(request.session_id)
    history = memory_store.get_recent_history(request.session_id, limit=4)
    tracer.record_step("Memory Agent", "success", {"history_turns": len(history), "facts": list(user_facts.keys())})

    # 2. Safety & Anti-Sycophancy Gate
    safety_audit = safety_gate.evaluate(sanitized_query)
    tracer.record_step("Safety Gate", "passed" if safety_audit.passed else "blocked", {
        "risk_type": safety_audit.risk_type
    })
    if not safety_audit.passed:
        return QueryResponse(
            answer=safety_audit.warning_message or "Request blocked by safety policy.",
            guardrail_passed=False,
            user_facts=user_facts,
            trace=tracer.finalize(),
        )

    # 3. Semantic Cache Lookup (< 5ms response on hit)
    cached_resp, sim_score = semantic_cache.lookup(sanitized_query)
    if cached_resp is not None:
        tracer.trace.cache_hit = True
        tracer.record_step("Semantic Cache", "hit", {"similarity": sim_score})
        cached_resp["trace"] = tracer.finalize()
        cached_resp["cache_hit"] = True
        cached_resp["user_facts"] = user_facts

        drift_monitor.log_query(
            trace_id=tracer.trace.trace_id,
            query=request.question,
            faithfulness=cached_resp.get("faithfulness_score", 1.0),
            hallucination_risk=cached_resp.get("hallucination_risk", 0.0),
            guardrail_passed=True,
            cache_hit=True,
            multi_hop_used=cached_resp.get("multi_hop_used", False),
            latency_ms=tracer.trace.total_duration_ms,
        )
        return QueryResponse(**cached_resp)

    tracer.record_step("Semantic Cache", "miss", {"best_similarity": sim_score})

    # 4. Multi-Hop Query Decomposition & Conversational Cleaning
    decomp = query_decomposer.decompose(sanitized_query)
    tracer.trace.multi_hop_used = decomp.is_complex
    tracer.record_step("Query Decomposition", "success", {
        "is_complex": decomp.is_complex,
        "sub_queries": decomp.sub_queries,
    })

    # 5. Knowledge Graph RAG Traversal
    graph_triples = []
    if request.enable_graph_rag:
        for sq in decomp.sub_queries:
            graph_triples.extend(knowledge_graph.query_graph(sq))
        tracer.record_step("GraphRAG Traversal", "success", {"relations_found": len(graph_triples)})

    current_query = decomp.sub_queries[0] if len(decomp.sub_queries) == 1 else sanitized_query
    heal_attempts = 0
    web_grounded = False

    while True:
        # Step 6: Hybrid Retrieval using Dynamic Self-Tuned Hyperparameters
        all_fused = []
        for q in decomp.sub_queries:
            v_res = vector_store.search(q, top_k=self_optimizer.params.top_k_retrieval)
            b_res = bm25_store.search(q, top_k=self_optimizer.params.top_k_retrieval)
            all_fused.extend(
                reciprocal_rank_fusion(
                    v_res, b_res,
                    k=self_optimizer.params.rrf_k,
                    vector_weight=self_optimizer.params.vector_weight,
                    bm25_weight=self_optimizer.params.bm25_weight,
                )
            )

        seen_ids = set()
        fused_results = []
        for r in all_fused:
            if r["chunk_id"] not in seen_ids:
                seen_ids.add(r["chunk_id"])
                fused_results.append(r)

        tracer.record_step("Hybrid Retrieval", "success", {"candidates_found": len(fused_results)})

        # Step 7: Cross-Encoder Reranking
        reranked = rerank(current_query, fused_results, top_k=request.top_k)
        tracer.record_step("Reranker", "success", {
            "top_candidates": len(reranked),
            "best_score": reranked[0]["rerank_score"] if reranked else -99.0
        })

        # Step 8: Pre-generation Relevance Gate with Self-Tuned Threshold
        relevance = check_relevance(reranked, threshold=self_optimizer.params.relevance_threshold)
        tracer.record_step("Relevance Gate", "passed" if relevance.passed else "blocked", {
            "best_score": relevance.best_score,
            "threshold": relevance.threshold,
        })

        # Live Web Grounding Fallback
        if not relevance.passed and request.enable_web_fallback and heal_attempts == 0:
            live_res = web_grounding.search_live_grounding(current_query)
            if live_res.found:
                web_grounded = True
                reranked = [{
                    "chunk_id": "web_live_01",
                    "content": live_res.content,
                    "metadata": {"source": live_res.source_title, "url": live_res.url},
                    "rerank_score": 1.0,
                }]
                relevance.passed = True
                tracer.record_step("Web Grounding Fallback", "success", {"source": live_res.source_title})

        if not relevance.passed:
            refusal_ans = f"s@r@h verified that the knowledge base does not contain enough context to answer: \"{request.question}\"\n\nReason: {relevance.reason}"
            trace_dict = tracer.finalize()
            drift_monitor.log_query(
                trace_id=tracer.trace.trace_id, query=request.question,
                faithfulness=0.0, hallucination_risk=1.0, guardrail_passed=False,
                cache_hit=False, multi_hop_used=decomp.is_complex, latency_ms=trace_dict["total_duration_ms"]
            )
            return QueryResponse(
                answer=refusal_ans,
                guardrail_passed=True,
                confidence=1.0,
                user_facts=user_facts,
                trace=trace_dict,
            )

        # Inject Graph Triples & Custom Persona Instructions into context
        generation_context = list(reranked)
        if graph_triples:
            graph_chunk = {
                "chunk_id": "graph_triples",
                "content": "Knowledge Graph Relationships:\n" + "\n".join(t["text"] for t in graph_triples[:4]),
                "metadata": {"source": "s@r@h Knowledge Graph"},
            }
            generation_context.append(graph_chunk)

        # Append Custom User Persona Instructions if set
        if user_settings.custom_instructions_user or user_settings.custom_instructions_style:
            custom_chunk = {
                "chunk_id": "custom_instructions",
                "content": f"User Persona & Preference Context:\n- Background: {user_settings.custom_instructions_user}\n- Output Style: {user_settings.custom_instructions_style}",
                "metadata": {"source": "User Settings & Persona"},
            }
            generation_context.append(custom_chunk)

        # Step 9: LLM Generation
        rag_response = generate_answer(current_query, generation_context)
        tracer.record_step("LLM Generation", "success" if not rag_response.error else "error")

        if rag_response.error:
            return QueryResponse(
                answer=rag_response.answer,
                error=rag_response.error,
                provider=settings.get_effective_provider(),
                user_facts=user_facts,
                trace=tracer.finalize(),
            )

        # Step 10: Guardrail Verification
        faithfulness = check_faithfulness(rag_response.answer, reranked)
        citations = verify_citations(rag_response.answer, len(generation_context))
        tracer.record_step("Guardrail Verification", "passed" if (faithfulness.passed and citations.passed) else "failed", {
            "faithfulness_score": faithfulness.score,
            "citations_valid": citations.passed,
        })

        guardrail_passed = faithfulness.passed and citations.passed

        if guardrail_passed:
            sanitized_answer = security_sanitizer.sanitize_output(rag_response.answer)

            # Memory persistence
            memory_store.add_message(request.session_id, "assistant", sanitized_answer)

            final_response = {
                "answer": sanitized_answer,
                "sources": [{"content": s["content"][:250], "metadata": s["metadata"]} for s in reranked],
                "graph_relations": [t["text"] for t in graph_triples],
                "confidence": faithfulness.score,
                "faithfulness_score": faithfulness.score,
                "hallucination_risk": round(1.0 - faithfulness.score, 3),
                "citations_valid": citations.passed,
                "guardrail_passed": True,
                "cache_hit": False,
                "multi_hop_used": decomp.is_complex,
                "web_grounded": web_grounded,
                "user_facts": memory_store.get_user_facts(request.session_id),
                "heal_attempts": heal_attempts,
                "provider": settings.get_effective_provider(),
            }

            semantic_cache.store(sanitized_query, final_response)
            final_response["trace"] = tracer.finalize()

            # Telemetry logging
            drift_monitor.log_query(
                trace_id=tracer.trace.trace_id, query=request.question,
                faithfulness=faithfulness.score, hallucination_risk=round(1.0 - faithfulness.score, 3),
                guardrail_passed=True, cache_hit=False, multi_hop_used=decomp.is_complex,
                latency_ms=final_response["trace"]["total_duration_ms"]
            )

            # Autonomous Auto-Refine trigger if enabled
            if user_settings.auto_refine_enabled:
                summary = drift_monitor.get_summary_metrics()
                if summary.get("total_queries", 0) % 10 == 0:
                    self_optimizer.auto_tune()

            return QueryResponse(**final_response)

        # Self-Heal loop
        if not should_heal(heal_attempts):
            reason = f"Faithfulness: {faithfulness.reason}. Citations: {citations.reason}"
            refusal_ans = build_refusal_response(request.question, reason)
            trace_dict = tracer.finalize()
            drift_monitor.log_query(
                trace_id=tracer.trace.trace_id, query=request.question,
                faithfulness=faithfulness.score, hallucination_risk=round(1.0 - faithfulness.score, 3),
                guardrail_passed=False, cache_hit=False, multi_hop_used=decomp.is_complex,
                latency_ms=trace_dict["total_duration_ms"]
            )
            return QueryResponse(
                answer=refusal_ans,
                faithfulness_score=faithfulness.score,
                hallucination_risk=round(1.0 - faithfulness.score, 3),
                citations_valid=citations.passed,
                guardrail_passed=False,
                user_facts=user_facts,
                heal_attempts=heal_attempts,
                provider=settings.get_effective_provider(),
                trace=trace_dict,
            )

        heal_attempts += 1
        current_query = rewrite_query(request.question)
        tracer.record_step("Self-Healing Rewrite", "retry", {"new_query": current_query})
