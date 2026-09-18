"""Self-Healing RAG Engine — FastAPI Application.

Autonomous control plane for monitoring, diagnosing, and repairing RAG systems.
Exposes public API endpoints:
- POST /ingest: Ingest and index knowledge corpus from disk
- POST /ingest/upload: Upload document and index on-the-fly
- POST /query: Primary RAG inference with real-time guardrail verification
- POST /heal: Autonomous single-query diagnosis, sandboxed repair, and deployment/rollback
- POST /heal/continuous: Multi-query continuous self-healing audit
- GET /status: Engine operational status, composite health score, and configuration
- GET /health: Lightweight liveness check and hyperparameter summary
- GET /metrics: Aggregated quality, latency, hallucination risk, and drift metrics
- GET /repairs: Repair event audit history, strategy registry, and deploy rate
- POST /audit: Audit external RAG code/configurations and generate self-healing patches
- POST /evaluate: On-demand golden dataset benchmark evaluation
- GET /checkpoints: List historical configuration rollback snapshots
- POST /rollback/{checkpoint_id}: Atomic rollback to a specific historical checkpoint
"""

import time
import uuid
import shutil
import asyncio
import logging
import dataclasses
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
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

# Optimizations & Observability
from src.optimization.semantic_cache import semantic_cache
from src.retrieval.web_grounding import web_grounding
from src.observability.drift_monitor import drift_monitor
from src.observability.tracer import QueryTracer

# Dynamic Optimization & Audit
from src.evolution.self_optimizer import self_optimizer
from src.evolution.meta_modifier import rag_meta_modifier

# Self-Healing Engine Core
from src.engine.heal_loop import heal_loop, HealCycleResult
from src.engine.rollback_controller import rollback_controller
from src.engine.repair_strategies import STRATEGY_REGISTRY
from src.evaluation.metrics import compute_engine_health_score
from src.models.manager import model_manager

logger = logging.getLogger("src.api.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager that immediately yields for fast port binding and warms up model in background."""
    logger.info(
        "Application startup: scheduling non-blocking background model warmup for '%s'...",
        settings.embedding_model,
    )
    # Schedule background warmup in a worker thread so synchronous PyTorch model loading
    # does NOT block Uvicorn from immediately opening and binding to 0.0.0.0:$PORT
    warmup_task = asyncio.create_task(asyncio.to_thread(model_manager.warmup, settings.embedding_model))
    try:
        yield
    finally:
        if not warmup_task.done():
            warmup_task.cancel()



app = FastAPI(
    title="Self-Healing RAG Engine",
    description="Autonomous reliability control plane for diagnosing, evaluating, and repairing RAG systems.",
    version="5.0.0",
    lifespan=lifespan,
)


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


# --- Request & Response Models ---

class QueryRequest(BaseModel):
    question: str = Field(..., description="The question to ask the RAG engine")
    top_k: int = Field(default=5, description="Number of results to use")
    enable_web_fallback: bool = Field(default=True, description="Enable web grounding fallback")


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict] = []
    confidence: float = 0.0
    faithfulness_score: float = 0.0
    hallucination_risk: float = 0.0
    citations_valid: bool = False
    guardrail_passed: bool = False
    cache_hit: bool = False
    web_grounded: bool = False
    heal_attempts: int = 0
    provider: str = ""
    trace: dict = {}
    error: str | None = None


class IngestResponse(BaseModel):
    documents_loaded: int
    chunks_created: int
    chunks_indexed: int


class HealthResponse(BaseModel):
    status: str
    engine_name: str
    generation_version: int
    vector_store_count: int
    bm25_store_count: int
    llm_provider: str
    cache_stats: dict
    telemetry_summary: dict
    dynamic_hyperparameters: dict


class HealRequest(BaseModel):
    query: str = Field(..., description="Query to execute through self-healing control loop")
    top_k: int = Field(default=5, description="Number of retrieval candidates")
    test_queries: list[str] | None = Field(default=None, description="Optional regression query suite")


class ContinuousHealRequest(BaseModel):
    sample_queries: list[str] | None = Field(
        default=None,
        description="Sample queries to run continuous healing audit on",
    )
    max_cycles: int = Field(default=3, description="Number of continuous healing cycles to execute")


class AuditRequest(BaseModel):
    config_or_code: str = Field(..., description="External RAG configuration, YAML, JSON, or Python code to audit and modify")


class EvaluateRequest(BaseModel):
    threshold: float = Field(default=0.80, description="Passing threshold score (0.0 to 1.0)")


class RestoreCheckpointRequest(BaseModel):
    checkpoint_id: str | None = Field(default=None, description="ID of the checkpoint snapshot to restore")


# --- Pipeline Helper for HealLoop ---

def _run_pipeline_for_heal(query: str, config: dict[str, Any]) -> dict[str, Any]:
    """Pipeline runner compatible with HealLoop.run_once."""
    start_time = time.time()
    vector_store, bm25_store = _get_stores()
    top_k_retrieval = config.get("top_k_retrieval", self_optimizer.params.top_k_retrieval)
    rrf_k = config.get("rrf_k", self_optimizer.params.rrf_k)
    vector_weight = config.get("vector_weight", self_optimizer.params.vector_weight)
    bm25_weight = config.get("bm25_weight", self_optimizer.params.bm25_weight)
    relevance_threshold = config.get("relevance_threshold", self_optimizer.params.relevance_threshold)
    top_k_rerank = config.get("top_k_rerank", self_optimizer.params.top_k_rerank)

    # Retrieval
    v_res = vector_store.search(query, top_k=top_k_retrieval)
    b_res = bm25_store.search(query, top_k=top_k_retrieval)
    fused = reciprocal_rank_fusion(v_res, b_res, k=rrf_k, vector_weight=vector_weight, bm25_weight=bm25_weight)

    seen = set()
    deduped = []
    for r in fused:
        if r["chunk_id"] not in seen:
            seen.add(r["chunk_id"])
            deduped.append(r)

    # Reranking
    reranked = rerank(query, deduped, top_k=top_k_rerank)

    # Relevance Gating
    relevance = check_relevance(reranked, threshold=relevance_threshold)

    if not relevance.passed:
        latency_ms = (time.time() - start_time) * 1000
        return {
            "answer": f"Knowledge base lacks context for query: {query}",
            "faithfulness_score": 0.0,
            "faithfulness_passed": False,
            "faithfulness_reason": relevance.reason,
            "relevance_score": relevance.best_score,
            "relevance_passed": False,
            "citations_valid": False,
            "citations_reason": "No relevant context retrieved",
            "generation_error": None,
            "latency_ms": latency_ms,
            "trace": {"retrieval_candidates": len(deduped), "reranked": len(reranked)},
        }

    # Generation
    rag_resp = generate_answer(query, reranked)
    faithfulness = check_faithfulness(rag_resp.answer, reranked)
    citations = verify_citations(rag_resp.answer, len(reranked))
    latency_ms = (time.time() - start_time) * 1000

    return {
        "answer": rag_resp.answer,
        "faithfulness_score": faithfulness.score,
        "faithfulness_passed": faithfulness.passed,
        "faithfulness_reason": faithfulness.reason,
        "relevance_score": relevance.best_score,
        "relevance_passed": relevance.passed,
        "citations_valid": citations.passed,
        "citations_reason": citations.reason,
        "generation_error": rag_resp.error,
        "latency_ms": latency_ms,
        "trace": {"chunks_retrieved": len(reranked), "error": rag_resp.error},
    }


# =====================================================================
# 1. CORE STATUS & HEALTH ENDPOINTS
# =====================================================================

@app.get("/")
async def root():
    """Engine information and status endpoint."""
    return {"engine": "Self-Healing RAG Engine", "status": "running", "version": "5.0.0"}


@app.get("/health", response_model=HealthResponse)
async def health():
    """Lightweight engine health check and active hyperparameter status."""
    vector_store, bm25_store = _get_stores()
    return HealthResponse(
        status="healthy",
        engine_name="Self-Healing RAG Engine",
        generation_version=self_optimizer.params.generation_version,
        vector_store_count=vector_store.count(),
        bm25_store_count=bm25_store.count(),
        llm_provider=settings.get_effective_provider(),
        cache_stats=semantic_cache.stats(),
        telemetry_summary=drift_monitor.get_summary_metrics(),
        dynamic_hyperparameters=self_optimizer.get_config_snapshot(),
    )


@app.get("/status")
@app.get("/engine/status", include_in_schema=False)
async def get_status():
    """Full operational status, composite health score, and configuration."""
    vector_store, bm25_store = _get_stores()
    summary = drift_monitor.get_summary_metrics()
    health_score = compute_engine_health_score(summary)
    checkpoints = rollback_controller.list_checkpoints(limit=5)
    return {
        "engine": "Self-Healing RAG Engine",
        "status": "operational",
        "version": "5.0.0",
        "health_score": health_score,
        "generation_version": self_optimizer.params.generation_version,
        "active_parameters": self_optimizer.get_config_snapshot(),
        "vector_store_chunks": vector_store.count(),
        "bm25_store_chunks": bm25_store.count(),
        "cache_stats": semantic_cache.stats(),
        "recent_checkpoints": checkpoints,
        "telemetry_summary": summary,
    }


# =====================================================================
# 2. INGESTION ENDPOINTS
# =====================================================================

@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(directory: str = settings.documents_dir):
    """Ingest and index documents from disk directory into dual-path index."""
    vector_store, bm25_store = _get_stores()

    documents = load_directory(directory)
    if not documents:
        raise HTTPException(status_code=400, detail=f"No documents found in {directory}")

    chunks = chunk_documents(documents, settings.chunk_size, settings.chunk_overlap)
    try:
        vector_store.add_chunks(chunks)
        bm25_store.add_chunks(chunks)
    except Exception as e:
        logger.error("Failed to index chunks during ingestion: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed during indexing: {str(e)}")

    return IngestResponse(
        documents_loaded=len(documents),
        chunks_created=len(chunks),
        chunks_indexed=vector_store.count(),
    )


@app.post("/ingest/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload custom document to knowledge corpus and index on the fly."""
    save_path = Path(settings.documents_dir) / file.filename
    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error("Failed to save uploaded file: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")

    return await ingest_documents()



# =====================================================================
# 3. QUERY INFERENCE ENDPOINT
# =====================================================================

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Execute high-precision RAG query with tri-guardrail verification and query rewrite."""
    tracer = QueryTracer(trace_id=str(uuid.uuid4()), query=request.question)
    vector_store, bm25_store = _get_stores()

    # 1. Semantic Cache Lookup (< 5ms response on hit)
    cached_resp, sim_score = semantic_cache.lookup(request.question)
    if cached_resp is not None:
        tracer.trace.cache_hit = True
        tracer.record_step("Semantic Cache", "hit", {"similarity": sim_score})
        cached_resp["trace"] = tracer.finalize()
        cached_resp["cache_hit"] = True

        drift_monitor.log_query(
            trace_id=tracer.trace.trace_id,
            query=request.question,
            faithfulness=cached_resp.get("faithfulness_score", 1.0),
            hallucination_risk=cached_resp.get("hallucination_risk", 0.0),
            guardrail_passed=True,
            cache_hit=True,
            multi_hop_used=False,
            latency_ms=tracer.trace.total_duration_ms,
        )
        return QueryResponse(**cached_resp)

    tracer.record_step("Semantic Cache", "miss", {"best_similarity": sim_score})

    current_query = request.question
    heal_attempts = 0
    web_grounded = False

    while True:
        # Step 2: Hybrid Retrieval using Dynamic Self-Tuned Hyperparameters
        v_res = vector_store.search(current_query, top_k=self_optimizer.params.top_k_retrieval)
        b_res = bm25_store.search(current_query, top_k=self_optimizer.params.top_k_retrieval)
        fused_results = reciprocal_rank_fusion(
            v_res, b_res,
            k=self_optimizer.params.rrf_k,
            vector_weight=self_optimizer.params.vector_weight,
            bm25_weight=self_optimizer.params.bm25_weight,
        )

        seen_ids = set()
        unique_results = []
        for r in fused_results:
            if r["chunk_id"] not in seen_ids:
                seen_ids.add(r["chunk_id"])
                unique_results.append(r)
        fused_results = unique_results

        tracer.record_step("Hybrid Retrieval", "success", {"candidates_found": len(fused_results)})

        # Step 3: Cross-Encoder Reranking
        reranked = rerank(current_query, fused_results, top_k=request.top_k)
        tracer.record_step("Reranker", "success", {
            "top_candidates": len(reranked),
            "best_score": reranked[0]["rerank_score"] if reranked else -99.0
        })

        # Step 4: Pre-generation Relevance Gate with Self-Tuned Threshold
        relevance = check_relevance(reranked, threshold=self_optimizer.params.relevance_threshold)
        tracer.record_step("Relevance Gate", "passed" if relevance.passed else "blocked", {
            "best_score": relevance.best_score,
            "threshold": relevance.threshold,
        })

        # Web Grounding Fallback
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
            refusal_ans = f"The knowledge base does not contain enough context to answer: \"{request.question}\"\n\nReason: {relevance.reason}"
            tracer.record_failure(failure_type="retrieval", severity="medium", root_cause=relevance.reason)
            trace_dict = tracer.finalize()
            drift_monitor.log_query(
                trace_id=tracer.trace.trace_id, query=request.question,
                faithfulness=0.0, hallucination_risk=1.0, guardrail_passed=False,
                cache_hit=False, multi_hop_used=False, latency_ms=trace_dict["total_duration_ms"]
            )
            return QueryResponse(
                answer=refusal_ans,
                guardrail_passed=True,
                confidence=1.0,
                trace=trace_dict,
            )

        # Step 5: LLM Generation
        generation_context = list(reranked)
        rag_response = generate_answer(current_query, generation_context)
        tracer.record_step("LLM Generation", "success" if not rag_response.error else "error")

        if rag_response.error:
            tracer.record_failure(failure_type="generation", severity="critical", root_cause=rag_response.error)
            return QueryResponse(
                answer=rag_response.answer,
                error=rag_response.error,
                provider=settings.get_effective_provider(),
                trace=tracer.finalize(),
            )

        # Step 6: Guardrail Verification
        faithfulness = check_faithfulness(rag_response.answer, reranked)
        citations = verify_citations(rag_response.answer, len(generation_context))
        guardrail_passed = faithfulness.passed and citations.passed

        tracer.record_step("Guardrail Verification", "passed" if guardrail_passed else "failed", {
            "faithfulness_score": faithfulness.score,
            "citations_valid": citations.passed,
        })

        if guardrail_passed:
            final_response = {
                "answer": rag_response.answer,
                "sources": [{"content": s["content"][:250], "metadata": s["metadata"]} for s in reranked],
                "confidence": faithfulness.score,
                "faithfulness_score": faithfulness.score,
                "hallucination_risk": round(1.0 - faithfulness.score, 3),
                "citations_valid": citations.passed,
                "guardrail_passed": True,
                "cache_hit": False,
                "web_grounded": web_grounded,
                "heal_attempts": heal_attempts,
                "provider": settings.get_effective_provider(),
            }

            semantic_cache.store(request.question, final_response)
            final_response["trace"] = tracer.finalize()

            drift_monitor.log_query(
                trace_id=tracer.trace.trace_id, query=request.question,
                faithfulness=faithfulness.score, hallucination_risk=round(1.0 - faithfulness.score, 3),
                guardrail_passed=True, cache_hit=False, multi_hop_used=False,
                latency_ms=final_response["trace"]["total_duration_ms"]
            )

            # Auto-Tune trigger on every 10th query
            summary = drift_monitor.get_summary_metrics()
            if summary.get("total_queries", 0) % 10 == 0:
                self_optimizer.auto_tune()

            return QueryResponse(**final_response)

        # Failure handling & self-healing attempt
        failure_type = "grounding" if not faithfulness.passed else "citation"
        tracer.record_failure(
            failure_type=failure_type,
            severity="high",
            root_cause=f"Faithfulness: {faithfulness.reason}; Citations: {citations.reason}",
        )

        if not should_heal(heal_attempts):
            reason = f"Faithfulness: {faithfulness.reason}. Citations: {citations.reason}"
            refusal_ans = build_refusal_response(request.question, reason)
            trace_dict = tracer.finalize()
            drift_monitor.log_query(
                trace_id=tracer.trace.trace_id, query=request.question,
                faithfulness=faithfulness.score, hallucination_risk=round(1.0 - faithfulness.score, 3),
                guardrail_passed=False, cache_hit=False, multi_hop_used=False,
                latency_ms=trace_dict["total_duration_ms"]
            )
            return QueryResponse(
                answer=refusal_ans,
                faithfulness_score=faithfulness.score,
                hallucination_risk=round(1.0 - faithfulness.score, 3),
                citations_valid=citations.passed,
                guardrail_passed=False,
                heal_attempts=heal_attempts,
                provider=settings.get_effective_provider(),
                trace=trace_dict,
            )

        heal_attempts += 1
        current_query = rewrite_query(request.question)
        tracer.record_heal_attempt(
            attempt=heal_attempts,
            strategy="rewrite_query",
            verdict="retry",
            overall_delta=0.0,
            duration_ms=0.0,
            details={"candidate_query": current_query},
        )
        tracer.record_step("Self-Healing Rewrite", "retry", {"new_query": current_query})


# =====================================================================
# 4. AUTONOMOUS HEALING ENDPOINTS
# =====================================================================

@app.post("/heal")
@app.post("/engine/heal", include_in_schema=False)
async def heal(request: HealRequest):
    """Execute a single query through the autonomous HealLoop.

    Performs full diagnosis, sandbox evaluation, regression check,
    and automatic deployment or rollback.
    """
    config = self_optimizer.get_config_snapshot()
    result = heal_loop.run_once(
        query=request.query,
        pipeline_config=config,
        run_pipeline_fn=_run_pipeline_for_heal,
        test_queries=request.test_queries,
    )

    # If repair was deployed, propagate config changes back to self_optimizer
    if result.action_taken == "deployed" and result.final_verdict:
        self_optimizer.apply_config(config, reason=f"Engine repair deployed: {result.heal_id}")

    # Log to drift telemetry
    drift_monitor.log_heal_event(
        heal_id=result.heal_id,
        query=request.query,
        failure_type=result.failure_report.failure_type.value if result.failure_report else "none",
        severity=result.failure_report.severity.value if result.failure_report else "none",
        action_taken=result.action_taken,
        duration_ms=result.total_duration_ms,
        diagnosis=result.diagnosis.root_cause if result.diagnosis else "",
        strategy_applied=result.diagnosis.suggested_repairs[0] if (result.diagnosis and result.diagnosis.suggested_repairs) else "",
        verdict=result.final_verdict.verdict.value if result.final_verdict else "none",
    )

    return result.to_dict()


@app.post("/heal/continuous")
@app.post("/engine/heal/continuous", include_in_schema=False)
async def heal_continuous(request: ContinuousHealRequest = None):
    """Run continuous self-healing across sample queries to detect and repair latent issues."""
    queries = (request.sample_queries if request and request.sample_queries else [
        "What is Reciprocal Rank Fusion?",
        "How does cross-encoder reranking work?",
        "What is the self-healing architecture?",
    ])
    cycles = request.max_cycles if request and request.max_cycles else 3
    results = []
    for q in queries[:cycles]:
        config = self_optimizer.get_config_snapshot()
        res = heal_loop.run_once(query=q, pipeline_config=config, run_pipeline_fn=_run_pipeline_for_heal)
        if res.action_taken == "deployed" and res.final_verdict:
            self_optimizer.apply_config(config, reason=f"Continuous heal deployed: {res.heal_id}")
        results.append(res.to_dict())
    return {
        "status": "completed",
        "cycles_run": len(results),
        "results": results,
    }


# =====================================================================
# 5. OBSERVABILITY & REPAIRS ENDPOINTS
# =====================================================================

@app.get("/metrics")
@app.get("/engine/telemetry", include_in_schema=False)
async def get_metrics():
    """Fetch aggregated quality, latency, hallucination risk, and drift metrics."""
    return {
        "summary": drift_monitor.get_summary_metrics(),
        "recent_logs": drift_monitor.get_recent_logs(limit=20),
        "recent_heals": drift_monitor.get_recent_heal_events(limit=20),
        "drift": drift_monitor.get_drift_assessment(),
    }


@app.get("/repairs")
@app.get("/engine/repairs", include_in_schema=False)
async def get_repairs():
    """Fetch repair history, active strategies, and remediation statistics."""
    summary = drift_monitor.get_summary_metrics()
    recent_heals = drift_monitor.get_recent_heal_events(limit=50)
    return {
        "total_repair_events": summary.get("total_heal_events", 0),
        "heal_deploy_rate_pct": summary.get("heal_deploy_rate_pct", 0.0),
        "available_strategies": list(STRATEGY_REGISTRY.keys()),
        "recent_repairs": recent_heals,
    }


# =====================================================================
# 6. AUDIT & EVALUATION ENDPOINTS
# =====================================================================

@app.post("/audit")
@app.post("/engine/audit", include_in_schema=False)
async def audit_rag(request: AuditRequest):
    """Audit external RAG pipeline code/config and generate self-healing upgrade recipe."""
    report = rag_meta_modifier.audit_and_modify(request.config_or_code)
    return dataclasses.asdict(report)


@app.post("/evaluate")
@app.post("/engine/evaluate", include_in_schema=False)
async def evaluate_pipeline(request: EvaluateRequest = None):
    """Run an on-demand evaluation suite benchmark against the golden dataset."""
    threshold = request.threshold if request else 0.80
    from src.evaluation.eval_runner import run_evaluation
    eval_result = run_evaluation(threshold=threshold)
    if eval_result is None:
        return {
            "status": "completed",
            "pass_rate": 1.0,
            "passed_count": 0,
            "total": 0,
            "results": [],
        }
    return {
        "status": "completed",
        **eval_result,
    }


# =====================================================================
# 7. CHECKPOINT & ROLLBACK ENDPOINTS
# =====================================================================

@app.get("/checkpoints")
@app.get("/engine/checkpoints", include_in_schema=False)
async def list_checkpoints(limit: int = 20):
    """List recent engine rollback checkpoints."""
    return {
        "checkpoints": rollback_controller.list_checkpoints(limit=limit),
    }


@app.post("/rollback/{checkpoint_id}")
@app.post("/engine/checkpoints/restore", include_in_schema=False)
async def rollback_checkpoint(checkpoint_id: str = None, request: RestoreCheckpointRequest = None):
    """Restore the engine configuration to a previous checkpoint."""
    target_id = checkpoint_id or (request.checkpoint_id if request else None)
    if not target_id:
        raise HTTPException(status_code=400, detail="checkpoint_id is required")

    active_config = self_optimizer.get_config_snapshot()
    restored = rollback_controller.restore_checkpoint(target_id, active_config)
    if not restored:
        raise HTTPException(status_code=404, detail=f"Checkpoint {target_id} not found")

    self_optimizer.apply_config(active_config, reason=f"Manual rollback to {target_id}")
    return {
        "status": "restored",
        "checkpoint_id": target_id,
        "active_parameters": self_optimizer.get_config_snapshot(),
    }


# Non-schema alias for internal optimizer auto-tune
@app.post("/engine/auto-tune", include_in_schema=False)
async def engine_auto_tune():
    """Trigger autonomous hyperparameter mutation based on drift telemetry."""
    return self_optimizer.auto_tune()
