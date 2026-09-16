"""End-to-end Self-Healing RAG Engine demonstration script.

Executes a complete autonomous healing cycle:
1. Simulates an edge-case or failure query
2. Triggers failure detection & classification (Taxonomy)
3. Diagnoses probable root cause (Diagnosis Engine)
4. Formulates a concrete repair plan with rollback snapshot (Repair Planner)
5. Executes sandboxed candidate evaluation (Sandbox)
6. Compares before/after metrics to prevent regression (Regression Evaluator)
7. Deploys successful repair or executes rollback (Rollback Controller)
"""

import sys
import json
import time
from pathlib import Path

# Add project root to sys.path for direct execution
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.config import settings
from src.ingestion.loader import load_directory
from src.ingestion.chunker import chunk_documents
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid import reciprocal_rank_fusion
from src.retrieval.reranker import rerank
from src.generation.generator import generate_answer
from src.guardrails.relevance_gate import check_relevance
from src.guardrails.faithfulness_checker import check_faithfulness
from src.guardrails.citation_verifier import verify_citations
from src.engine.heal_loop import heal_loop
from src.engine.rollback_controller import rollback_controller
from src.evolution.self_optimizer import self_optimizer

console = Console()


def run_pipeline(query: str, config: dict) -> dict:
    """Standard RAG pipeline runner for the heal loop."""
    start_time = time.time()
    vector_store = VectorStore()
    bm25_store = BM25Store()

    top_k = config.get("top_k_retrieval", settings.top_k_retrieval)
    relevance_threshold = config.get("relevance_threshold", settings.relevance_threshold)
    vector_weight = config.get("vector_weight", 0.5)
    bm25_weight = config.get("bm25_weight", 0.5)
    rrf_k = config.get("rrf_k", 60)

    # Retrieval
    v_res = vector_store.search(query, top_k=top_k)
    b_res = bm25_store.search(query, top_k=top_k)
    fused = reciprocal_rank_fusion(v_res, b_res, k=rrf_k, vector_weight=vector_weight, bm25_weight=bm25_weight)

    seen = set()
    deduped = []
    for r in fused:
        if r["chunk_id"] not in seen:
            seen.add(r["chunk_id"])
            deduped.append(r)

    # Reranking
    reranked = rerank(query, deduped, top_k=5)
    relevance = check_relevance(reranked, threshold=relevance_threshold)

    if not relevance.passed:
        latency_ms = (time.time() - start_time) * 1000
        return {
            "answer": f"Low relevance context for query: {query}",
            "faithfulness_score": 0.0,
            "faithfulness_passed": False,
            "faithfulness_reason": relevance.reason,
            "relevance_score": relevance.best_score,
            "relevance_passed": False,
            "citations_valid": False,
            "citations_reason": "No relevant chunks",
            "generation_error": None,
            "latency_ms": latency_ms,
            "trace": {},
        }

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
        "trace": {},
    }


def main():
    console.print(Panel.fit(
        "[bold cyan]🛡 SELF-HEALING RAG ENGINE — AUTONOMOUS CONTROL PLANE[/bold cyan]\n"
        "[white]Running End-to-End Self-Healing Lifecycle Demonstration[/white]",
        border_style="cyan"
    ))

    # Ingest documents if needed
    v_store = VectorStore()
    if v_store.count() == 0:
        console.print("[yellow]Ingesting sample documents...[/yellow]")
        docs = load_directory(settings.documents_dir)
        chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
        v_store.add_chunks(chunks)
        b_store = BM25Store()
        b_store.add_chunks(chunks)
        console.print(f"[green]Indexed {len(chunks)} chunks.[/green]\n")

    # Target Query
    test_query = "What is the detailed implementation mechanism of Reciprocal Rank Fusion?"
    console.print(f"[bold]Target Query:[/bold] [italic]{test_query}[/italic]\n")

    # Initial pipeline configuration
    config = self_optimizer.get_config_snapshot()
    console.print(f"[dim]Active Configuration: {config}[/dim]\n")

    # Execute Heal Loop
    result = heal_loop.run_once(
        query=test_query,
        pipeline_config=config,
        run_pipeline_fn=run_pipeline,
        test_queries=[test_query],
    )

    # Print summary table
    table = Table(title="Heal Cycle Summary", show_header=True, header_style="bold magenta")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Heal ID", result.heal_id)
    table.add_row("Query", result.query[:50] + "...")
    table.add_row("Failure Detected", str(result.failure_detected))
    table.add_row("Failure Type", result.failure_report.failure_type.value if result.failure_report else "None")
    table.add_row("Root Cause", result.diagnosis.root_cause if result.diagnosis else "Healthy")
    table.add_row("Action Taken", f"[bold green]{result.action_taken.upper()}[/bold green]")
    table.add_row("Duration", f"{result.total_duration_ms:.1f} ms")

    console.print("\n")
    console.print(table)
    console.print("\n[bold green]✓ Heal loop cycle execution completed successfully.[/bold green]")


if __name__ == "__main__":
    main()
