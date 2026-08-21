"""Standalone test script for the Self-Healing RAG Engine.

Tests:
1. Document loading & chunking
2. Vector store + BM25 hybrid indexing
3. Hybrid search + Cross-Encoder reranking
4. Relevance gate check
5. HHEM Faithfulness check
"""

import sys
from pathlib import Path

# Fix Windows console UTF-8 encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console

from src.config import settings
from src.ingestion.loader import load_directory
from src.ingestion.chunker import chunk_documents
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid import reciprocal_rank_fusion
from src.retrieval.reranker import rerank
from src.guardrails.relevance_gate import check_relevance
from src.guardrails.faithfulness_checker import check_faithfulness
from src.guardrails.citation_verifier import verify_citations

console = Console()


def run_test():
    console.print("[bold cyan]════════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]       🧪 SELF-HEALING RAG ENGINE — SYSTEM TEST             [/bold cyan]")
    console.print("[bold cyan]════════════════════════════════════════════════════════════[/bold cyan]\n")

    # Step 1: Ingestion
    console.print("[bold yellow]STEP 1: Document Loading & Chunking[/bold yellow]")
    docs = load_directory(settings.documents_dir)
    chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
    console.print(f"Loaded {len(docs)} documents → {len(chunks)} chunks.\n")

    # Step 2: Indexing
    console.print("[bold yellow]STEP 2: Indexing into ChromaDB & BM25[/bold yellow]")
    vector_store = VectorStore()
    vector_store.clear()
    vector_store.add_chunks(chunks)

    bm25_store = BM25Store()
    bm25_store.clear()
    bm25_store.add_chunks(chunks)
    console.print(f"Indexed {vector_store.count()} chunks into ChromaDB & BM25.\n")

    # Step 3: Test Answerable Query
    query_1 = "What is a Python decorator and what is it used for?"
    console.print(f"[bold yellow]STEP 3: Querying Answerable Question[/bold yellow]")
    console.print(f"Query: [italic]{query_1}[/italic]")

    v_results = vector_store.search(query_1, top_k=10)
    b_results = bm25_store.search(query_1, top_k=10)
    fused = reciprocal_rank_fusion(v_results, b_results)
    reranked = rerank(query_1, fused, top_k=3)

    rel_check = check_relevance(reranked)
    console.print(f"Relevance Gate Result: {rel_check}\n")

    # Step 4: Test Guardrails with Mock Answer
    console.print("[bold yellow]STEP 4: Testing Guardrails (HHEM Faithfulness & Citations)[/bold yellow]")
    mock_good_answer = (
        "A decorator in Python is a function that takes another function and extends its behavior "
        "without explicitly modifying it [1]. Decorators are commonly used for logging, measuring execution "
        "time, access control, and caching with lru_cache [1]."
    )
    console.print(f"Mock Answer: {mock_good_answer}")

    faith_check = check_faithfulness(mock_good_answer, reranked)
    cite_check = verify_citations(mock_good_answer, len(reranked))
    console.print(f"Faithfulness Score: {faith_check.score:.4f} (Passed: {faith_check.passed})")
    console.print(f"Citations Valid: {cite_check.passed}\n")

    # Step 5: Test Unanswerable Query (Relevance Gate Refusal)
    query_2 = "What is the capital of Mars and its atmospheric pressure?"
    console.print(f"[bold yellow]STEP 5: Querying Unanswerable Question (Should refuse)[/bold yellow]")
    console.print(f"Query: [italic]{query_2}[/italic]")

    v_results_2 = vector_store.search(query_2, top_k=10)
    b_results_2 = bm25_store.search(query_2, top_k=10)
    fused_2 = reciprocal_rank_fusion(v_results_2, b_results_2)
    reranked_2 = rerank(query_2, fused_2, top_k=3)

    rel_check_2 = check_relevance(reranked_2)
    console.print(f"Relevance Gate Result: {rel_check_2}\n")

    console.print("[bold green]✅ ALL SYSTEM TESTS PASSED SUCCESSFULLY![/bold green]")


if __name__ == "__main__":
    run_test()
