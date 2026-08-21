"""Unit tests for Phase 1 (Ingestion & Storage) and Phase 2 (Hybrid Retrieval)."""

import pytest
from pathlib import Path
from src.ingestion.loader import load_directory, Document
from src.ingestion.chunker import chunk_documents
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid import reciprocal_rank_fusion
from src.retrieval.reranker import rerank
from src.config import settings


def test_document_loader():
    """Verify document loading from sample docs directory."""
    docs = load_directory(settings.documents_dir)
    assert len(docs) >= 2, "Should load at least 2 markdown docs"
    for doc in docs:
        assert isinstance(doc, Document)
        assert len(doc.content) > 100


def test_chunking_preserves_metadata():
    """Verify semantic chunking splits text and keeps metadata."""
    sample_doc = Document(
        content="First paragraph about Python.\n\nSecond paragraph about Python decorators.",
        metadata={"source": "test.md"}
    )
    chunks = chunk_documents([sample_doc], chunk_size=100, chunk_overlap=20)
    assert len(chunks) >= 1
    assert chunks[0].metadata["source"] == "test.md"
    assert "chunk_index" in chunks[0].metadata


def test_vector_and_bm25_hybrid_retrieval():
    """Verify both ChromaDB and BM25 index and retrieve relevant content."""
    docs = load_directory(settings.documents_dir)
    chunks = chunk_documents(docs, chunk_size=512, chunk_overlap=64)

    # 1. Test Vector Store
    v_store = VectorStore()
    v_store.clear()
    v_store.add_chunks(chunks)
    assert v_store.count() > 0

    v_results = v_store.search("decorator", top_k=5)
    assert len(v_results) > 0
    assert any("decorator" in r["content"].lower() for r in v_results)

    # 2. Test BM25 Store
    b_store = BM25Store()
    b_store.clear()
    b_store.add_chunks(chunks)
    assert b_store.count() > 0

    b_results = b_store.search("decorator", top_k=5)
    assert len(b_results) > 0
    assert any("decorator" in r["content"].lower() for r in b_results)

    # 3. Test Reciprocal Rank Fusion
    fused = reciprocal_rank_fusion(v_results, b_results)
    assert len(fused) > 0
    assert "rrf_score" in fused[0]
    assert fused[0]["rrf_score"] >= fused[-1]["rrf_score"]

    # 4. Test Cross-Encoder Reranking
    reranked = rerank("decorator", fused, top_k=3)
    assert len(reranked) <= 3
    assert "rerank_score" in reranked[0]
    assert reranked[0]["rerank_score"] >= reranked[-1]["rerank_score"]
