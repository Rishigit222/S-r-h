"""Unit tests for reusable RAG pipeline components: Semantic Cache and Observability Tracer."""

import pytest
from src.optimization.semantic_cache import semantic_cache
from src.observability.tracer import QueryTracer


def test_semantic_cache():
    """Verify semantic cache hit on identical or similar queries."""
    semantic_cache.clear()
    query = "What is a Python decorator?"
    mock_resp = {"answer": "A decorator wraps a function.", "faithfulness_score": 1.0}

    semantic_cache.store(query, mock_resp)
    assert len(semantic_cache.entries) == 1

    hit_resp, sim = semantic_cache.lookup("What is a Python decorator?")
    assert hit_resp is not None
    assert sim >= 0.95
    assert semantic_cache.stats()["hits"] == 1


def test_observability_tracer():
    """Verify query execution tracer records step durations and status."""
    tracer = QueryTracer(trace_id="test-trace-01", query="test query")
    tracer.record_step("Security Check", "passed", {"threats": 0})
    tracer.record_step("Retrieval", "success", {"chunks": 5})
    trace_data = tracer.finalize()

    assert trace_data["trace_id"] == "test-trace-01"
    assert len(trace_data["steps"]) == 2
    assert trace_data["total_duration_ms"] > 0
