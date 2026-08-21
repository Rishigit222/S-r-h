"""Unit tests for Phase 3 (Guardrails & Self-Healing)."""

import pytest
from src.guardrails.relevance_gate import check_relevance
from src.guardrails.faithfulness_checker import check_faithfulness
from src.guardrails.citation_verifier import verify_citations
from src.guardrails.self_healer import rewrite_query, should_heal, build_refusal_response


def test_relevance_gate_pass():
    """Verify high-scoring reranked results pass the relevance gate."""
    high_score_results = [{"content": "Decorators are functions...", "rerank_score": 3.5}]
    result = check_relevance(high_score_results)
    assert result.passed is True
    assert result.best_score == 3.5


def test_relevance_gate_block():
    """Verify low-scoring results get blocked by the relevance gate."""
    low_score_results = [{"content": "Unrelated topic...", "rerank_score": -5.0}]
    result = check_relevance(low_score_results)
    assert result.passed is False
    assert "not relevant enough" in result.reason


def test_citation_verifier():
    """Verify citation validator detects valid and invalid citations."""
    good_answer = "Decorators extend behavior [1] and can be used for caching [2]."
    good_res = verify_citations(good_answer, num_context_chunks=3)
    assert good_res.passed is True
    assert good_res.cited_sources == [1, 2]

    out_of_bounds_answer = "This claims source [5]."
    bad_res = verify_citations(out_of_bounds_answer, num_context_chunks=2)
    assert bad_res.passed is False

    refusal_answer = "I don't have enough information to answer this question."
    refusal_res = verify_citations(refusal_answer, num_context_chunks=2)
    assert refusal_res.passed is True


def test_self_healing_logic():
    """Verify retry limit and refusal message generation."""
    assert should_heal(0) is True
    assert should_heal(1) is True
    assert should_heal(2) is False

    refusal = build_refusal_response("What is quantum gravity?", "No context found")
    assert "What is quantum gravity?" in refusal
    assert "Reason: No context found" in refusal
