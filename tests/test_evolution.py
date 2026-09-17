"""Tests for Meta-RAG, Self-Evolution, and External RAG Modifier Engine."""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.evolution.self_optimizer import self_optimizer
from src.evolution.synthetic_trainer import synthetic_trainer
from src.evolution.meta_modifier import rag_meta_modifier

client = TestClient(app)


def test_self_optimizer_initial_state():
    params = self_optimizer.params
    assert params.vector_weight >= 0.0
    assert params.bm25_weight >= 0.0
    assert params.relevance_threshold > 0.0
    assert params.generation_version >= 1


def test_self_optimizer_record_mutation():
    initial_ver = self_optimizer.params.generation_version
    self_optimizer.record_mutation(
        param_name="relevance_threshold",
        old_val=0.30,
        new_val=0.35,
        reason="Test high hallucination defense"
    )
    assert self_optimizer.params.generation_version > initial_ver
    history = self_optimizer.get_mutation_history(5)
    assert len(history) > 0
    assert history[0]["parameter"] == "relevance_threshold"


def test_synthetic_trainer_qa_generation():
    qa_pairs = synthetic_trainer.generate_synthetic_dataset(max_pairs=5)
    assert isinstance(qa_pairs, list)
    if qa_pairs:
        first = qa_pairs[0]
        assert "?" in first.question
        assert len(first.target_entity) > 0
        assert first.difficulty in ["easy", "medium", "hard"]


def test_meta_modifier_audit_naive_rag():
    naive_rag_code = """
    from langchain.vectorstores import Chroma
    from langchain.embeddings import OpenAIEmbeddings
    retriever = Chroma.from_documents(docs, OpenAIEmbeddings()).as_retriever()
    """
    report = rag_meta_modifier.audit_and_modify(naive_rag_code)
    assert report.overall_health_score < 60
    assert len(report.vulnerabilities_detected) >= 3
    assert len(report.optimization_recommendations) >= 3
    assert "upgraded_rag_pipeline" in report.generated_code_patch
    assert "Hybrid" in report.upgraded_rag_recipe["retrieval_strategy"]


def test_meta_modifier_audit_optimized_rag():
    optimized_rag_code = """
    # Hybrid search with BM25 and Chroma vector store
    # Cross-encoder reranker with NLI guardrail and semantic cache
    # Security sanitizer and multi-hop graph decomposition
    """
    report = rag_meta_modifier.audit_and_modify(optimized_rag_code)
    assert report.overall_health_score >= 80


def test_api_evolution_endpoints():
    # 1. Status
    res = client.get("/status")
    assert res.status_code == 200
    data = res.json()
    assert "active_parameters" in data
    assert "generation_version" in data

    # 2. Auto-Tune
    tune_res = self_optimizer.auto_tune()
    assert tune_res["status"] == "optimized"

    # 3. Audit RAG API (aligned to POST /audit)
    audit_res = client.post("/audit", json={"config_or_code": "retriever = faiss_index.as_retriever()"})
    assert audit_res.status_code == 200
    audit_data = audit_res.json()
    assert audit_data["overall_health_score"] < 70
