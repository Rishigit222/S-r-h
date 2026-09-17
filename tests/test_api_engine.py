"""API Integration Tests for the Self-Healing RAG Engine public endpoints."""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.engine.rollback_controller import rollback_controller
from src.evolution.self_optimizer import self_optimizer

client = TestClient(app)


def test_root_status():
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["engine"] == "Self-Healing RAG Engine"
    assert data["status"] == "running"


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["engine_name"] == "Self-Healing RAG Engine"
    assert "generation_version" in data
    assert "telemetry_summary" in data
    assert "dynamic_hyperparameters" in data


def test_status_endpoint():
    resp = client.get("/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "operational"
    assert "health_score" in data
    assert "active_parameters" in data
    assert "recent_checkpoints" in data


def test_metrics_endpoint():
    resp = client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "summary" in data
    assert "recent_logs" in data
    assert "recent_heals" in data
    assert "drift" in data


def test_repairs_endpoint():
    resp = client.get("/repairs")
    assert resp.status_code == 200
    data = resp.json()
    assert "available_strategies" in data
    assert "recent_repairs" in data
    assert len(data["available_strategies"]) >= 7


def test_checkpoints_list_and_rollback():
    # Save a test checkpoint
    cp = rollback_controller.save_checkpoint(
        pipeline_config={"relevance_threshold": 0.42, "vector_weight": 0.7},
        reason="API checkpoint test",
    )
    cp_id = cp.checkpoint_id

    # List checkpoints via GET /checkpoints
    resp = client.get("/checkpoints")
    assert resp.status_code == 200
    data = resp.json()
    assert "checkpoints" in data
    assert len(data["checkpoints"]) > 0

    # Rollback via POST /rollback/{checkpoint_id}
    restore_resp = client.post(f"/rollback/{cp_id}")
    assert restore_resp.status_code == 200
    restore_data = restore_resp.json()
    assert restore_data["status"] == "restored"
    assert restore_data["checkpoint_id"] == cp_id
    assert restore_data["active_parameters"]["relevance_threshold"] == 0.42


def test_audit_endpoint():
    audit_resp = client.post("/audit", json={"config_or_code": "retriever = faiss_index.as_retriever()"})
    assert audit_resp.status_code == 200
    audit_data = audit_resp.json()
    assert audit_data["overall_health_score"] < 70
    assert len(audit_data["vulnerabilities_detected"]) > 0


def test_heal_endpoint():
    heal_resp = client.post(
        "/heal",
        json={"query": "What is self-healing RAG?", "top_k": 3},
    )
    assert heal_resp.status_code == 200
    heal_data = heal_resp.json()
    assert "heal_id" in heal_data
    assert "query" in heal_data
    assert "action_taken" in heal_data
    assert "total_duration_ms" in heal_data
