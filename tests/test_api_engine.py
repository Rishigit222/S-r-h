"""API Integration Tests for the Self-Healing RAG Engine endpoints."""

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


def test_engine_status_endpoint():
    resp = client.get("/engine/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "operational"
    assert "health_score" in data
    assert "active_parameters" in data
    assert "recent_checkpoints" in data


def test_engine_telemetry_endpoint():
    resp = client.get("/engine/telemetry")
    assert resp.status_code == 200
    data = resp.json()
    assert "summary" in data
    assert "recent_logs" in data
    assert "recent_heals" in data


def test_engine_drift_endpoint():
    resp = client.get("/engine/drift")
    assert resp.status_code == 200
    data = resp.json()
    assert "metrics" in data
    assert "drift_detected" in data


def test_engine_checkpoints_list_and_restore():
    # Save a test checkpoint
    cp = rollback_controller.save_checkpoint(
        pipeline_config={"relevance_threshold": 0.42, "vector_weight": 0.7},
        reason="API checkpoint test",
    )
    cp_id = cp.checkpoint_id

    # List checkpoints
    resp = client.get("/engine/checkpoints")
    assert resp.status_code == 200
    data = resp.json()
    assert "checkpoints" in data
    assert len(data["checkpoints"]) > 0

    # Restore checkpoint
    restore_resp = client.post(
        "/engine/checkpoints/restore",
        json={"checkpoint_id": cp_id},
    )
    assert restore_resp.status_code == 200
    restore_data = restore_resp.json()
    assert restore_data["status"] == "restored"
    assert restore_data["checkpoint_id"] == cp_id
    assert restore_data["active_parameters"]["relevance_threshold"] == 0.42


def test_engine_auto_tune_endpoint():
    resp = client.post("/engine/auto-tune")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "optimized"
    assert "parameters" in data


def test_engine_heal_endpoint():
    heal_resp = client.post(
        "/engine/heal",
        json={"query": "What is self-healing RAG?", "top_k": 3},
    )
    assert heal_resp.status_code == 200
    heal_data = heal_resp.json()
    assert "heal_id" in heal_data
    assert "query" in heal_data
    assert "action_taken" in heal_data
    assert "total_duration_ms" in heal_data
