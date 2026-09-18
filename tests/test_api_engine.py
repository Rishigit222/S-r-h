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


def test_model_manager_caching_and_lifecycle():
    from src.models.manager import model_manager
    from src.config import settings
    model = model_manager.get(settings.embedding_model)
    assert model is not None
    assert model_manager.is_loaded(settings.embedding_model) is True
    status = model_manager.status()
    assert settings.embedding_model in status["loaded_models"]
    assert settings.embedding_model in status["load_times"]


def test_embedder_generation_and_logging():
    from src.ingestion.embedder import embed_texts, embed_query
    # Test batch embedding
    embs = embed_texts(["Test sentence for embedding verification.", "Second test sentence."])
    assert len(embs) == 2
    assert embs.shape[1] == 384
    # Test empty batch
    empty_embs = embed_texts([])
    assert len(empty_embs) == 0
    # Test query embedding
    q_emb = embed_query("What is self-healing RAG?")
    assert len(q_emb) == 384


def test_ingest_upload_endpoint():
    from pathlib import Path
    from src.config import settings
    test_content = (
        b"# Render Upload Test Verification\n\n"
        b"Verifying dynamic upload and embedding on Render Free. "
        b"This content must be long enough to exceed one hundred characters for document loaders."
    )
    uploaded_path = Path(settings.documents_dir) / "test_upload_doc.md"
    try:
        files = {"file": ("test_upload_doc.md", test_content, "text/markdown")}
        resp = client.post("/ingest/upload", files=files)
        assert resp.status_code == 200
        data = resp.json()
        assert "documents_loaded" in data
        assert data["documents_loaded"] > 0
        assert data["chunks_created"] > 0
        assert data["chunks_indexed"] > 0
    finally:
        if uploaded_path.exists():
            uploaded_path.unlink()


@pytest.mark.asyncio
async def test_startup_lifespan_is_non_blocking():
    """Verify that lifespan __aenter__ yields in milliseconds without blocking on model loading."""
    import time
    from src.api.main import lifespan
    start = time.perf_counter()
    async with lifespan(app):
        elapsed = time.perf_counter() - start
        # Startup must yield in less than 500ms to guarantee Render port detection
        assert elapsed < 0.5, f"Lifespan took {elapsed:.3f}s; must be < 0.5s for Render port detection"


def test_concurrent_model_requests_single_flight():
    """Verify that concurrent requests for the same model only load once without race conditions."""
    import threading
    from src.models.manager import ModelManager
    mgr = ModelManager(max_models=2)
    load_count = 0
    lock = threading.Lock()

    def mock_load(model_name):
        nonlocal load_count
        import time
        time.sleep(0.05)
        with lock:
            load_count += 1
        return f"mock_model_{model_name}"

    mgr._load_model = mock_load
    results = [None] * 5

    def worker(idx):
        results[idx] = mgr.get("test-embedder")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert load_count == 1, f"Expected exactly 1 model load, got {load_count}"
    assert all(r == "mock_model_test-embedder" for r in results)


def test_request_waits_during_warmup():
    """Verify that when a model is being warmed up, another thread waits and receives the warm model."""
    import threading
    import time
    from src.models.manager import ModelManager
    mgr = ModelManager(max_models=2)

    def slow_load(model_name):
        time.sleep(0.1)
        return "slow_loaded_model"

    mgr._load_model = slow_load

    # Start warmup in background
    warmup_thread = threading.Thread(target=mgr.warmup, args=("slow-model",))
    warmup_thread.start()

    time.sleep(0.02)  # Give warmup time to acquire lock and begin
    assert mgr.is_loading("slow-model") is True

    # Request arrives while warmup is in progress
    waiter_result = []
    def waiter():
        waiter_result.append(mgr.get("slow-model", timeout=2.0))

    req_thread = threading.Thread(target=waiter)
    req_thread.start()
    req_thread.join()
    warmup_thread.join()

    assert len(waiter_result) == 1
    assert waiter_result[0] == "slow_loaded_model"


def test_model_loading_failure_does_not_hang():
    """Verify that if model loading fails, callers receive a RuntimeError immediately rather than hanging."""
    from src.models.manager import ModelManager
    mgr = ModelManager(max_models=2)

    def failing_load(model_name):
        raise ValueError("Corrupt weights file simulated")

    mgr._load_model = failing_load

    with pytest.raises(ValueError, match="Corrupt weights file simulated"):
        mgr.get("failing-model")

    # Subsequent call immediately gets cached RuntimeError without hanging
    with pytest.raises(RuntimeError, match="failed to load"):
        mgr.get("failing-model")



