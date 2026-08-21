"""Unit tests for s@r@h specific subsystems: GraphRAG, Telemetry, Memory Agent, and Auth."""

import pytest
from src.graph.knowledge_graph import knowledge_graph, GraphTriple
from src.observability.drift_monitor import drift_monitor
from src.memory.episodic_memory import memory_store
from src.api.auth import auth_manager, ApiKeyVaultRequest


def test_auth_session_and_vault():
    """Verify session token creation and API key storage."""
    session = auth_manager.create_session("rishi_dev", "AI Architect")
    assert session.token is not None
    assert session.username == "rishi_dev"

    saved = auth_manager.save_api_keys(ApiKeyVaultRequest(
        token=session.token,
        groq_api_key="gsk_test123",
    ))
    assert saved is True

    keys = auth_manager.get_api_keys(session.token)
    assert keys.get("groq_configured") is True


def test_graph_rag_ai_seed_triples():
    """Verify Knowledge Graph retrieval on AI/ML entities."""
    results = knowledge_graph.query_graph("DeepSeek R1 GRPO")
    assert len(results) > 0
    assert any("GRPO" in r["text"] or "DeepSeek" in r["text"] for r in results)

    transformer_res = knowledge_graph.query_graph("Transformer Attention")
    assert len(transformer_res) > 0


def test_telemetry_drift_monitor():
    """Verify real-time inference telemetry logging and metric computation."""
    drift_monitor.log_query(
        trace_id="sarah-test-01",
        query="What is DeepSeek R1?",
        faithfulness=1.0,
        hallucination_risk=0.0,
        guardrail_passed=True,
        cache_hit=False,
        multi_hop_used=False,
        latency_ms=120.5,
    )

    summary = drift_monitor.get_summary_metrics()
    assert summary["total_queries"] >= 1
    assert summary["avg_faithfulness"] > 0.0

    recent = drift_monitor.get_recent_logs(5)
    assert len(recent) >= 1
    assert recent[0]["trace_id"] == "sarah-test-01"


def test_memory_agent_auto_extraction():
    """Verify automatic user fact extraction from conversation turns."""
    session_id = "sarah_user_auto"
    memory_store.clear_session(session_id)

    memory_store.add_message(session_id, "user", "Hello, my name is Alex and I am a software engineer.")
    facts = memory_store.get_user_facts(session_id)

    assert facts.get("user_name") == "Alex"
    assert facts.get("user_role") == "software engineer"
