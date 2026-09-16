"""Unit tests for observability subsystems: Telemetry and Drift Monitor."""

import pytest
from src.observability.drift_monitor import drift_monitor


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
