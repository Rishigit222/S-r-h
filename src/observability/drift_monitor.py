"""LLM Quality, Telemetry & Drift Monitoring Platform.

Logs real-time production inference metrics into an SQLite telemetry store:
- Faithfulness trends & hallucination drift over time
- Latency percentiles (P50, P90)
- Guardrail pass/refusal distributions
- Query volume and cache efficiency
- Self-healing event tracking and repair audit logs
"""

import sqlite3
import time
from pathlib import Path
from dataclasses import dataclass
from rich.console import Console

console = Console()

TELEMETRY_DB_PATH = Path("./data/telemetry.sqlite")


@dataclass
class TelemetryRecord:
    trace_id: str
    query: str
    faithfulness_score: float
    hallucination_risk: float
    guardrail_passed: bool
    cache_hit: bool
    multi_hop_used: bool
    latency_ms: float
    timestamp: float


class DriftTelemetryMonitor:
    """Manages real-time LLM quality, performance telemetry, and heal event logging."""

    def __init__(self, db_path: Path = TELEMETRY_DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS telemetry_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    faithfulness_score REAL NOT NULL,
                    hallucination_risk REAL NOT NULL,
                    guardrail_passed INTEGER NOT NULL,
                    cache_hit INTEGER NOT NULL,
                    multi_hop_used INTEGER NOT NULL,
                    latency_ms REAL NOT NULL,
                    timestamp REAL NOT NULL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS heal_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    heal_id TEXT NOT NULL,
                    trace_id TEXT,
                    query TEXT NOT NULL,
                    failure_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    diagnosis TEXT,
                    strategy_applied TEXT,
                    verdict TEXT,
                    action_taken TEXT NOT NULL,
                    duration_ms REAL NOT NULL,
                    timestamp REAL NOT NULL
                )
            """)
            conn.commit()

    def log_query(
        self,
        trace_id: str,
        query: str,
        faithfulness: float,
        hallucination_risk: float,
        guardrail_passed: bool,
        cache_hit: bool,
        multi_hop_used: bool,
        latency_ms: float,
    ):
        """Record an inference event in telemetry."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO telemetry_logs (
                    trace_id, query, faithfulness_score, hallucination_risk,
                    guardrail_passed, cache_hit, multi_hop_used, latency_ms, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace_id, query, faithfulness, hallucination_risk,
                    1 if guardrail_passed else 0,
                    1 if cache_hit else 0,
                    1 if multi_hop_used else 0,
                    latency_ms, time.time()
                )
            )
            conn.commit()

    def log_heal_event(
        self,
        heal_id: str,
        query: str,
        failure_type: str,
        severity: str,
        action_taken: str,
        duration_ms: float,
        trace_id: str = "",
        diagnosis: str = "",
        strategy_applied: str = "",
        verdict: str = "",
    ):
        """Record a self-healing event in telemetry."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO heal_events (
                    heal_id, trace_id, query, failure_type, severity,
                    diagnosis, strategy_applied, verdict, action_taken,
                    duration_ms, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    heal_id, trace_id, query, failure_type, severity,
                    diagnosis, strategy_applied, verdict, action_taken,
                    duration_ms, time.time()
                )
            )
            conn.commit()

    def get_summary_metrics(self) -> dict:
        """Calculate high-level quality & drift metrics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM telemetry_logs")
            total = cursor.fetchone()[0]

            if total == 0:
                return {
                    "total_queries": 0,
                    "avg_faithfulness": 1.0,
                    "hallucination_rate_pct": 0.0,
                    "cache_hit_rate_pct": 0.0,
                    "guardrail_pass_rate_pct": 100.0,
                    "avg_latency_ms": 0.0,
                    "total_heal_events": 0,
                    "heal_deploy_rate_pct": 0.0,
                }

            cursor.execute("SELECT AVG(faithfulness_score), AVG(latency_ms), AVG(cache_hit), AVG(guardrail_passed) FROM telemetry_logs")
            avg_faith, avg_lat, avg_cache, avg_passed = cursor.fetchone()

            # Heal event stats
            cursor.execute("SELECT COUNT(*) FROM heal_events")
            total_heals = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM heal_events WHERE action_taken = 'deployed'")
            deployed_heals = cursor.fetchone()[0]

            return {
                "total_queries": total,
                "avg_faithfulness": round(avg_faith or 1.0, 3),
                "hallucination_rate_pct": round((1.0 - (avg_faith or 1.0)) * 100, 1),
                "cache_hit_rate_pct": round((avg_cache or 0.0) * 100, 1),
                "guardrail_pass_rate_pct": round((avg_passed or 1.0) * 100, 1),
                "avg_latency_ms": round(avg_lat or 0.0, 1),
                "total_heal_events": total_heals,
                "heal_deploy_rate_pct": round((deployed_heals / total_heals * 100) if total_heals > 0 else 0.0, 1),
            }

    def get_recent_logs(self, limit: int = 20) -> list[dict]:
        """Fetch recent query logs for the analytics dashboard."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT trace_id, query, faithfulness_score, guardrail_passed, cache_hit, latency_ms, timestamp
                FROM telemetry_logs
                ORDER BY id DESC LIMIT ?
                """,
                (limit,)
            )
            rows = cursor.fetchall()
            return [
                {
                    "trace_id": r[0],
                    "query": r[1],
                    "faithfulness": f"{r[2]:.0%}",
                    "passed": bool(r[3]),
                    "cache_hit": bool(r[4]),
                    "latency_ms": f"{r[5]:.1f} ms",
                    "time": time.strftime("%H:%M:%S", time.localtime(r[6])),
                }
                for r in rows
            ]

    def get_recent_heal_events(self, limit: int = 20) -> list[dict]:
        """Fetch recent self-healing events."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT heal_id, query, failure_type, severity, strategy_applied,
                       verdict, action_taken, duration_ms, timestamp
                FROM heal_events
                ORDER BY id DESC LIMIT ?
                """,
                (limit,)
            )
            rows = cursor.fetchall()
            return [
                {
                    "heal_id": r[0],
                    "query": r[1][:80],
                    "failure_type": r[2],
                    "severity": r[3],
                    "strategy": r[4],
                    "verdict": r[5],
                    "action": r[6],
                    "duration_ms": f"{r[7]:.0f}ms",
                    "time": time.strftime("%H:%M:%S", time.localtime(r[8])),
                }
                for r in rows
            ]

    def get_drift_assessment(self) -> dict:
        """Assess current system health and detect configuration drift.

        Integrates with the engine's failure taxonomy for systemic detection.
        """
        from src.engine.failure_taxonomy import classify_config_drift

        metrics = self.get_summary_metrics()
        failure = classify_config_drift(
            hallucination_rate=metrics["hallucination_rate_pct"] / 100.0,
            refusal_rate=(100.0 - metrics["guardrail_pass_rate_pct"]) / 100.0,
            avg_latency_ms=metrics["avg_latency_ms"],
        )
        return {
            "metrics": metrics,
            "drift_detected": failure is not None,
            "drift_report": failure.to_dict() if failure else None,
        }


drift_monitor = DriftTelemetryMonitor()
