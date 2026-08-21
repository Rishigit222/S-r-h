"""Option C: LLM Quality, Telemetry & Drift Monitoring Platform.

Logs real-time production inference metrics into an SQLite telemetry store:
- Faithfulness trends & hallucination drift over time
- Latency percentiles (P50, P90)
- Guardrail pass/refusal distributions
- Query volume and cache efficiency
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
    """Manages real-time LLM quality and performance telemetry."""

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
                    "avg_latency_ms": 0.0,
                }

            cursor.execute("SELECT AVG(faithfulness_score), AVG(latency_ms), AVG(cache_hit), AVG(guardrail_passed) FROM telemetry_logs")
            avg_faith, avg_lat, avg_cache, avg_passed = cursor.fetchone()

            return {
                "total_queries": total,
                "avg_faithfulness": round(avg_faith or 1.0, 3),
                "hallucination_rate_pct": round((1.0 - (avg_faith or 1.0)) * 100, 1),
                "cache_hit_rate_pct": round((avg_cache or 0.0) * 100, 1),
                "guardrail_pass_rate_pct": round((avg_passed or 1.0) * 100, 1),
                "avg_latency_ms": round(avg_lat or 0.0, 1),
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


drift_monitor = DriftTelemetryMonitor()
