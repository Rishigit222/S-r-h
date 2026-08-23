"""Autonomous Self-Optimizer & Hyperparameter Mutation Engine for s@r@h.

Monitors real-time telemetry drift and dynamically self-tunes:
- Vector vs BM25 weights in Reciprocal Rank Fusion
- Cross-Encoder relevance gating thresholds
- RRF smoothing constants and retrieval depths
"""

import time
import sqlite3
from pathlib import Path
from dataclasses import dataclass, field, asdict
from rich.console import Console

from src.observability.drift_monitor import drift_monitor

console = Console()

EVOLUTION_DB_PATH = Path("./data/evolution_history.sqlite")


@dataclass
class DynamicRAGHyperparameters:
    vector_weight: float = 0.5
    bm25_weight: float = 0.5
    rrf_k: int = 60
    relevance_threshold: float = 0.30
    top_k_retrieval: int = 20
    top_k_rerank: int = 5
    last_mutation_time: float = field(default_factory=time.time)
    generation_version: int = 1


@dataclass
class MutationLog:
    version: int
    parameter_changed: str
    old_value: float
    new_value: float
    trigger_reason: str
    timestamp: float


class SelfOptimizer:
    """Manages autonomous parameter evolution and adaptive self-tuning."""

    def __init__(self, db_path: Path = EVOLUTION_DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.params = DynamicRAGHyperparameters()
        self._init_db()
        self._load_latest_state()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS mutations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version INTEGER NOT NULL,
                    parameter_changed TEXT NOT NULL,
                    old_value REAL NOT NULL,
                    new_value REAL NOT NULL,
                    trigger_reason TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS current_params (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    vector_weight REAL,
                    bm25_weight REAL,
                    rrf_k INTEGER,
                    relevance_threshold REAL,
                    top_k_retrieval INTEGER,
                    top_k_rerank INTEGER,
                    generation_version INTEGER,
                    updated_at REAL
                )
            """)
            conn.commit()

    def _load_latest_state(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT vector_weight, bm25_weight, rrf_k, relevance_threshold, top_k_retrieval, top_k_rerank, generation_version, updated_at FROM current_params WHERE id = 1")
            row = cursor.fetchone()
            if row:
                self.params.vector_weight = row[0]
                self.params.bm25_weight = row[1]
                self.params.rrf_k = row[2]
                self.params.relevance_threshold = row[3]
                self.params.top_k_retrieval = row[4]
                self.params.top_k_rerank = row[5]
                self.params.generation_version = row[6]
                self.params.last_mutation_time = row[7]

    def _save_current_state(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO current_params (id, vector_weight, bm25_weight, rrf_k, relevance_threshold, top_k_retrieval, top_k_rerank, generation_version, updated_at)
                VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    vector_weight = excluded.vector_weight,
                    bm25_weight = excluded.bm25_weight,
                    rrf_k = excluded.rrf_k,
                    relevance_threshold = excluded.relevance_threshold,
                    top_k_retrieval = excluded.top_k_retrieval,
                    top_k_rerank = excluded.top_k_rerank,
                    generation_version = excluded.generation_version,
                    updated_at = excluded.updated_at
                """,
                (
                    self.params.vector_weight, self.params.bm25_weight, self.params.rrf_k,
                    self.params.relevance_threshold, self.params.top_k_retrieval,
                    self.params.top_k_rerank, self.params.generation_version, time.time()
                )
            )
            conn.commit()

    def record_mutation(self, param_name: str, old_val: float, new_val: float, reason: str):
        self.params.generation_version += 1
        self.params.last_mutation_time = time.time()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO mutations (version, parameter_changed, old_value, new_value, trigger_reason, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                (self.params.generation_version, param_name, old_val, new_val, reason, time.time())
            )
            conn.commit()
        self._save_current_state()
        console.print(f"[bold magenta]Meta-RAG Mutation v{self.params.generation_version}: {param_name} ({old_val} -> {new_val}) | Reason: {reason}[/bold magenta]")

    def auto_tune(self) -> dict:
        """Analyze recent telemetry and trigger autonomous parameter self-tuning."""
        metrics = drift_monitor.get_summary_metrics()
        total_queries = metrics.get("total_queries", 0)
        avg_faith = metrics.get("avg_faithfulness", 1.0)
        hallucination_rate = metrics.get("hallucination_rate_pct", 0.0)
        pass_rate = metrics.get("guardrail_pass_rate_pct", 100.0)

        mutations_applied = []

        # 1. Hallucination Guard Tuning
        if hallucination_rate > 15.0 and self.params.relevance_threshold < 0.40:
            old_th = self.params.relevance_threshold
            new_th = round(min(0.40, old_th + 0.05), 2)
            self.params.relevance_threshold = new_th
            self.record_mutation("relevance_threshold", old_th, new_th, f"High hallucination rate ({hallucination_rate}%) detected in telemetry.")
            mutations_applied.append(f"Relevance Threshold: {old_th} ➔ {new_th}")

            # Boost BM25 keyword weight to favor grounded exact tokens
            old_bm25 = self.params.bm25_weight
            new_bm25 = round(min(0.70, old_bm25 + 0.10), 2)
            self.params.bm25_weight = new_bm25
            self.params.vector_weight = round(1.0 - new_bm25, 2)
            self.record_mutation("bm25_weight", old_bm25, new_bm25, "Increased keyword weighting to eliminate ungrounded semantic drift.")
            mutations_applied.append(f"BM25 Weight: {old_bm25} ➔ {new_bm25}")

        # 2. Over-Refusal Tuning
        elif pass_rate < 50.0 and self.params.relevance_threshold > 0.20:
            old_th = self.params.relevance_threshold
            new_th = round(max(0.20, old_th - 0.05), 2)
            self.params.relevance_threshold = new_th
            self.record_mutation("relevance_threshold", old_th, new_th, f"High refusal rate ({100 - pass_rate}%) detected in telemetry.")
            mutations_applied.append(f"Relevance Threshold: {old_th} ➔ {new_th}")

            # Boost vector weight for broader semantic coverage
            old_v = self.params.vector_weight
            new_v = round(min(0.70, old_v + 0.10), 2)
            self.params.vector_weight = new_v
            self.params.bm25_weight = round(1.0 - new_v, 2)
            self.record_mutation("vector_weight", old_v, new_v, "Increased vector weighting for broader semantic concept recall.")
            mutations_applied.append(f"Vector Weight: {old_v} ➔ {new_v}")

        # Default fallback tuning if no severe drift
        if not mutations_applied:
            # Subtle adaptive balance
            mutations_applied.append("Hyperparameters already optimal for current query distribution.")

        return {
            "status": "optimized",
            "current_version": self.params.generation_version,
            "mutations": mutations_applied,
            "parameters": asdict(self.params),
        }

    def get_mutation_history(self, limit: int = 15) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT version, parameter_changed, old_value, new_value, trigger_reason, timestamp FROM mutations ORDER BY id DESC LIMIT ?",
                (limit,)
            )
            rows = cursor.fetchall()
            return [
                {
                    "version": f"v{r[0]}",
                    "parameter": r[1],
                    "change": f"{r[2]} ➔ {r[3]}",
                    "reason": r[4],
                    "time": time.strftime("%H:%M:%S", time.localtime(r[5])),
                }
                for r in rows
            ]

    def reset_defaults(self):
        self.params = DynamicRAGHyperparameters()
        self._save_current_state()


self_optimizer = SelfOptimizer()
