"""Decision Traceability, Explainability & Observability.

Records complete audit trails for every query:
- Explains why chunks were retrieved and reranked
- Records gate decisions and confidence thresholds
- Transparent latency breakdown across all pipeline stages
- Self-healing cycle tracking and repair audit trails
"""

import time
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class StepTrace:
    step_name: str
    duration_ms: float
    status: str
    details: dict = field(default_factory=dict)


@dataclass
class HealStepTrace:
    """Trace entry for a self-healing repair attempt."""
    attempt: int
    strategy: str
    verdict: str
    overall_delta: float
    duration_ms: float
    details: dict = field(default_factory=dict)


@dataclass
class ExecutionTrace:
    trace_id: str
    query: str
    total_duration_ms: float = 0.0
    cache_hit: bool = False
    security_passed: bool = True
    multi_hop_used: bool = False
    rerank_best_score: float = 0.0
    faithfulness_score: float = 0.0
    citations_count: int = 0
    steps: list[StepTrace] = field(default_factory=list)

    # Self-healing engine fields
    failure_detected: bool = False
    failure_type: str | None = None
    failure_severity: str | None = None
    diagnosis_root_cause: str | None = None
    heal_attempts: list[HealStepTrace] = field(default_factory=list)
    heal_action_taken: str = "none"

    def to_dict(self) -> dict:
        return asdict(self)


class QueryTracer:
    """Collects execution metrics and explainability data per query."""

    def __init__(self, trace_id: str, query: str):
        self.trace = ExecutionTrace(trace_id=trace_id, query=query)
        self.start_time = time.time()
        self._current_step_start = time.time()

    def record_step(self, name: str, status: str = "success", details: dict | None = None):
        now = time.time()
        duration_ms = (now - self._current_step_start) * 1000
        self.trace.steps.append(
            StepTrace(
                step_name=name,
                duration_ms=round(duration_ms, 2),
                status=status,
                details=details or {},
            )
        )
        self._current_step_start = now

    def record_failure(
        self,
        failure_type: str,
        severity: str,
        root_cause: str,
    ):
        """Record a detected failure and its diagnosis."""
        self.trace.failure_detected = True
        self.trace.failure_type = failure_type
        self.trace.failure_severity = severity
        self.trace.diagnosis_root_cause = root_cause

    def record_heal_attempt(
        self,
        attempt: int,
        strategy: str,
        verdict: str,
        overall_delta: float,
        duration_ms: float,
        details: dict | None = None,
    ):
        """Record a self-healing repair attempt and its outcome."""
        self.trace.heal_attempts.append(
            HealStepTrace(
                attempt=attempt,
                strategy=strategy,
                verdict=verdict,
                overall_delta=overall_delta,
                duration_ms=duration_ms,
                details=details or {},
            )
        )

    def record_heal_action(self, action: str):
        """Record the final action taken: 'deployed', 'rolled_back', 'exhausted', 'none'."""
        self.trace.heal_action_taken = action

    def finalize(self) -> dict:
        self.trace.total_duration_ms = round((time.time() - self.start_time) * 1000, 2)
        return self.trace.to_dict()
