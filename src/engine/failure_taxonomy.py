"""Failure Taxonomy — Structured classification of RAG pipeline failures.

Defines the canonical failure types, severity levels, and failure report
structures used by the Diagnosis Engine and Repair Planner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class FailureType(str, Enum):
    """Canonical failure categories for RAG pipelines."""

    RETRIEVAL_FAILURE = "retrieval_failure"
    """No relevant documents retrieved, low recall, or empty result set."""

    GROUNDING_FAILURE = "grounding_failure"
    """Answer is not grounded in the retrieved context (hallucination)."""

    GENERATION_FAILURE = "generation_failure"
    """LLM produced incoherent, empty, or error output."""

    CITATION_FAILURE = "citation_failure"
    """Citations do not map to actual source chunks."""

    LATENCY_FAILURE = "latency_failure"
    """Pipeline exceeded the configured latency SLA."""

    CONFIG_FAILURE = "config_failure"
    """Misconfigured thresholds, weights, or parameters causing systemic issues."""


class Severity(str, Enum):
    """Failure severity levels."""

    CRITICAL = "critical"
    """System is non-functional or producing dangerous outputs."""

    HIGH = "high"
    """Significant quality degradation affecting most queries."""

    MEDIUM = "medium"
    """Noticeable quality issues affecting some queries."""

    LOW = "low"
    """Minor quality issues, system mostly functional."""

    INFO = "info"
    """Informational — potential issue detected but not yet impacting quality."""


@dataclass
class FailureEvidence:
    """Concrete evidence supporting a failure classification."""

    metric_name: str
    """Name of the metric or check that surfaced this evidence."""

    observed_value: float
    """The observed value of the metric."""

    threshold: float
    """The threshold the metric was compared against."""

    details: dict[str, Any] = field(default_factory=dict)
    """Additional context (e.g., which chunks, which query, trace excerpt)."""

    @property
    def delta(self) -> float:
        """How far the observed value is from the threshold."""
        return abs(self.observed_value - self.threshold)


@dataclass
class FailureReport:
    """A structured report describing a detected failure in the RAG pipeline."""

    failure_type: FailureType
    """The classified category of this failure."""

    severity: Severity
    """How severe this failure is."""

    evidence: list[FailureEvidence]
    """Concrete evidence items supporting this classification."""

    query: str
    """The query that triggered this failure."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """When this failure was detected."""

    pipeline_trace: dict[str, Any] = field(default_factory=dict)
    """The full pipeline execution trace at the time of failure."""

    heal_attempt: int = 0
    """Which heal attempt this failure was detected on (0 = initial)."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for logging/storage."""
        return {
            "failure_type": self.failure_type.value,
            "severity": self.severity.value,
            "evidence": [
                {
                    "metric": e.metric_name,
                    "observed": e.observed_value,
                    "threshold": e.threshold,
                    "delta": e.delta,
                    "details": e.details,
                }
                for e in self.evidence
            ],
            "query": self.query,
            "timestamp": self.timestamp.isoformat(),
            "heal_attempt": self.heal_attempt,
        }


def classify_failure(
    *,
    relevance_passed: bool = True,
    relevance_score: float = 1.0,
    relevance_threshold: float = 0.3,
    faithfulness_passed: bool = True,
    faithfulness_score: float = 1.0,
    faithfulness_threshold: float = 0.6,
    citations_passed: bool = True,
    citations_detail: str = "",
    generation_error: str | None = None,
    latency_ms: float = 0.0,
    latency_sla_ms: float = 10000.0,
    query: str = "",
    pipeline_trace: dict[str, Any] | None = None,
    heal_attempt: int = 0,
) -> FailureReport | None:
    """Analyze pipeline outputs and classify any detected failure.

    Returns a FailureReport if a failure is detected, or None if all checks pass.
    The function prioritizes failures by severity:
      generation > retrieval > grounding > citation > latency > config.
    """
    evidence: list[FailureEvidence] = []

    # 1. Generation failure (most critical — no output at all)
    if generation_error:
        evidence.append(FailureEvidence(
            metric_name="generation_error",
            observed_value=0.0,
            threshold=1.0,
            details={"error": generation_error},
        ))
        return FailureReport(
            failure_type=FailureType.GENERATION_FAILURE,
            severity=Severity.CRITICAL,
            evidence=evidence,
            query=query,
            pipeline_trace=pipeline_trace or {},
            heal_attempt=heal_attempt,
        )

    # 2. Retrieval failure (nothing relevant found)
    if not relevance_passed:
        evidence.append(FailureEvidence(
            metric_name="relevance_score",
            observed_value=relevance_score,
            threshold=relevance_threshold,
        ))
        severity = Severity.HIGH if relevance_score < relevance_threshold * 0.5 else Severity.MEDIUM
        return FailureReport(
            failure_type=FailureType.RETRIEVAL_FAILURE,
            severity=severity,
            evidence=evidence,
            query=query,
            pipeline_trace=pipeline_trace or {},
            heal_attempt=heal_attempt,
        )

    # 3. Grounding failure (answer not faithful to context)
    if not faithfulness_passed:
        evidence.append(FailureEvidence(
            metric_name="faithfulness_score",
            observed_value=faithfulness_score,
            threshold=faithfulness_threshold,
        ))
        severity = Severity.HIGH if faithfulness_score < 0.3 else Severity.MEDIUM
        return FailureReport(
            failure_type=FailureType.GROUNDING_FAILURE,
            severity=severity,
            evidence=evidence,
            query=query,
            pipeline_trace=pipeline_trace or {},
            heal_attempt=heal_attempt,
        )

    # 4. Citation failure
    if not citations_passed:
        evidence.append(FailureEvidence(
            metric_name="citation_validity",
            observed_value=0.0,
            threshold=1.0,
            details={"reason": citations_detail},
        ))
        return FailureReport(
            failure_type=FailureType.CITATION_FAILURE,
            severity=Severity.MEDIUM,
            evidence=evidence,
            query=query,
            pipeline_trace=pipeline_trace or {},
            heal_attempt=heal_attempt,
        )

    # 5. Latency failure
    if latency_ms > latency_sla_ms:
        evidence.append(FailureEvidence(
            metric_name="latency_ms",
            observed_value=latency_ms,
            threshold=latency_sla_ms,
        ))
        severity = Severity.HIGH if latency_ms > latency_sla_ms * 2 else Severity.LOW
        return FailureReport(
            failure_type=FailureType.LATENCY_FAILURE,
            severity=severity,
            evidence=evidence,
            query=query,
            pipeline_trace=pipeline_trace or {},
            heal_attempt=heal_attempt,
        )

    # All checks passed
    return None


def classify_config_drift(
    *,
    hallucination_rate: float,
    refusal_rate: float,
    avg_latency_ms: float,
    query: str = "<system_audit>",
    hallucination_threshold: float = 0.3,
    refusal_threshold: float = 0.5,
    latency_threshold_ms: float = 5000.0,
) -> FailureReport | None:
    """Detect configuration-level failures from aggregate telemetry metrics.

    Used by the continuous monitor to detect systemic drift.
    """
    evidence: list[FailureEvidence] = []

    if hallucination_rate > hallucination_threshold:
        evidence.append(FailureEvidence(
            metric_name="hallucination_rate",
            observed_value=hallucination_rate,
            threshold=hallucination_threshold,
        ))

    if refusal_rate > refusal_threshold:
        evidence.append(FailureEvidence(
            metric_name="refusal_rate",
            observed_value=refusal_rate,
            threshold=refusal_threshold,
        ))

    if avg_latency_ms > latency_threshold_ms:
        evidence.append(FailureEvidence(
            metric_name="avg_latency_ms",
            observed_value=avg_latency_ms,
            threshold=latency_threshold_ms,
        ))

    if evidence:
        worst_severity = Severity.HIGH if hallucination_rate > hallucination_threshold else Severity.MEDIUM
        return FailureReport(
            failure_type=FailureType.CONFIG_FAILURE,
            severity=worst_severity,
            evidence=evidence,
            query=query,
        )

    return None
