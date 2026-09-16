"""Diagnosis Engine — Root cause analysis for RAG pipeline failures.

Maps observed failures to probable root causes and recommends repair strategies
based on evidence analysis and heuristic rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.engine.failure_taxonomy import FailureReport, FailureType, Severity


class RepairStrategyType:
    """Canonical repair strategy identifiers."""

    REWRITE_QUERY = "rewrite_query"
    """Use LLM to reformulate the user's query for better retrieval."""

    RECHUNK_DOCUMENTS = "rechunk_documents"
    """Adjust chunk_size and chunk_overlap parameters."""

    ADJUST_RETRIEVAL = "adjust_retrieval"
    """Modify top_k, bm25_weight, vector_weight, or RRF parameters."""

    EXPAND_CONTEXT = "expand_context"
    """Increase retrieval window or add web grounding fallback."""

    ADJUST_THRESHOLDS = "adjust_thresholds"
    """Tune relevance_threshold or faithfulness_threshold."""

    SWAP_RERANKER = "swap_reranker"
    """Switch reranking model or adjust reranker top_k."""

    REGENERATE = "regenerate"
    """Retry generation with a modified prompt or more context."""


@dataclass
class Diagnosis:
    """Result of root cause analysis on a failure."""

    root_cause: str
    """Human-readable description of the probable root cause."""

    confidence: float
    """Confidence level of this diagnosis (0.0 to 1.0)."""

    suggested_repairs: list[str]
    """Ordered list of RepairStrategyType identifiers to try."""

    evidence_chain: list[str]
    """Step-by-step reasoning chain explaining the diagnosis."""

    failure_report: FailureReport
    """The original failure report that was diagnosed."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "root_cause": self.root_cause,
            "confidence": self.confidence,
            "suggested_repairs": self.suggested_repairs,
            "evidence_chain": self.evidence_chain,
            "failure_type": self.failure_report.failure_type.value,
            "severity": self.failure_report.severity.value,
        }


class DiagnosisEngine:
    """Analyzes failure reports and produces root cause diagnoses.

    Uses a rule-based approach combining failure type, evidence metrics,
    and pipeline trace analysis to determine probable causes and recommend
    repair strategies in priority order.
    """

    def diagnose(self, failure: FailureReport) -> Diagnosis:
        """Analyze a failure and produce a diagnosis with repair recommendations.

        Args:
            failure: The structured failure report from the taxonomy classifier.

        Returns:
            A Diagnosis containing root cause, confidence, and ordered repair strategies.
        """
        handler = self._DIAGNOSIS_HANDLERS.get(failure.failure_type)
        if handler is None:
            return Diagnosis(
                root_cause="Unknown failure type — no diagnostic rules available.",
                confidence=0.1,
                suggested_repairs=[RepairStrategyType.REWRITE_QUERY],
                evidence_chain=["No matching diagnosis handler found."],
                failure_report=failure,
            )
        return handler(self, failure)

    # --- Diagnosis handlers per failure type ---

    def _diagnose_retrieval(self, failure: FailureReport) -> Diagnosis:
        """Diagnose retrieval failures — no relevant documents found."""
        evidence_chain: list[str] = []
        suggested_repairs: list[str] = []
        confidence = 0.7

        # Check if relevance score is very low (near zero)
        relevance_evidence = next(
            (e for e in failure.evidence if e.metric_name == "relevance_score"), None
        )
        score = relevance_evidence.observed_value if relevance_evidence else 0.0
        threshold = relevance_evidence.threshold if relevance_evidence else 0.3

        if score < 0.05:
            evidence_chain.append(
                f"Relevance score is near zero ({score:.3f}) — query likely has no matching content in the knowledge base."
            )
            suggested_repairs.extend([
                RepairStrategyType.REWRITE_QUERY,
                RepairStrategyType.EXPAND_CONTEXT,
                RepairStrategyType.RECHUNK_DOCUMENTS,
            ])
            root_cause = "Query vocabulary mismatch or content gap — no chunks contain semantically relevant information."
            confidence = 0.85
        elif score < threshold * 0.7:
            evidence_chain.append(
                f"Relevance score ({score:.3f}) is well below threshold ({threshold:.3f}) — content exists but retrieval is poor."
            )
            suggested_repairs.extend([
                RepairStrategyType.ADJUST_RETRIEVAL,
                RepairStrategyType.REWRITE_QUERY,
                RepairStrategyType.RECHUNK_DOCUMENTS,
            ])
            root_cause = "Retrieval parameters are suboptimal — chunks may be too coarse or BM25/vector weights are misaligned."
            confidence = 0.75
        else:
            evidence_chain.append(
                f"Relevance score ({score:.3f}) is close to threshold ({threshold:.3f}) — borderline retrieval quality."
            )
            suggested_repairs.extend([
                RepairStrategyType.ADJUST_THRESHOLDS,
                RepairStrategyType.REWRITE_QUERY,
                RepairStrategyType.ADJUST_RETRIEVAL,
            ])
            root_cause = "Relevance threshold may be too aggressive — slightly lowering it could pass borderline-relevant chunks."
            confidence = 0.65

        return Diagnosis(
            root_cause=root_cause,
            confidence=confidence,
            suggested_repairs=suggested_repairs,
            evidence_chain=evidence_chain,
            failure_report=failure,
        )

    def _diagnose_grounding(self, failure: FailureReport) -> Diagnosis:
        """Diagnose grounding failures — answer not faithful to context."""
        evidence_chain: list[str] = []

        faith_evidence = next(
            (e for e in failure.evidence if e.metric_name == "faithfulness_score"), None
        )
        score = faith_evidence.observed_value if faith_evidence else 0.0

        if score < 0.2:
            evidence_chain.append(
                f"Faithfulness score is very low ({score:.3f}) — LLM is largely hallucinating."
            )
            root_cause = "LLM is ignoring retrieved context. Possible causes: context window overflow, poor prompt structure, or model temperature too high."
            suggested_repairs = [
                RepairStrategyType.REGENERATE,
                RepairStrategyType.REWRITE_QUERY,
                RepairStrategyType.ADJUST_RETRIEVAL,
            ]
            confidence = 0.8
        else:
            evidence_chain.append(
                f"Faithfulness score ({score:.3f}) is below threshold — partial grounding failure."
            )
            root_cause = "Retrieved context is partially relevant but LLM is extrapolating beyond it. May need better retrieval or stricter generation prompt."
            suggested_repairs = [
                RepairStrategyType.REWRITE_QUERY,
                RepairStrategyType.REGENERATE,
                RepairStrategyType.ADJUST_THRESHOLDS,
            ]
            confidence = 0.7

        return Diagnosis(
            root_cause=root_cause,
            confidence=confidence,
            suggested_repairs=suggested_repairs,
            evidence_chain=evidence_chain,
            failure_report=failure,
        )

    def _diagnose_generation(self, failure: FailureReport) -> Diagnosis:
        """Diagnose generation failures — LLM error or empty output."""
        error_evidence = next(
            (e for e in failure.evidence if e.metric_name == "generation_error"), None
        )
        error_msg = error_evidence.details.get("error", "unknown") if error_evidence else "unknown"

        return Diagnosis(
            root_cause=f"LLM generation failed with error: {error_msg}. Possible causes: API connectivity, rate limiting, context length exceeded, or model unavailability.",
            confidence=0.9,
            suggested_repairs=[
                RepairStrategyType.REGENERATE,
                RepairStrategyType.REWRITE_QUERY,
            ],
            evidence_chain=[
                f"Generation produced error: {error_msg}",
                "This is typically a transient or infrastructure issue.",
            ],
            failure_report=failure,
        )

    def _diagnose_citation(self, failure: FailureReport) -> Diagnosis:
        """Diagnose citation failures — citations don't map to sources."""
        return Diagnosis(
            root_cause="Citation extraction or formatting issue — the LLM generated citations that don't match source chunk indices.",
            confidence=0.75,
            suggested_repairs=[
                RepairStrategyType.REGENERATE,
                RepairStrategyType.REWRITE_QUERY,
            ],
            evidence_chain=[
                "Citation verification failed.",
                "The answer may be correct but citations are malformed or reference non-existent sources.",
            ],
            failure_report=failure,
        )

    def _diagnose_latency(self, failure: FailureReport) -> Diagnosis:
        """Diagnose latency failures — pipeline too slow."""
        latency_evidence = next(
            (e for e in failure.evidence if e.metric_name == "latency_ms"), None
        )
        observed = latency_evidence.observed_value if latency_evidence else 0.0
        sla = latency_evidence.threshold if latency_evidence else 5000.0

        # Analyze trace to find bottleneck
        trace = failure.pipeline_trace
        bottleneck = "unknown stage"
        if trace and "steps" in trace:
            steps = trace["steps"]
            if steps:
                slowest = max(steps, key=lambda s: s.get("duration_ms", 0))
                bottleneck = slowest.get("name", "unknown stage")

        return Diagnosis(
            root_cause=f"Pipeline latency ({observed:.0f}ms) exceeds SLA ({sla:.0f}ms). Bottleneck identified: {bottleneck}.",
            confidence=0.85,
            suggested_repairs=[
                RepairStrategyType.ADJUST_RETRIEVAL,
                RepairStrategyType.ADJUST_THRESHOLDS,
            ],
            evidence_chain=[
                f"Total latency: {observed:.0f}ms (SLA: {sla:.0f}ms)",
                f"Slowest pipeline stage: {bottleneck}",
                "Reducing top_k or enabling caching may help.",
            ],
            failure_report=failure,
        )

    def _diagnose_config(self, failure: FailureReport) -> Diagnosis:
        """Diagnose configuration drift failures — systemic parameter issues."""
        evidence_chain: list[str] = []
        for ev in failure.evidence:
            evidence_chain.append(
                f"{ev.metric_name}: observed={ev.observed_value:.3f}, threshold={ev.threshold:.3f}"
            )

        return Diagnosis(
            root_cause="Systemic configuration drift detected — aggregate metrics indicate thresholds or weights need recalibration.",
            confidence=0.7,
            suggested_repairs=[
                RepairStrategyType.ADJUST_THRESHOLDS,
                RepairStrategyType.ADJUST_RETRIEVAL,
                RepairStrategyType.RECHUNK_DOCUMENTS,
            ],
            evidence_chain=evidence_chain,
            failure_report=failure,
        )

    # Handler dispatch table
    _DIAGNOSIS_HANDLERS = {
        FailureType.RETRIEVAL_FAILURE: _diagnose_retrieval,
        FailureType.GROUNDING_FAILURE: _diagnose_grounding,
        FailureType.GENERATION_FAILURE: _diagnose_generation,
        FailureType.CITATION_FAILURE: _diagnose_citation,
        FailureType.LATENCY_FAILURE: _diagnose_latency,
        FailureType.CONFIG_FAILURE: _diagnose_config,
    }


# Module-level singleton
diagnosis_engine = DiagnosisEngine()
