"""Repair Planner — Selects and parameterizes repair strategies based on diagnosis.

Takes a Diagnosis from the DiagnosisEngine and produces a concrete RepairPlan
with parameters, rollback snapshot, and estimated impact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.engine.diagnosis_engine import Diagnosis, RepairStrategyType


@dataclass
class RepairPlan:
    """A concrete plan for repairing a detected RAG failure."""

    strategy: str
    """The RepairStrategyType identifier for this repair."""

    parameters: dict[str, Any]
    """Strategy-specific parameters for executing the repair."""

    rollback_snapshot: dict[str, Any]
    """Snapshot of current configuration state for rollback."""

    estimated_impact: str
    """Human-readable estimate of the expected improvement."""

    diagnosis: Diagnosis
    """The diagnosis that led to this repair plan."""

    attempt_index: int = 0
    """Which repair strategy from the diagnosis list this represents (0 = first)."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "strategy": self.strategy,
            "parameters": self.parameters,
            "estimated_impact": self.estimated_impact,
            "attempt_index": self.attempt_index,
            "diagnosis_root_cause": self.diagnosis.root_cause,
        }


class RepairPlanner:
    """Plans concrete repair actions from a diagnosis.

    For each diagnosis, produces a RepairPlan with:
    - The repair strategy to execute
    - Strategy-specific parameters
    - A rollback snapshot of the current config
    - An estimated impact description
    """

    def plan(
        self,
        diagnosis: Diagnosis,
        current_config: dict[str, Any],
        attempt_index: int = 0,
    ) -> RepairPlan | None:
        """Create a repair plan from a diagnosis.

        Args:
            diagnosis: The diagnosis result from the DiagnosisEngine.
            current_config: Current RAG pipeline configuration (for rollback snapshot).
            attempt_index: Which strategy from the diagnosis to try (0 = first).

        Returns:
            A RepairPlan, or None if all strategies have been exhausted.
        """
        if attempt_index >= len(diagnosis.suggested_repairs):
            return None

        strategy = diagnosis.suggested_repairs[attempt_index]
        params = self._build_strategy_params(strategy, diagnosis, current_config)
        impact = self._estimate_impact(strategy, diagnosis)

        return RepairPlan(
            strategy=strategy,
            parameters=params,
            rollback_snapshot=dict(current_config),
            estimated_impact=impact,
            diagnosis=diagnosis,
            attempt_index=attempt_index,
        )

    def plan_next(self, previous_plan: RepairPlan, current_config: dict[str, Any]) -> RepairPlan | None:
        """Plan the next repair strategy after a failed attempt.

        Args:
            previous_plan: The plan that was tried and failed.
            current_config: Current configuration state.

        Returns:
            The next RepairPlan, or None if exhausted.
        """
        return self.plan(
            previous_plan.diagnosis,
            current_config,
            attempt_index=previous_plan.attempt_index + 1,
        )

    def _build_strategy_params(
        self, strategy: str, diagnosis: Diagnosis, config: dict[str, Any]
    ) -> dict[str, Any]:
        """Build strategy-specific execution parameters."""
        builders = {
            RepairStrategyType.REWRITE_QUERY: self._params_rewrite_query,
            RepairStrategyType.RECHUNK_DOCUMENTS: self._params_rechunk,
            RepairStrategyType.ADJUST_RETRIEVAL: self._params_adjust_retrieval,
            RepairStrategyType.EXPAND_CONTEXT: self._params_expand_context,
            RepairStrategyType.ADJUST_THRESHOLDS: self._params_adjust_thresholds,
            RepairStrategyType.SWAP_RERANKER: self._params_swap_reranker,
            RepairStrategyType.REGENERATE: self._params_regenerate,
        }
        builder = builders.get(strategy, lambda d, c: {})
        return builder(diagnosis, config)

    def _params_rewrite_query(self, diagnosis: Diagnosis, config: dict[str, Any]) -> dict[str, Any]:
        return {
            "original_query": diagnosis.failure_report.query,
            "rewrite_temperature": 0.3,
            "max_tokens": 128,
        }

    def _params_rechunk(self, diagnosis: Diagnosis, config: dict[str, Any]) -> dict[str, Any]:
        current_chunk_size = config.get("chunk_size", 512)
        current_overlap = config.get("chunk_overlap", 64)
        # If retrieval is poor, try smaller chunks for finer granularity
        return {
            "new_chunk_size": max(128, current_chunk_size - 128),
            "new_chunk_overlap": max(32, current_overlap),
            "original_chunk_size": current_chunk_size,
            "original_chunk_overlap": current_overlap,
        }

    def _params_adjust_retrieval(self, diagnosis: Diagnosis, config: dict[str, Any]) -> dict[str, Any]:
        current_top_k = config.get("top_k_retrieval", 20)
        current_bm25_w = config.get("bm25_weight", 0.4)
        current_vector_w = config.get("vector_weight", 0.6)
        # Increase retrieval breadth and adjust fusion weights
        return {
            "new_top_k": min(40, current_top_k + 10),
            "new_bm25_weight": round(min(0.7, current_bm25_w + 0.1), 2),
            "new_vector_weight": round(max(0.3, current_vector_w - 0.1), 2),
            "original_top_k": current_top_k,
            "original_bm25_weight": current_bm25_w,
            "original_vector_weight": current_vector_w,
        }

    def _params_expand_context(self, diagnosis: Diagnosis, config: dict[str, Any]) -> dict[str, Any]:
        return {
            "enable_web_fallback": True,
            "increase_top_k_rerank": min(10, config.get("top_k_rerank", 5) + 3),
            "original_top_k_rerank": config.get("top_k_rerank", 5),
        }

    def _params_adjust_thresholds(self, diagnosis: Diagnosis, config: dict[str, Any]) -> dict[str, Any]:
        current_rel_th = config.get("relevance_threshold", 0.3)
        current_faith_th = config.get("faithfulness_threshold", 0.5)
        failure_type = diagnosis.failure_report.failure_type.value

        # If retrieval failure with borderline scores, lower relevance threshold
        if "retrieval" in failure_type:
            return {
                "new_relevance_threshold": round(max(0.15, current_rel_th - 0.05), 2),
                "original_relevance_threshold": current_rel_th,
                "faithfulness_threshold": current_faith_th,
            }
        # If config drift with high hallucination, raise thresholds
        return {
            "new_relevance_threshold": round(min(0.5, current_rel_th + 0.05), 2),
            "new_faithfulness_threshold": round(min(0.8, current_faith_th + 0.1), 2),
            "original_relevance_threshold": current_rel_th,
            "original_faithfulness_threshold": current_faith_th,
        }

    def _params_swap_reranker(self, diagnosis: Diagnosis, config: dict[str, Any]) -> dict[str, Any]:
        return {
            "new_top_k_rerank": min(10, config.get("top_k_rerank", 5) + 2),
            "original_top_k_rerank": config.get("top_k_rerank", 5),
        }

    def _params_regenerate(self, diagnosis: Diagnosis, config: dict[str, Any]) -> dict[str, Any]:
        return {
            "query": diagnosis.failure_report.query,
            "temperature": 0.2,
            "max_retries": 1,
            "use_stricter_prompt": True,
        }

    def _estimate_impact(self, strategy: str, diagnosis: Diagnosis) -> str:
        """Produce a human-readable impact estimate."""
        estimates = {
            RepairStrategyType.REWRITE_QUERY: "Query reformulation typically resolves 40-60% of retrieval failures by bridging vocabulary gaps.",
            RepairStrategyType.RECHUNK_DOCUMENTS: "Re-chunking with smaller sizes can improve retrieval precision by 15-25% for specific queries.",
            RepairStrategyType.ADJUST_RETRIEVAL: "Adjusting retrieval weights can shift recall/precision balance by 10-20%.",
            RepairStrategyType.EXPAND_CONTEXT: "Expanding context via web fallback provides coverage for out-of-corpus queries.",
            RepairStrategyType.ADJUST_THRESHOLDS: "Threshold tuning can reduce false refusals by 20-30% or false accepts by 10-15%.",
            RepairStrategyType.SWAP_RERANKER: "Increasing reranker top-k provides more candidates for better precision.",
            RepairStrategyType.REGENERATE: "Regeneration with stricter prompting resolves 30-50% of grounding and citation failures.",
        }
        return estimates.get(strategy, "Impact unknown for this strategy.")


# Module-level singleton
repair_planner = RepairPlanner()
