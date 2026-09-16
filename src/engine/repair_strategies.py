"""Repair Strategies — Concrete implementations of each repair action.

Each strategy implements a common interface:
  apply(plan, pipeline_config) -> RepairResult
  rollback(snapshot, pipeline_config) -> None
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from abc import ABC, abstractmethod

from rich.console import Console

from src.engine.repair_planner import RepairPlan

console = Console()


@dataclass
class RepairResult:
    """Outcome of applying a repair strategy."""

    success: bool
    """Whether the repair was applied successfully (not whether it improved quality)."""

    strategy: str
    """Which strategy was applied."""

    changes_made: dict[str, Any]
    """Description of what was changed."""

    new_config: dict[str, Any]
    """The pipeline configuration after repair."""

    rewritten_query: str | None = None
    """If the strategy rewrote the query, the new query text."""

    error: str | None = None
    """Error message if the repair failed to apply."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "strategy": self.strategy,
            "changes_made": self.changes_made,
            "rewritten_query": self.rewritten_query,
            "error": self.error,
        }


class BaseRepairStrategy(ABC):
    """Abstract base class for repair strategies."""

    @abstractmethod
    def apply(self, plan: RepairPlan, pipeline_config: dict[str, Any]) -> RepairResult:
        """Apply this repair strategy."""
        ...

    @abstractmethod
    def rollback(self, snapshot: dict[str, Any], pipeline_config: dict[str, Any]) -> None:
        """Rollback this repair by restoring the snapshot."""
        ...


class RewriteQueryStrategy(BaseRepairStrategy):
    """Rewrites the user query using LLM for better retrieval."""

    def apply(self, plan: RepairPlan, pipeline_config: dict[str, Any]) -> RepairResult:
        from src.guardrails.self_healer import rewrite_query

        original = plan.parameters.get("original_query", plan.diagnosis.failure_report.query)
        try:
            rewritten = rewrite_query(original)
            console.print(f"[cyan]🔄 REPAIR: Query rewritten: '{original}' → '{rewritten}'[/cyan]")
            return RepairResult(
                success=True,
                strategy="rewrite_query",
                changes_made={"original_query": original, "rewritten_query": rewritten},
                new_config=dict(pipeline_config),
                rewritten_query=rewritten,
            )
        except Exception as e:
            return RepairResult(
                success=False,
                strategy="rewrite_query",
                changes_made={},
                new_config=dict(pipeline_config),
                error=str(e),
            )

    def rollback(self, snapshot: dict[str, Any], pipeline_config: dict[str, Any]) -> None:
        pass  # Query rewrite doesn't modify persistent config


class RechunkDocumentsStrategy(BaseRepairStrategy):
    """Adjusts chunk_size and chunk_overlap parameters."""

    def apply(self, plan: RepairPlan, pipeline_config: dict[str, Any]) -> RepairResult:
        new_size = plan.parameters.get("new_chunk_size", 384)
        new_overlap = plan.parameters.get("new_chunk_overlap", 64)
        old_size = pipeline_config.get("chunk_size", 512)
        old_overlap = pipeline_config.get("chunk_overlap", 64)

        pipeline_config["chunk_size"] = new_size
        pipeline_config["chunk_overlap"] = new_overlap

        console.print(f"[yellow]🔧 REPAIR: Chunk size {old_size}→{new_size}, overlap {old_overlap}→{new_overlap}[/yellow]")
        return RepairResult(
            success=True,
            strategy="rechunk_documents",
            changes_made={
                "chunk_size": {"old": old_size, "new": new_size},
                "chunk_overlap": {"old": old_overlap, "new": new_overlap},
            },
            new_config=dict(pipeline_config),
        )

    def rollback(self, snapshot: dict[str, Any], pipeline_config: dict[str, Any]) -> None:
        pipeline_config["chunk_size"] = snapshot.get("chunk_size", 512)
        pipeline_config["chunk_overlap"] = snapshot.get("chunk_overlap", 64)


class AdjustRetrievalStrategy(BaseRepairStrategy):
    """Modifies retrieval weights and top_k parameters."""

    def apply(self, plan: RepairPlan, pipeline_config: dict[str, Any]) -> RepairResult:
        changes: dict[str, Any] = {}

        for key in ("top_k_retrieval", "bm25_weight", "vector_weight"):
            new_key = f"new_{key}"
            new_val = plan.parameters.get(new_key, plan.parameters.get(key))
            if new_val is not None:
                old_val = pipeline_config.get(key)
                pipeline_config[key] = new_val
                changes[key] = {"old": old_val, "new": new_val}

        console.print(f"[yellow]🔧 REPAIR: Retrieval params adjusted: {changes}[/yellow]")
        return RepairResult(
            success=True,
            strategy="adjust_retrieval",
            changes_made=changes,
            new_config=dict(pipeline_config),
        )

    def rollback(self, snapshot: dict[str, Any], pipeline_config: dict[str, Any]) -> None:
        for key in ("top_k_retrieval", "bm25_weight", "vector_weight"):
            if key in snapshot:
                pipeline_config[key] = snapshot[key]


class ExpandContextStrategy(BaseRepairStrategy):
    """Expands retrieval context via web fallback or increased top_k_rerank."""

    def apply(self, plan: RepairPlan, pipeline_config: dict[str, Any]) -> RepairResult:
        changes: dict[str, Any] = {}

        if plan.parameters.get("enable_web_fallback"):
            pipeline_config["enable_web_fallback"] = True
            changes["enable_web_fallback"] = True

        new_top_k = plan.parameters.get("increase_top_k_rerank")
        if new_top_k:
            old_val = pipeline_config.get("top_k_rerank", 5)
            pipeline_config["top_k_rerank"] = new_top_k
            changes["top_k_rerank"] = {"old": old_val, "new": new_top_k}

        console.print(f"[yellow]🔧 REPAIR: Context expanded: {changes}[/yellow]")
        return RepairResult(
            success=True,
            strategy="expand_context",
            changes_made=changes,
            new_config=dict(pipeline_config),
        )

    def rollback(self, snapshot: dict[str, Any], pipeline_config: dict[str, Any]) -> None:
        for key in ("enable_web_fallback", "top_k_rerank"):
            if key in snapshot:
                pipeline_config[key] = snapshot[key]


class AdjustThresholdsStrategy(BaseRepairStrategy):
    """Tunes relevance and faithfulness thresholds."""

    def apply(self, plan: RepairPlan, pipeline_config: dict[str, Any]) -> RepairResult:
        changes: dict[str, Any] = {}

        for key in ("relevance_threshold", "faithfulness_threshold"):
            new_key = f"new_{key}"
            if new_key in plan.parameters:
                old_val = pipeline_config.get(key)
                new_val = plan.parameters[new_key]
                pipeline_config[key] = new_val
                changes[key] = {"old": old_val, "new": new_val}

        console.print(f"[yellow]🔧 REPAIR: Thresholds adjusted: {changes}[/yellow]")
        return RepairResult(
            success=True,
            strategy="adjust_thresholds",
            changes_made=changes,
            new_config=dict(pipeline_config),
        )

    def rollback(self, snapshot: dict[str, Any], pipeline_config: dict[str, Any]) -> None:
        for key in ("relevance_threshold", "faithfulness_threshold"):
            if key in snapshot:
                pipeline_config[key] = snapshot[key]


class SwapRerankerStrategy(BaseRepairStrategy):
    """Adjusts reranker parameters (top_k_rerank)."""

    def apply(self, plan: RepairPlan, pipeline_config: dict[str, Any]) -> RepairResult:
        new_top_k = plan.parameters.get("new_top_k_rerank", 7)
        old_top_k = pipeline_config.get("top_k_rerank", 5)
        pipeline_config["top_k_rerank"] = new_top_k

        console.print(f"[yellow]🔧 REPAIR: Reranker top_k {old_top_k}→{new_top_k}[/yellow]")
        return RepairResult(
            success=True,
            strategy="swap_reranker",
            changes_made={"top_k_rerank": {"old": old_top_k, "new": new_top_k}},
            new_config=dict(pipeline_config),
        )

    def rollback(self, snapshot: dict[str, Any], pipeline_config: dict[str, Any]) -> None:
        pipeline_config["top_k_rerank"] = snapshot.get("top_k_rerank", 5)


class RegenerateStrategy(BaseRepairStrategy):
    """Retries LLM generation with stricter prompting."""

    def apply(self, plan: RepairPlan, pipeline_config: dict[str, Any]) -> RepairResult:
        # Regenerate modifies the generation approach, not persistent config
        pipeline_config["_regenerate_with_strict_prompt"] = True
        pipeline_config["_regenerate_temperature"] = plan.parameters.get("temperature", 0.2)

        console.print("[yellow]🔧 REPAIR: Regenerating with stricter prompt and lower temperature[/yellow]")
        return RepairResult(
            success=True,
            strategy="regenerate",
            changes_made={
                "strict_prompt": True,
                "temperature": plan.parameters.get("temperature", 0.2),
            },
            new_config=dict(pipeline_config),
        )

    def rollback(self, snapshot: dict[str, Any], pipeline_config: dict[str, Any]) -> None:
        pipeline_config.pop("_regenerate_with_strict_prompt", None)
        pipeline_config.pop("_regenerate_temperature", None)


# --- Strategy Registry ---

STRATEGY_REGISTRY: dict[str, BaseRepairStrategy] = {
    "rewrite_query": RewriteQueryStrategy(),
    "rechunk_documents": RechunkDocumentsStrategy(),
    "adjust_retrieval": AdjustRetrievalStrategy(),
    "expand_context": ExpandContextStrategy(),
    "adjust_thresholds": AdjustThresholdsStrategy(),
    "swap_reranker": SwapRerankerStrategy(),
    "regenerate": RegenerateStrategy(),
}


def execute_repair(plan: RepairPlan, pipeline_config: dict[str, Any]) -> RepairResult:
    """Execute a repair plan using the appropriate strategy.

    Args:
        plan: The repair plan to execute.
        pipeline_config: Mutable pipeline configuration dictionary.

    Returns:
        RepairResult describing what was changed.
    """
    strategy = STRATEGY_REGISTRY.get(plan.strategy)
    if strategy is None:
        return RepairResult(
            success=False,
            strategy=plan.strategy,
            changes_made={},
            new_config=dict(pipeline_config),
            error=f"Unknown repair strategy: {plan.strategy}",
        )
    return strategy.apply(plan, pipeline_config)


def rollback_repair(plan: RepairPlan, pipeline_config: dict[str, Any]) -> None:
    """Rollback a repair by restoring the snapshot from the plan.

    Args:
        plan: The repair plan whose rollback_snapshot should be restored.
        pipeline_config: Mutable pipeline configuration to restore.
    """
    strategy = STRATEGY_REGISTRY.get(plan.strategy)
    if strategy is not None:
        strategy.rollback(plan.rollback_snapshot, pipeline_config)
        console.print(f"[red]⏪ ROLLBACK: Reverted {plan.strategy} repair[/red]")
