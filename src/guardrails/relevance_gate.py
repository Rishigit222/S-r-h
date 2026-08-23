"""Pre-generation relevance gate.

Refuses to answer if the retrieved context is not relevant enough,
preventing the LLM from hallucinating when it has no good information.
"""

from dataclasses import dataclass
from rich.console import Console

from src.config import settings

console = Console()


@dataclass
class RelevanceResult:
    passed: bool
    best_score: float
    threshold: float
    reason: str


def check_relevance(reranked_results: list[dict], threshold: float | None = None) -> RelevanceResult:
    """Check if the top retrieved results are relevant enough to answer."""
    effective_threshold = threshold if threshold is not None else settings.relevance_threshold

    if not reranked_results:
        return RelevanceResult(
            passed=False, best_score=0.0,
            threshold=effective_threshold,
            reason="No results retrieved. Cannot answer.",
        )

    best_score = reranked_results[0].get("rerank_score", 0.0)

    if best_score < effective_threshold:
        console.print(
            f"[yellow]Relevance gate BLOCKED: best_score={best_score:.3f} "
            f"< threshold={effective_threshold}[/yellow]"
        )
        return RelevanceResult(
            passed=False, best_score=best_score, threshold=effective_threshold,
            reason=f"Retrieved context is not relevant enough (score: {best_score:.3f} < {effective_threshold}).",
        )

    console.print(f"[green][OK] Relevance gate passed: {best_score:.3f} >= {effective_threshold}[/green]")
    return RelevanceResult(
        passed=True, best_score=best_score, threshold=effective_threshold,
        reason="Context is relevant.",
    )
