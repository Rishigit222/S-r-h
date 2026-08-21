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


def check_relevance(reranked_results: list[dict]) -> RelevanceResult:
    """Check if the top retrieved results are relevant enough to answer."""
    if not reranked_results:
        return RelevanceResult(
            passed=False, best_score=0.0,
            threshold=settings.relevance_threshold,
            reason="No results retrieved. Cannot answer.",
        )

    best_score = reranked_results[0].get("rerank_score", 0.0)
    threshold = settings.relevance_threshold

    if best_score < threshold:
        console.print(
            f"[yellow]⚠ Relevance gate BLOCKED: best_score={best_score:.3f} "
            f"< threshold={threshold}[/yellow]"
        )
        return RelevanceResult(
            passed=False, best_score=best_score, threshold=threshold,
            reason=f"Retrieved context is not relevant enough (score: {best_score:.3f} < {threshold}).",
        )

    console.print(f"[green]✓ Relevance gate passed: {best_score:.3f} >= {threshold}[/green]")
    return RelevanceResult(
        passed=True, best_score=best_score, threshold=threshold,
        reason="Context is relevant.",
    )
