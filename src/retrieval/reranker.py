"""Cross-encoder reranker — re-scores candidates for precision.

Uses ms-marco-MiniLM-L6-v2 (smallest, fastest cross-encoder).
On CPU: ~50-100ms per pair, so we limit to top 10-20 candidates.
"""

from rich.console import Console

from src.config import settings
from src.models.manager import model_manager

console = Console()


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int | None = None,
) -> list[dict]:
    """Rerank candidate results using a cross-encoder for precision."""
    top_k = top_k or settings.top_k_rerank

    if not candidates:
        return []

    model = model_manager.get(settings.reranker_model)
    pairs = [(query, c["content"]) for c in candidates]
    scores = model.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]

    console.print(
        f"[green]✓ Reranked {len(candidates)} → top {len(reranked)} "
        f"(best={reranked[0]['rerank_score']:.3f})[/green]"
    )
    return reranked
