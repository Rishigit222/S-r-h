"""Hybrid retrieval — fuses vector + BM25 results with Reciprocal Rank Fusion."""

from src.config import settings


def reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
    k: int | None = None,
    vector_weight: float | None = None,
    bm25_weight: float | None = None,
) -> list[dict]:
    """Merge vector and BM25 results using Reciprocal Rank Fusion (RRF).

    RRF score = sum(weight / (k + rank)) for each result list.
    """
    k = k or settings.rrf_k
    vector_weight = vector_weight or settings.vector_weight
    bm25_weight = bm25_weight or settings.bm25_weight

    fused: dict[str, dict] = {}

    for rank, result in enumerate(vector_results):
        chunk_id = result["chunk_id"]
        rrf_score = vector_weight / (k + rank + 1)
        if chunk_id not in fused:
            fused[chunk_id] = {
                "chunk_id": chunk_id,
                "content": result["content"],
                "metadata": result["metadata"],
                "rrf_score": 0.0,
                "vector_score": result.get("score", 0.0),
                "bm25_score": 0.0,
            }
        fused[chunk_id]["rrf_score"] += rrf_score

    for rank, result in enumerate(bm25_results):
        chunk_id = result["chunk_id"]
        rrf_score = bm25_weight / (k + rank + 1)
        if chunk_id not in fused:
            fused[chunk_id] = {
                "chunk_id": chunk_id,
                "content": result["content"],
                "metadata": result["metadata"],
                "rrf_score": 0.0,
                "vector_score": 0.0,
                "bm25_score": result.get("score", 0.0),
            }
        fused[chunk_id]["rrf_score"] += rrf_score
        fused[chunk_id]["bm25_score"] = result.get("score", 0.0)

    return sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)
