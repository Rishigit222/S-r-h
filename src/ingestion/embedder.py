"""Embedding generator — converts text chunks to vector embeddings.

Optimized for CPU: uses all-MiniLM-L6-v2 (22M params, ~90MB RAM)
with batch processing for speed.
"""

import time
import logging
import numpy as np
from rich.console import Console

from src.config import settings
from src.models.manager import model_manager

console = Console()
logger = logging.getLogger("src.ingestion.embedder")


def embed_texts(texts: list[str]) -> np.ndarray:
    """Generate embeddings for a list of texts using batched encoding."""
    if not texts:
        return np.empty((0, 384))

    logger.info(
        "Embedding generation started: %d texts (batch_size=%d, model=%s)",
        len(texts),
        settings.embedding_batch_size,
        settings.embedding_model,
    )
    console.print(f"[cyan]Embedding {len(texts)} texts (batch_size={settings.embedding_batch_size})...[/cyan]")
    start = time.time()

    try:
        model = model_manager.get(settings.embedding_model)
        embeddings = model.encode(
            texts,
            batch_size=settings.embedding_batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,
        )
        elapsed = time.time() - start
        emb_array = np.array(embeddings)
        dim = emb_array.shape[1] if len(emb_array.shape) > 1 else 0
        logger.info(
            "Embedding generation completed: %d embeddings (dim=%d) in %.2fs",
            len(embeddings),
            dim,
            elapsed,
        )
        console.print(f"[green][OK] Generated {len(embeddings)} embeddings (dim={dim}) in {elapsed:.1f}s[/green]")
        return emb_array
    except Exception as e:
        elapsed = time.time() - start
        logger.error(
            "Embedding generation failed after %.2fs: %s",
            elapsed,
            e,
            exc_info=True,
        )
        console.print(f"[red][ERROR] Embedding generation failed after {elapsed:.1f}s: {e}[/red]")
        raise


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string."""
    start = time.time()
    try:
        model = model_manager.get(settings.embedding_model)
        emb = np.array(model.encode(query, normalize_embeddings=True))
        return emb
    except Exception as e:
        elapsed = time.time() - start
        logger.error("Query embedding failed after %.2fs: %s", elapsed, e, exc_info=True)
        raise
