"""Embedding generator — converts text chunks to vector embeddings.

Optimized for CPU: uses all-MiniLM-L6-v2 (22M params, ~90MB RAM)
with batch processing for speed.
"""

import numpy as np
from rich.console import Console

from src.config import settings
from src.models.manager import model_manager

console = Console()


def embed_texts(texts: list[str]) -> np.ndarray:
    """Generate embeddings for a list of texts using batched encoding."""
    model = model_manager.get(settings.embedding_model)

    console.print(f"[cyan]Embedding {len(texts)} texts (batch_size={settings.embedding_batch_size})...[/cyan]")

    embeddings = model.encode(
        texts,
        batch_size=settings.embedding_batch_size,
        show_progress_bar=len(texts) > 100,
        normalize_embeddings=True,
    )

    console.print(f"[green]✓ Generated {len(embeddings)} embeddings (dim={embeddings.shape[1]})[/green]")
    return np.array(embeddings)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string."""
    model = model_manager.get(settings.embedding_model)
    return np.array(model.encode(query, normalize_embeddings=True))
