"""Lazy model loader with LRU eviction for CPU-constrained environments.

Loads ML models (embeddings, reranker, HHEM) on-demand and caches them.
When RAM is tight, evicts least-recently-used models before loading new ones.
This prevents OOM on machines with 8-16GB RAM.
"""

import time
import gc
from typing import Any
from collections import OrderedDict

from rich.console import Console

console = Console()


class ModelManager:
    """Manages ML model lifecycle with lazy loading and LRU eviction."""

    def __init__(self, max_models: int = 2):
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._max_models = max_models
        self._load_times: dict[str, float] = {}

    def get(self, model_name: str) -> Any:
        """Get a model, loading it if not cached."""
        if model_name in self._cache:
            self._cache.move_to_end(model_name)
            return self._cache[model_name]

        self._evict_if_needed()
        model = self._load_model(model_name)
        self._cache[model_name] = model
        return model

    def _evict_if_needed(self) -> None:
        """Evict the least-recently-used model if cache is full."""
        while len(self._cache) >= self._max_models:
            evicted_name, evicted_model = self._cache.popitem(last=False)
            console.print(f"[yellow]♻ Evicting model: {evicted_name}[/yellow]")
            del evicted_model
            gc.collect()

    def _load_model(self, model_name: str) -> Any:
        """Load a model by name, routing to the appropriate loader."""
        console.print(f"[cyan]📦 Loading model: {model_name}...[/cyan]")
        start = time.time()

        if "all-MiniLM" in model_name or "embedding" in model_name.lower():
            model = self._load_sentence_transformer(model_name)
        elif "cross-encoder" in model_name or "ms-marco" in model_name:
            model = self._load_cross_encoder(model_name)
        elif "hallucination" in model_name or "hhem" in model_name.lower():
            model = self._load_cross_encoder(model_name)
        else:
            model = self._load_sentence_transformer(model_name)

        elapsed = time.time() - start
        self._load_times[model_name] = elapsed
        console.print(f"[green]✓ Loaded {model_name} in {elapsed:.1f}s[/green]")
        return model

    def _load_sentence_transformer(self, model_name: str) -> Any:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name, device="cpu")

    def _load_cross_encoder(self, model_name: str) -> Any:
        """Load a CrossEncoder model for reranking or hallucination detection."""
        from sentence_transformers import CrossEncoder
        return CrossEncoder(model_name, trust_remote_code=True)

    def is_loaded(self, model_name: str) -> bool:
        return model_name in self._cache

    def unload_all(self) -> None:
        self._cache.clear()
        gc.collect()

    def status(self) -> dict:
        return {
            "loaded_models": list(self._cache.keys()),
            "capacity": f"{len(self._cache)}/{self._max_models}",
            "load_times": self._load_times,
        }


# Singleton instance
model_manager = ModelManager(max_models=2)
