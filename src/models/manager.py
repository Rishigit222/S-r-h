"""Lazy model loader with LRU eviction for CPU-constrained environments.

Loads ML models (embeddings, reranker, HHEM) on-demand and caches them.
When RAM is tight, evicts least-recently-used models before loading new ones.
This prevents OOM on machines with 8-16GB RAM.
"""

import time
import gc
import logging
import threading
from typing import Any
from collections import OrderedDict

from rich.console import Console

console = Console()
logger = logging.getLogger("src.models.manager")


class ModelManager:
    """Manages ML model lifecycle with lazy loading, background warmup, and LRU eviction."""

    def __init__(self, max_models: int = 2):
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._max_models = max_models
        self._load_times: dict[str, float] = {}
        self._lock = threading.Lock()
        self._model_locks: dict[str, threading.Lock] = {}
        self._loading_models: set[str] = set()
        self._load_errors: dict[str, Exception] = {}

    def is_loading(self, model_name: str) -> bool:
        """Check if a model is currently being loaded in a background thread."""
        with self._lock:
            return model_name in self._loading_models

    def warmup(self, model_name: str) -> None:
        """Trigger background model warmup safely."""
        logger.info("Background model warmup started: %s", model_name)
        start = time.time()
        try:
            self.get(model_name)
            elapsed = time.time() - start
            logger.info("Background model warmup completed: %s in %.2fs", model_name, elapsed)
        except Exception as e:
            elapsed = time.time() - start
            logger.error("Background model warmup failed for %s after %.2fs: %s", model_name, elapsed, e, exc_info=True)

    def get(self, model_name: str, timeout: float = 60.0) -> Any:
        """Get a model, loading it if not cached. Thread-safe with single-flight loading."""
        # 1. Fast path: check cache under master lock
        with self._lock:
            if model_name in self._cache:
                self._cache.move_to_end(model_name)
                return self._cache[model_name]

            if model_name in self._load_errors:
                err = self._load_errors[model_name]
                logger.error("Model '%s' previously failed to load: %s", model_name, err)
                raise RuntimeError(f"Model '{model_name}' failed to load: {err}") from err

            if model_name not in self._model_locks:
                self._model_locks[model_name] = threading.Lock()
            model_lock = self._model_locks[model_name]
            already_loading = model_name in self._loading_models

        if already_loading:
            logger.info("Request waiting for model '%s' initialization/warmup to complete...", model_name)

        # 2. Acquire per-model lock so only one thread loads the model
        acquired = model_lock.acquire(timeout=timeout)
        if not acquired:
            logger.error("Timed out after %.1fs waiting for model '%s' initialization/warmup", timeout, model_name)
            raise TimeoutError(f"Timed out after {timeout}s waiting for model '{model_name}' initialization")

        try:
            with self._lock:
                # Double-check cache in case the loading thread just finished
                if model_name in self._cache:
                    self._cache.move_to_end(model_name)
                    return self._cache[model_name]

                if model_name in self._load_errors:
                    err = self._load_errors[model_name]
                    logger.error("Model '%s' failed to load: %s", model_name, err)
                    raise RuntimeError(f"Model '{model_name}' failed to load: {err}") from err

                self._loading_models.add(model_name)

            try:
                with self._lock:
                    self._evict_if_needed()
                model = self._load_model(model_name)
                with self._lock:
                    self._cache[model_name] = model
                    self._loading_models.discard(model_name)
                    self._load_errors.pop(model_name, None)
                return model
            except Exception as e:
                with self._lock:
                    self._loading_models.discard(model_name)
                    self._load_errors[model_name] = e
                raise
        finally:
            model_lock.release()

    def _evict_if_needed(self) -> None:
        """Evict the least-recently-used model if cache is full (called with master lock held)."""
        while len(self._cache) >= self._max_models:
            evicted_name, evicted_model = self._cache.popitem(last=False)
            logger.info("Evicting model from cache: %s", evicted_name)
            console.print(f"[yellow]Evicting model: {evicted_name}[/yellow]")
            del evicted_model
            gc.collect()


    def _load_model(self, model_name: str) -> Any:
        """Load a model by name, routing to the appropriate loader."""
        logger.info("Model initialization started: %s", model_name)
        console.print(f"[cyan]Loading model: {model_name}...[/cyan]")
        start = time.time()

        try:
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
            logger.info("Model initialization completed: %s in %.2fs", model_name, elapsed)
            console.print(f"[green][OK] Loaded {model_name} in {elapsed:.1f}s[/green]")
            return model
        except Exception as e:
            elapsed = time.time() - start
            logger.error("Model initialization failed for %s after %.2fs: %s", model_name, elapsed, e, exc_info=True)
            console.print(f"[red][ERROR] Failed to load model {model_name} after {elapsed:.1f}s: {e}[/red]")
            raise

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
