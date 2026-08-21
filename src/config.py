"""Central configuration for the Self-Healing RAG Engine."""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # --- LLM Provider ---
    llm_provider: str = Field(default="ollama", description="ollama | groq | google")

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "phi4-mini"

    # Groq
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # Google AI Studio
    google_api_key: str = ""
    google_model: str = "gemini-2.5-flash"

    # --- Embedding ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 64

    # --- Retrieval ---
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_retrieval: int = 20
    top_k_rerank: int = 5
    rrf_k: int = 60
    vector_weight: float = 0.6
    bm25_weight: float = 0.4

    # --- Reranker ---
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"

    # --- Guardrails ---
    relevance_threshold: float = 0.3
    faithfulness_threshold: float = 0.5
    hhem_model: str = "cross-encoder/nli-deberta-v3-xsmall"
    max_heal_retries: int = 2

    # --- Paths ---
    chroma_persist_dir: str = "./data/chroma_db"
    documents_dir: str = "./data/sample_docs"
    project_root: str = str(Path(__file__).parent.parent)

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    def get_effective_provider(self) -> str:
        """Auto-detect the best available LLM provider."""
        if self.llm_provider != "ollama":
            return self.llm_provider
        if self.groq_api_key:
            return "groq"
        if self.google_api_key:
            return "google"
        return "ollama"


# Singleton instance
settings = Settings()
