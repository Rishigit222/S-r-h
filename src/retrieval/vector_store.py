"""ChromaDB vector store wrapper with resilient collection handling."""

import chromadb
from chromadb.config import Settings as ChromaSettings
from rich.console import Console

from src.config import settings
from src.ingestion.embedder import embed_texts, embed_query

console = Console()


class VectorStore:
    """ChromaDB-backed vector store with persistent storage."""

    def __init__(self, collection_name: str = "rag_documents"):
        self._client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection_name = collection_name
        self._ensure_collection()

    def _ensure_collection(self):
        try:
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception:
            self._collection = self._client.get_collection(name=self._collection_name)

    @property
    def collection(self):
        try:
            # Check if alive
            _ = self._collection.count()
            return self._collection
        except Exception:
            self._ensure_collection()
            return self._collection

    def add_chunks(self, chunks: list) -> None:
        """Add chunks to the vector store with embeddings."""
        if not chunks:
            return

        texts = [c.content for c in chunks]
        ids = [c.chunk_id for c in chunks]
        metadatas = [c.metadata for c in chunks]

        embeddings = embed_texts(texts)

        batch_size = 5000
        for i in range(0, len(chunks), batch_size):
            end = min(i + batch_size, len(chunks))
            self.collection.add(
                ids=ids[i:end],
                embeddings=embeddings[i:end].tolist(),
                documents=texts[i:end],
                metadatas=metadatas[i:end],
            )

        console.print(f"[green]✓ Added {len(chunks)} chunks to vector store[/green]")

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """Search for similar chunks by cosine similarity."""
        top_k = top_k or settings.top_k_retrieval
        query_embedding = embed_query(query)

        count = self.count()
        if count == 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )

        if not results["ids"][0]:
            return []

        return [
            {
                "chunk_id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1.0 - results["distances"][0][i],
            }
            for i in range(len(results["ids"][0]))
        ]

    def count(self) -> int:
        try:
            return self.collection.count()
        except Exception:
            self._ensure_collection()
            return self._collection.count()

    def clear(self) -> None:
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:
            pass
        self._ensure_collection()
        console.print("[yellow]♻ Vector store cleared[/yellow]")
