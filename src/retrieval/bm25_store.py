"""BM25 keyword search index.

Complements vector search by catching exact keyword matches
that semantic search might miss (e.g., code symbols, acronyms).
"""

import re
from rank_bm25 import BM25Okapi
from rich.console import Console

console = Console()


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r'\b\w+\b', text.lower())


class BM25Store:
    """BM25-based keyword search index."""

    def __init__(self):
        self._chunks: list[dict] = []
        self._tokenized_corpus: list[list[str]] = []
        self._index: BM25Okapi | None = None

    def add_chunks(self, chunks: list) -> None:
        for chunk in chunks:
            self._chunks.append({
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "metadata": chunk.metadata,
            })
            self._tokenized_corpus.append(_tokenize(chunk.content))

        self._index = BM25Okapi(self._tokenized_corpus)
        console.print(f"[green]✓ BM25 index built with {len(self._chunks)} chunks[/green]")

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        if self._index is None or not self._chunks:
            return []

        scores = self._index.get_scores(_tokenize(query))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        return [
            {
                "chunk_id": self._chunks[idx]["chunk_id"],
                "content": self._chunks[idx]["content"],
                "metadata": self._chunks[idx]["metadata"],
                "score": float(scores[idx]),
            }
            for idx in top_indices if scores[idx] > 0
        ]

    def count(self) -> int:
        return len(self._chunks)

    def clear(self) -> None:
        self._chunks.clear()
        self._tokenized_corpus.clear()
        self._index = None
