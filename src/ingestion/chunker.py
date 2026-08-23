"""Semantic chunking — splits documents into retrieval-friendly chunks."""

from dataclasses import dataclass, field
from rich.console import Console

console = Console()


@dataclass
class Chunk:
    """A text chunk with metadata and a unique ID."""
    chunk_id: str
    content: str
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.content[:80].replace("\n", " ")
        return f"Chunk(id={self.chunk_id!r}, preview={preview!r}...)"


def chunk_documents(
    documents: list,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Chunk]:
    """Split documents into overlapping chunks for retrieval.

    Uses a recursive character splitting strategy that tries to split on
    natural boundaries (paragraphs > sentences > words) before falling
    back to character-level splits.
    """
    separators = ["\n\n", "\n", ". ", ", ", " ", ""]
    chunks: list[Chunk] = []
    chunk_counter = 0

    for doc in documents:
        text = doc.content
        doc_chunks = _recursive_split(text, chunk_size, chunk_overlap, separators)

        for i, chunk_text in enumerate(doc_chunks):
            chunk_counter += 1
            chunks.append(Chunk(
                chunk_id=f"chunk_{chunk_counter:04d}",
                content=chunk_text,
                metadata={
                    **doc.metadata,
                    "chunk_index": i,
                    "chunk_total": len(doc_chunks),
                }
            ))

    console.print(f"[green][OK] Created {len(chunks)} chunks from {len(documents)} documents[/green]")
    return chunks


def _recursive_split(
    text: str, chunk_size: int, chunk_overlap: int, separators: list[str],
) -> list[str]:
    """Recursively split text using the best available separator."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    separator = ""
    for sep in separators:
        if sep in text:
            separator = sep
            break

    if separator:
        parts = text.split(separator)
    else:
        parts = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - chunk_overlap)]
        return [p for p in parts if p.strip()]

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_length = 0

    for part in parts:
        part_length = len(part) + len(separator)

        if current_length + part_length > chunk_size and current_chunk:
            chunk_text = separator.join(current_chunk).strip()
            if chunk_text:
                chunks.append(chunk_text)

            overlap_parts: list[str] = []
            overlap_length = 0
            for prev_part in reversed(current_chunk):
                if overlap_length + len(prev_part) > chunk_overlap:
                    break
                overlap_parts.insert(0, prev_part)
                overlap_length += len(prev_part)

            current_chunk = overlap_parts
            current_length = overlap_length

        current_chunk.append(part)
        current_length += part_length

    if current_chunk:
        chunk_text = separator.join(current_chunk).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks
