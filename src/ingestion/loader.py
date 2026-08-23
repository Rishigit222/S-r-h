"""Document loader — reads PDF, Markdown, and plain text files."""

from pathlib import Path
from dataclasses import dataclass, field

from rich.console import Console

console = Console()


@dataclass
class Document:
    """A loaded document with content and metadata."""
    content: str
    metadata: dict = field(default_factory=dict)

    @property
    def source(self) -> str:
        return self.metadata.get("source", "unknown")


def load_pdf(file_path: Path) -> list[Document]:
    """Load a PDF file, returning one Document per page."""
    try:
        from pypdf import PdfReader
    except ImportError:
        console.print("[red]pypdf not installed. Run: pip install pypdf[/red]")
        return []

    reader = PdfReader(str(file_path))
    documents = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            documents.append(Document(
                content=text.strip(),
                metadata={"source": str(file_path.name), "page": i + 1, "type": "pdf"}
            ))
    return documents


def load_text(file_path: Path) -> list[Document]:
    """Load a plain text or markdown file as a single Document."""
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        return []

    file_type = "markdown" if file_path.suffix.lower() in (".md", ".mdx") else "text"
    return [Document(
        content=text.strip(),
        metadata={"source": str(file_path.name), "type": file_type}
    )]


def load_directory(directory: str | Path) -> list[Document]:
    """Load all supported documents from a directory.

    Supports: .pdf, .txt, .md, .mdx, .rst, .py
    """
    directory = Path(directory)
    if not directory.exists():
        console.print(f"[red]Directory not found: {directory}[/red]")
        return []

    supported_extensions = {".pdf", ".txt", ".md", ".mdx", ".rst", ".py"}
    documents: list[Document] = []

    for file_path in sorted(directory.rglob("*")):
        if file_path.suffix.lower() not in supported_extensions:
            continue
        if file_path.is_dir():
            continue

        try:
            if file_path.suffix.lower() == ".pdf":
                docs = load_pdf(file_path)
            else:
                docs = load_text(file_path)
            documents.extend(docs)
            console.print(f"  [dim]Loaded {file_path.name} ({len(docs)} doc(s))[/dim]")
        except Exception as e:
            console.print(f"  [red]Failed to load {file_path.name}: {e}[/red]")

    console.print(f"[green][OK] Loaded {len(documents)} documents from {directory}[/green]")
    return documents
