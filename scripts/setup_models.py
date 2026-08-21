"""Model Pre-Caching and Setup Script.

Pre-downloads and verifies all required ML models:
1. Embedding: sentence-transformers/all-MiniLM-L6-v2 (~90MB)
2. Reranker: cross-encoder/ms-marco-MiniLM-L6-v2 (~90MB)
3. Guardrail: cross-encoder/nli-deberta-v3-xsmall (~100MB)
"""

import sys
from pathlib import Path
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from src.config import settings
from src.models.manager import model_manager

console = Console()


def setup_all_models():
    console.print("[bold cyan]════════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]       📦 SELF-HEALING RAG ENGINE — MODEL SETUP             [/bold cyan]")
    console.print("[bold cyan]════════════════════════════════════════════════════════════[/bold cyan]\n")

    models = [
        ("Embedding Model", settings.embedding_model),
        ("Reranker Model", settings.reranker_model),
        ("Faithfulness Guardrail", settings.hhem_model),
    ]

    for label, model_name in models:
        console.print(f"[yellow]Downloading / verifying {label}:[/yellow] {model_name}")
        try:
            m = model_manager.get(model_name)
            console.print(f"[green]✓ {label} is ready and cached on disk.[/green]\n")
        except Exception as e:
            console.print(f"[red]✗ Failed to load {label}: {e}[/red]\n")

    console.print("[bold green]✅ All models are verified and cached for offline use![/bold green]")


if __name__ == "__main__":
    setup_all_models()
