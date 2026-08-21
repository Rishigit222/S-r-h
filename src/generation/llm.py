"""Multi-provider LLM abstraction.

Supports:
- Groq (free cloud API, fast)
- Google AI Studio (free cloud API)
- Ollama (local/free)
- Smart Local Extractive Generator (zero-setup fallback when offline & no API key)
"""

import re
from openai import OpenAI
from rich.console import Console

from src.config import settings

console = Console()


def _get_client() -> tuple[OpenAI | None, str, str]:
    """Get the appropriate OpenAI-compatible client based on config."""
    provider = settings.get_effective_provider()

    if provider == "groq" and settings.groq_api_key:
        client = OpenAI(
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        return client, settings.groq_model, "groq"

    elif provider == "google" and settings.google_api_key:
        client = OpenAI(
            api_key=settings.google_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        return client, settings.google_model, "google"

    elif settings.groq_api_key:
        client = OpenAI(
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        return client, settings.groq_model, "groq"

    elif settings.google_api_key:
        client = OpenAI(
            api_key=settings.google_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        return client, settings.google_model, "google"

    else:
        # Try local Ollama if configured
        try:
            client = OpenAI(
                api_key="ollama",
                base_url=f"{settings.ollama_base_url}/v1",
                timeout=3.0,
            )
            return client, settings.ollama_model, "ollama"
        except Exception:
            return None, "fallback-extractor", "local_fallback"


def _extractive_fallback_generate(prompt: str) -> str:
    """Smart local fallback when no external LLM API or Ollama is active.

    Extracts key relevant information directly from context chunks with citations.
    """
    if "CONTEXT CHUNKS:" not in prompt:
        return "I don't have enough information to answer this question."

    # Parse context chunks from prompt
    chunks_part = prompt.split("QUESTION:")[0]
    chunks = re.findall(r'\[(\d+)\]\s*\([^)]*\)\s*\n(.*?)(?=\n\n\[\d+\]|\n\nQUESTION:|$)', chunks_part, re.DOTALL)

    if not chunks:
        return "I don't have enough information to answer this question."

    sentences_with_citations = []
    for idx, text in chunks:
        clean_text = text.strip()
        lines = [line.strip() for line in clean_text.split("\n") if line.strip() and not line.startswith("#")]
        if lines:
            first_few = " ".join(lines[:3])
            sentences_with_citations.append(f"{first_few} [{idx}]")

    if not sentences_with_citations:
        return "I don't have enough information to answer this question."

    answer = "\n\n".join(sentences_with_citations[:2])
    return answer


def generate(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> str:
    """Generate a response from the LLM or fallback engine."""
    client, model, provider = _get_client()

    console.print(f"[cyan]🤖 Generating via {provider} ({model})...[/cyan]")

    if client is not None:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = response.choices[0].message.content or ""
            if text.strip():
                console.print(f"[green]✓ Generated {len(text)} chars via {provider}[/green]")
                return text
        except Exception as e:
            console.print(f"[yellow]⚠ {provider} unavailable ({e}), using smart local fallback[/yellow]")

    # Fallback to local extractive generator
    fallback_text = _extractive_fallback_generate(prompt)
    console.print(f"[green]✓ Generated {len(fallback_text)} chars via local_fallback[/green]")
    return fallback_text
