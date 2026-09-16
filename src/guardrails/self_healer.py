"""Self-healing query rewriter and refusal builder.

Provides the core LLM-based query rewrite capability used by
the engine's RewriteQueryStrategy, plus graceful refusal construction.

When guardrails detect a problem, this module:
1. Rewrites the query for better retrieval
2. Signals to re-retrieve and re-generate
3. Max N retries, then returns a graceful refusal
"""

from rich.console import Console

from src.config import settings
from src.generation.llm import generate
from src.generation.prompts import QUERY_REWRITE_PROMPT

console = Console()


def rewrite_query(original_query: str) -> str:
    """Use the LLM to rewrite a query for better retrieval.

    This is the underlying implementation used by
    src.engine.repair_strategies.RewriteQueryStrategy.
    """
    try:
        prompt = QUERY_REWRITE_PROMPT.format(query=original_query)
        rewritten = generate(
            prompt=prompt,
            system_prompt="You are a search query optimizer. Return only the rewritten query.",
            temperature=0.3,
            max_tokens=128,
        )
        rewritten = rewritten.strip().strip('"').strip("'")
        console.print(f"[cyan]🔄 Query rewritten: '{original_query}' → '{rewritten}'[/cyan]")
        return rewritten
    except Exception:
        return original_query


def should_heal(heal_attempt: int) -> bool:
    """Check if we should attempt another healing cycle."""
    return heal_attempt < settings.max_heal_retries


def build_refusal_response(original_query: str, reason: str) -> str:
    """Build a graceful refusal message when healing fails."""
    return (
        f'I\'m not confident in my answer to: "{original_query}"\n\n'
        f"Reason: {reason}\n\n"
        "The retrieved information either didn't match your question well enough, "
        "or the generated answer couldn't be verified against the sources. "
        "Try rephrasing your question or adding more documents to the knowledge base."
    )
