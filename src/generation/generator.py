"""RAG Generator — orchestrates retrieval + LLM + structured output."""

from dataclasses import dataclass, field
from rich.console import Console

from src.generation.llm import generate
from src.generation.prompts import RAG_SYSTEM_PROMPT, build_rag_prompt

console = Console()


@dataclass
class RAGResponse:
    """Structured response from the RAG pipeline."""
    answer: str
    sources: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    faithfulness_score: float = 0.0
    hallucination_risk: float = 0.0
    citations_valid: bool = False
    guardrail_passed: bool = False
    heal_attempts: int = 0
    provider: str = ""
    error: str | None = None


def generate_answer(query: str, context_chunks: list[dict]) -> RAGResponse:
    """Generate an answer from the LLM using retrieved context."""
    if not context_chunks:
        return RAGResponse(
            answer="I don't have enough information to answer this question.",
            guardrail_passed=True,
            confidence=1.0,
        )

    prompt = build_rag_prompt(query, context_chunks)

    try:
        answer = generate(
            prompt=prompt,
            system_prompt=RAG_SYSTEM_PROMPT,
            temperature=0.1,
        )
        return RAGResponse(answer=answer, sources=context_chunks)

    except Exception as e:
        return RAGResponse(
            answer="Sorry, I encountered an error generating the answer.",
            error=str(e),
        )
