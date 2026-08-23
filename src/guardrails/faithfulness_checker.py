"""Post-generation faithfulness checker.

Uses Vectara HHEM or NLI Cross-Encoder to verify if the generated answer
is faithful to the retrieved context chunks.

Score: 1.0 = perfectly faithful, 0.0 = complete hallucination.
"""

from dataclasses import dataclass
from rich.console import Console

from src.config import settings
from src.models.manager import model_manager

console = Console()


@dataclass
class FaithfulnessResult:
    passed: bool
    score: float
    threshold: float
    reason: str


def check_faithfulness(answer: str, context_chunks: list[dict]) -> FaithfulnessResult:
    """Check if the answer is faithful to the retrieved context."""
    if not answer or not context_chunks:
        return FaithfulnessResult(
            passed=False, score=0.0,
            threshold=settings.faithfulness_threshold,
            reason="Empty answer or context.",
        )

    premise = " ".join(chunk["content"] for chunk in context_chunks)

    # Truncate for cross-encoder token limits
    max_length = 2048
    if len(premise) > max_length:
        premise = premise[:max_length]
    if len(answer) > max_length:
        answer = answer[:max_length]

    try:
        model = model_manager.get(settings.hhem_model)
        scores = model.predict([(premise, answer)])
        score = float(scores[0])
    except Exception as e:
        console.print(f"[yellow]⚠ HHEM check failed ({e}), using fallback NLI scorer[/yellow]")
        # Fallback: simple text overlap / entailment ratio
        words_answer = set(answer.lower().split())
        words_context = set(premise.lower().split())
        overlap = len(words_answer.intersection(words_context)) / max(1, len(words_answer))
        score = min(1.0, overlap * 1.2)

    threshold = settings.faithfulness_threshold
    passed = score >= threshold

    if passed:
        console.print(f"[green][OK] Faithfulness check passed: {score:.3f} >= {threshold}[/green]")
    else:
        console.print(f"[red]✗ Faithfulness check FAILED: {score:.3f} < {threshold}[/red]")

    return FaithfulnessResult(
        passed=passed, score=score, threshold=threshold,
        reason=f"Faithfulness score: {score:.3f} ({'passed' if passed else 'failed'}).",
    )
