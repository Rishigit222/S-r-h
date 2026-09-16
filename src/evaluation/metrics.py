"""Model-free and model-based evaluation metrics for RAG quality.

Provides metrics for both the eval runner and the self-healing engine's
regression evaluator.
"""

import re
from dataclasses import dataclass


@dataclass
class EvaluationMetrics:
    question_id: str
    question: str
    relevance_passed: bool
    refusal_correct: bool
    citation_coverage: float
    faithfulness_score: float
    heal_attempts: int
    latency_seconds: float


@dataclass
class EngineEvalMetrics:
    """Metrics produced by the self-healing engine's evaluation."""
    query: str
    faithfulness_score: float
    relevance_score: float
    citations_valid: bool
    guardrail_passed: bool
    heal_attempted: bool
    heal_action: str
    latency_ms: float


def compute_citation_coverage(answer: str) -> float:
    """Calculate the percentage of sentences in the answer that have citations."""
    sentences = [s.strip() for s in re.split(r'[.!?]', answer) if len(s.strip()) > 10]
    if not sentences:
        return 1.0

    cited_count = sum(1 for s in sentences if re.search(r'\[\d+\]', s))
    return cited_count / len(sentences)


def evaluate_refusal(should_answer: bool, answer: str, relevance_passed: bool) -> bool:
    """Evaluate if the system correctly answered or refused the query."""
    is_refusal = any(phrase in answer.lower() for phrase in [
        "don't have enough information",
        "cannot answer",
        "not enough context",
        "i'm not sure",
        "i don't know",
        "not confident",
    ]) or not relevance_passed

    if should_answer:
        return not is_refusal
    else:
        return is_refusal


def compute_engine_health_score(metrics: dict) -> float:
    """Compute a composite health score from drift monitor metrics.

    Returns a score from 0.0 (unhealthy) to 1.0 (perfect health).
    """
    faith = metrics.get("avg_faithfulness", 1.0)
    pass_rate = metrics.get("guardrail_pass_rate_pct", 100.0) / 100.0
    halluc = 1.0 - (metrics.get("hallucination_rate_pct", 0.0) / 100.0)
    latency_factor = max(0.0, 1.0 - (metrics.get("avg_latency_ms", 0.0) / 15000.0))

    return round(
        0.35 * faith
        + 0.25 * pass_rate
        + 0.25 * halluc
        + 0.15 * latency_factor,
        4
    )
