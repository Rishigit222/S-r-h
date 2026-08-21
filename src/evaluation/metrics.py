"""Model-free and model-based evaluation metrics for RAG quality."""

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
    ]) or not relevance_passed

    if should_answer:
        return not is_refusal
    else:
        return is_refusal
