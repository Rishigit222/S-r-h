"""Regression Evaluator — Before/after metric comparison with improvement detection.

Compares two evaluation snapshots and produces a verdict on whether
a repair improved the system, had no effect, or caused a regression.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from enum import Enum

from rich.console import Console

from src.engine.sandbox import SandboxEvalResult

console = Console()


class Verdict(str, Enum):
    """Possible outcomes of a before/after comparison."""

    DEPLOY = "deploy"
    """Repair improved the system — safe to deploy."""

    ROLLBACK = "rollback"
    """Repair degraded the system — should rollback."""

    NEUTRAL = "neutral"
    """No significant change — deployment is optional."""


@dataclass
class MetricDelta:
    """Change in a single metric between before and after."""

    metric_name: str
    before: float
    after: float

    @property
    def delta(self) -> float:
        return self.after - self.before

    @property
    def improved(self) -> bool:
        """Whether this metric improved (higher is better for quality metrics)."""
        if self.metric_name == "avg_latency_ms":
            return self.delta < 0  # Lower latency is better
        return self.delta > 0  # Higher quality scores are better

    @property
    def regressed(self) -> bool:
        if self.metric_name == "avg_latency_ms":
            return self.delta > 0
        return self.delta < 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric_name,
            "before": round(self.before, 4),
            "after": round(self.after, 4),
            "delta": round(self.delta, 4),
            "improved": self.improved,
        }


@dataclass
class ComparisonVerdict:
    """Full comparison result between before and after evaluation."""

    verdict: Verdict
    """The recommendation: DEPLOY, ROLLBACK, or NEUTRAL."""

    overall_delta: float
    """Change in overall composite score."""

    metric_deltas: list[MetricDelta]
    """Per-metric comparison details."""

    regression_detected: bool
    """Whether any critical metric regressed significantly."""

    regressed_metrics: list[str]
    """Names of metrics that regressed."""

    improved_metrics: list[str]
    """Names of metrics that improved."""

    explanation: str
    """Human-readable explanation of the verdict."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "overall_delta": round(self.overall_delta, 4),
            "metric_deltas": [d.to_dict() for d in self.metric_deltas],
            "regression_detected": self.regression_detected,
            "regressed_metrics": self.regressed_metrics,
            "improved_metrics": self.improved_metrics,
            "explanation": self.explanation,
        }


class RegressionEvaluator:
    """Compares before/after evaluation snapshots and produces deployment verdicts.

    Uses configurable thresholds to determine if a repair is an improvement,
    neutral, or a regression.
    """

    def __init__(
        self,
        improvement_threshold: float = 0.02,
        regression_threshold: float = -0.03,
        critical_regression_threshold: float = -0.10,
    ):
        """
        Args:
            improvement_threshold: Minimum overall_score delta to recommend DEPLOY.
            regression_threshold: Overall_score delta below which to recommend ROLLBACK.
            critical_regression_threshold: Per-metric delta below which regression is flagged.
        """
        self.improvement_threshold = improvement_threshold
        self.regression_threshold = regression_threshold
        self.critical_regression_threshold = critical_regression_threshold

    def compare(
        self,
        before: SandboxEvalResult,
        after: SandboxEvalResult,
    ) -> ComparisonVerdict:
        """Compare before and after evaluation results.

        Args:
            before: Evaluation metrics before the repair.
            after: Evaluation metrics after the repair.

        Returns:
            ComparisonVerdict with deploy/rollback recommendation.
        """
        metric_deltas = [
            MetricDelta("avg_faithfulness", before.avg_faithfulness, after.avg_faithfulness),
            MetricDelta("avg_relevance", before.avg_relevance, after.avg_relevance),
            MetricDelta("citation_pass_rate", before.citation_pass_rate, after.citation_pass_rate),
            MetricDelta("avg_latency_ms", before.avg_latency_ms, after.avg_latency_ms),
            MetricDelta("overall_score", before.overall_score, after.overall_score),
        ]

        overall_delta = after.overall_score - before.overall_score
        regressed = [d.metric_name for d in metric_deltas if d.regressed and d.metric_name != "avg_latency_ms" and d.delta < self.critical_regression_threshold]
        improved = [d.metric_name for d in metric_deltas if d.improved]

        # Check for latency regression separately
        latency_delta = next((d for d in metric_deltas if d.metric_name == "avg_latency_ms"), None)
        if latency_delta and latency_delta.delta > 2000:  # >2s increase is a regression
            regressed.append("avg_latency_ms")

        regression_detected = len(regressed) > 0

        # Determine verdict
        if regression_detected and any(m in ("avg_faithfulness", "avg_relevance") for m in regressed):
            verdict = Verdict.ROLLBACK
            explanation = f"Critical regression detected in: {', '.join(regressed)}. Overall delta: {overall_delta:+.4f}. Recommending rollback."
        elif overall_delta >= self.improvement_threshold:
            verdict = Verdict.DEPLOY
            explanation = f"Repair improved overall score by {overall_delta:+.4f}. Improved metrics: {', '.join(improved) or 'none'}. Recommending deployment."
        elif overall_delta <= self.regression_threshold:
            verdict = Verdict.ROLLBACK
            explanation = f"Repair degraded overall score by {overall_delta:+.4f}. Regressed metrics: {', '.join(regressed) or 'none'}. Recommending rollback."
        else:
            verdict = Verdict.NEUTRAL
            explanation = f"Repair had minimal effect (delta: {overall_delta:+.4f}). No significant improvement or regression detected."

        result = ComparisonVerdict(
            verdict=verdict,
            overall_delta=overall_delta,
            metric_deltas=metric_deltas,
            regression_detected=regression_detected,
            regressed_metrics=regressed,
            improved_metrics=improved,
            explanation=explanation,
        )

        color = {"deploy": "green", "rollback": "red", "neutral": "yellow"}[verdict.value]
        console.print(f"[bold {color}]📊 VERDICT: {verdict.value.upper()} — {explanation}[/bold {color}]")

        return result


# Module-level singleton
regression_evaluator = RegressionEvaluator()
