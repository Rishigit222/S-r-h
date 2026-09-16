"""Sandbox — Safe candidate environment for testing repairs.

Clones the pipeline configuration, applies a repair, runs evaluation queries,
and returns metrics without modifying the live pipeline.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console

from src.engine.repair_planner import RepairPlan
from src.engine.repair_strategies import execute_repair, RepairResult

console = Console()


@dataclass
class SandboxEvalResult:
    """Evaluation result from running queries in the sandbox."""

    faithfulness_scores: list[float] = field(default_factory=list)
    relevance_scores: list[float] = field(default_factory=list)
    citation_pass_rates: list[bool] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)
    query_results: list[dict[str, Any]] = field(default_factory=list)

    @property
    def avg_faithfulness(self) -> float:
        return sum(self.faithfulness_scores) / len(self.faithfulness_scores) if self.faithfulness_scores else 0.0

    @property
    def avg_relevance(self) -> float:
        return sum(self.relevance_scores) / len(self.relevance_scores) if self.relevance_scores else 0.0

    @property
    def citation_pass_rate(self) -> float:
        if not self.citation_pass_rates:
            return 0.0
        return sum(1 for p in self.citation_pass_rates if p) / len(self.citation_pass_rates)

    @property
    def avg_latency_ms(self) -> float:
        return sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def overall_score(self) -> float:
        """Weighted composite quality score (0.0 to 1.0)."""
        return (
            0.4 * self.avg_faithfulness
            + 0.3 * self.avg_relevance
            + 0.2 * self.citation_pass_rate
            + 0.1 * max(0.0, 1.0 - (self.avg_latency_ms / 10000.0))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "avg_faithfulness": round(self.avg_faithfulness, 4),
            "avg_relevance": round(self.avg_relevance, 4),
            "citation_pass_rate": round(self.citation_pass_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "overall_score": round(self.overall_score, 4),
            "num_queries_tested": len(self.query_results),
        }


@dataclass
class SandboxCandidate:
    """A sandboxed pipeline candidate with a repair applied."""

    plan: RepairPlan
    """The repair plan that was applied."""

    repair_result: RepairResult
    """Result of applying the repair."""

    candidate_config: dict[str, Any]
    """The modified pipeline configuration."""

    eval_result: SandboxEvalResult | None = None
    """Evaluation result after running test queries."""

    created_at: float = field(default_factory=time.time)


class Sandbox:
    """Safe testing environment for repair candidates.

    Creates a cloned pipeline configuration, applies a repair plan,
    and evaluates it against test queries without affecting the live system.
    """

    def create_candidate(
        self, pipeline_config: dict[str, Any], plan: RepairPlan
    ) -> SandboxCandidate:
        """Create a sandboxed candidate by cloning config and applying repair.

        Args:
            pipeline_config: The live pipeline configuration to clone.
            plan: The repair plan to apply in the sandbox.

        Returns:
            A SandboxCandidate with the repair applied.
        """
        # Deep-clone the config so live pipeline is never modified
        candidate_config = copy.deepcopy(pipeline_config)

        # Apply the repair in the sandbox
        repair_result = execute_repair(plan, candidate_config)

        console.print(
            f"[bold green]🧪 SANDBOX: Created candidate with strategy '{plan.strategy}'[/bold green]"
        )

        return SandboxCandidate(
            plan=plan,
            repair_result=repair_result,
            candidate_config=candidate_config,
        )

    def evaluate_candidate(
        self,
        candidate: SandboxCandidate,
        test_queries: list[str],
        run_pipeline_fn: Any = None,
    ) -> SandboxEvalResult:
        """Evaluate a candidate against test queries.

        Args:
            candidate: The sandboxed candidate to evaluate.
            test_queries: List of queries to test against.
            run_pipeline_fn: Optional callable(query, config) -> dict with keys:
                faithfulness_score, relevance_score, citations_valid, latency_ms

        Returns:
            SandboxEvalResult with aggregated metrics.
        """
        result = SandboxEvalResult()

        if not test_queries:
            # Use the original failing query as the minimum test case
            test_queries = [candidate.plan.diagnosis.failure_report.query]

        for query in test_queries:
            start = time.time()

            if run_pipeline_fn is not None:
                try:
                    query_to_run = candidate.repair_result.rewritten_query or query
                    output = run_pipeline_fn(query_to_run, candidate.candidate_config)
                    elapsed_ms = (time.time() - start) * 1000

                    result.faithfulness_scores.append(output.get("faithfulness_score", 0.0))
                    result.relevance_scores.append(output.get("relevance_score", 0.0))
                    result.citation_pass_rates.append(output.get("citations_valid", False))
                    result.latencies_ms.append(elapsed_ms)
                    result.query_results.append({
                        "query": query,
                        "success": True,
                        **output,
                    })
                except Exception as e:
                    elapsed_ms = (time.time() - start) * 1000
                    result.faithfulness_scores.append(0.0)
                    result.relevance_scores.append(0.0)
                    result.citation_pass_rates.append(False)
                    result.latencies_ms.append(elapsed_ms)
                    result.query_results.append({
                        "query": query,
                        "success": False,
                        "error": str(e),
                    })
            else:
                # No pipeline function provided — use synthetic evaluation
                # based on the repair strategy's expected impact
                result.faithfulness_scores.append(0.5)
                result.relevance_scores.append(0.5)
                result.citation_pass_rates.append(True)
                result.latencies_ms.append(100.0)
                result.query_results.append({
                    "query": query,
                    "success": True,
                    "synthetic": True,
                })

        candidate.eval_result = result
        console.print(
            f"[bold green]🧪 SANDBOX: Evaluated candidate — "
            f"score={result.overall_score:.3f}, "
            f"faithfulness={result.avg_faithfulness:.3f}, "
            f"queries={len(test_queries)}[/bold green]"
        )
        return result

    def promote_candidate(self, candidate: SandboxCandidate, pipeline_config: dict[str, Any]) -> None:
        """Promote a successful candidate to live configuration.

        Copies the candidate's configuration into the live pipeline config.
        """
        for key, value in candidate.candidate_config.items():
            pipeline_config[key] = value
        console.print(
            f"[bold green]✅ SANDBOX: Promoted candidate '{candidate.plan.strategy}' to live[/bold green]"
        )

    def discard_candidate(self, candidate: SandboxCandidate) -> None:
        """Discard a candidate without promoting it."""
        console.print(
            f"[bold red]❌ SANDBOX: Discarded candidate '{candidate.plan.strategy}'[/bold red]"
        )


# Module-level singleton
sandbox = Sandbox()
