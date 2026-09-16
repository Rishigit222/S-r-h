"""Heal Loop — Master orchestrator for the Self-Healing RAG Engine.

Drives the complete detect → diagnose → repair → evaluate → deploy/rollback cycle.
This is the core product loop — the autonomous control plane for RAG systems.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from rich.console import Console

from src.config import settings
from src.engine.failure_taxonomy import (
    FailureReport,
    classify_failure,
    classify_config_drift,
)
from src.engine.diagnosis_engine import Diagnosis, diagnosis_engine
from src.engine.repair_planner import RepairPlan, repair_planner
from src.engine.repair_strategies import execute_repair, rollback_repair, RepairResult
from src.engine.sandbox import Sandbox, SandboxCandidate, SandboxEvalResult, sandbox
from src.engine.regression_evaluator import (
    ComparisonVerdict,
    Verdict,
    regression_evaluator,
)
from src.engine.rollback_controller import rollback_controller

console = Console()


@dataclass
class HealCycleResult:
    """Complete result of a single heal cycle."""

    heal_id: str
    """Unique identifier for this heal cycle."""

    query: str
    """The query that was processed."""

    success: bool
    """Whether the cycle ended with a successful outcome."""

    failure_detected: bool
    """Whether a failure was detected."""

    failure_report: FailureReport | None = None
    """The detected failure, if any."""

    diagnosis: Diagnosis | None = None
    """Root cause diagnosis, if failure was detected."""

    repairs_attempted: list[dict[str, Any]] = field(default_factory=list)
    """Log of all repair attempts and their outcomes."""

    final_verdict: ComparisonVerdict | None = None
    """The final before/after comparison verdict."""

    action_taken: str = "none"
    """What action was taken: 'none', 'deployed', 'rolled_back', 'exhausted'."""

    total_duration_ms: float = 0.0
    """Total time spent in the heal cycle."""

    pipeline_response: dict[str, Any] = field(default_factory=dict)
    """The final pipeline response (if successful)."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "heal_id": self.heal_id,
            "query": self.query,
            "success": self.success,
            "failure_detected": self.failure_detected,
            "failure_type": self.failure_report.failure_type.value if self.failure_report else None,
            "diagnosis_root_cause": self.diagnosis.root_cause if self.diagnosis else None,
            "repairs_attempted": self.repairs_attempted,
            "final_verdict": self.final_verdict.to_dict() if self.final_verdict else None,
            "action_taken": self.action_taken,
            "total_duration_ms": round(self.total_duration_ms, 1),
        }


class HealLoop:
    """The autonomous self-healing control loop.

    Orchestrates:
    1. Execute RAG pipeline → get response + trace
    2. Evaluate response quality (multi-dimensional)
    3. If all metrics pass → return SUCCESS
    4. If failure detected → classify failure type
    5. Diagnose root cause
    6. Plan repair strategy
    7. Create sandbox candidate
    8. Evaluate candidate (regression test)
    9. Compare before/after metrics
    10. If improved → deploy repair
    11. If worse → rollback, try next strategy
    12. Log everything to observability layer
    """

    def __init__(
        self,
        max_repair_attempts: int = 3,
        improvement_threshold: float = 0.02,
        latency_sla_ms: float = 10000.0,
    ):
        self.max_repair_attempts = max_repair_attempts
        self.improvement_threshold = improvement_threshold
        self.latency_sla_ms = latency_sla_ms

    def run_once(
        self,
        query: str,
        pipeline_config: dict[str, Any],
        run_pipeline_fn: Callable[[str, dict[str, Any]], dict[str, Any]],
        test_queries: list[str] | None = None,
    ) -> HealCycleResult:
        """Run a single heal cycle on one query.

        Args:
            query: The user query to process.
            pipeline_config: Current mutable pipeline configuration.
            run_pipeline_fn: Callable(query, config) -> dict with keys:
                answer, faithfulness_score, faithfulness_passed, faithfulness_reason,
                relevance_score, relevance_passed, citations_valid, citations_reason,
                generation_error, latency_ms, trace
            test_queries: Optional list of queries for regression testing.
                          Defaults to [query] if not provided.

        Returns:
            HealCycleResult with full details of the cycle.
        """
        start_time = time.time()
        heal_id = f"heal_{uuid.uuid4().hex[:12]}"

        result = HealCycleResult(
            heal_id=heal_id,
            query=query,
            success=False,
            failure_detected=False,
        )

        if test_queries is None:
            test_queries = [query]

        console.print(f"\n[bold white]{'='*60}[/bold white]")
        console.print(f"[bold white]🔄 HEAL CYCLE: {heal_id}[/bold white]")
        console.print(f"[bold white]   Query: {query[:80]}...[/bold white]")
        console.print(f"[bold white]{'='*60}[/bold white]\n")

        # --- Step 1: Run the pipeline ---
        try:
            pipeline_output = run_pipeline_fn(query, pipeline_config)
        except Exception as e:
            pipeline_output = {
                "answer": "",
                "faithfulness_score": 0.0,
                "faithfulness_passed": False,
                "faithfulness_reason": str(e),
                "relevance_score": 0.0,
                "relevance_passed": False,
                "citations_valid": False,
                "citations_reason": str(e),
                "generation_error": str(e),
                "latency_ms": (time.time() - start_time) * 1000,
                "trace": {},
            }

        result.pipeline_response = pipeline_output

        # --- Step 2: Classify failure ---
        failure = classify_failure(
            relevance_passed=pipeline_output.get("relevance_passed", False),
            relevance_score=pipeline_output.get("relevance_score", 0.0),
            relevance_threshold=pipeline_config.get("relevance_threshold", settings.relevance_threshold),
            faithfulness_passed=pipeline_output.get("faithfulness_passed", True),
            faithfulness_score=pipeline_output.get("faithfulness_score", 1.0),
            faithfulness_threshold=pipeline_config.get("faithfulness_threshold", settings.faithfulness_threshold),
            citations_passed=pipeline_output.get("citations_valid", True),
            citations_detail=pipeline_output.get("citations_reason", ""),
            generation_error=pipeline_output.get("generation_error"),
            latency_ms=pipeline_output.get("latency_ms", 0.0),
            latency_sla_ms=self.latency_sla_ms,
            query=query,
            pipeline_trace=pipeline_output.get("trace", {}),
        )

        # --- Step 3: No failure → success ---
        if failure is None:
            console.print("[bold green]✅ HEAL: No failure detected — pipeline healthy[/bold green]")
            result.success = True
            result.failure_detected = False
            result.action_taken = "none"
            result.total_duration_ms = (time.time() - start_time) * 1000
            return result

        # --- Step 4: Failure detected ---
        result.failure_detected = True
        result.failure_report = failure
        console.print(
            f"[bold red]⚠ FAILURE DETECTED: {failure.failure_type.value} "
            f"(severity: {failure.severity.value})[/bold red]"
        )

        # --- Step 5: Diagnose ---
        diagnosis = diagnosis_engine.diagnose(failure)
        result.diagnosis = diagnosis
        console.print(
            f"[bold yellow]🔍 DIAGNOSIS: {diagnosis.root_cause}[/bold yellow]\n"
            f"   Confidence: {diagnosis.confidence:.0%}\n"
            f"   Suggested repairs: {diagnosis.suggested_repairs}"
        )

        # --- Step 6: Save checkpoint before attempting repairs ---
        rollback_controller.save_checkpoint(
            pipeline_config,
            reason=f"Pre-repair for {failure.failure_type.value} on query: {query[:50]}",
        )

        # --- Capture "before" metrics for comparison ---
        before_eval = SandboxEvalResult(
            faithfulness_scores=[pipeline_output.get("faithfulness_score", 0.0)],
            relevance_scores=[pipeline_output.get("relevance_score", 0.0)],
            citation_pass_rates=[pipeline_output.get("citations_valid", False)],
            latencies_ms=[pipeline_output.get("latency_ms", 0.0)],
        )

        # --- Step 7-11: Repair loop ---
        for attempt in range(min(self.max_repair_attempts, len(diagnosis.suggested_repairs))):
            console.print(f"\n[bold cyan]🔧 REPAIR ATTEMPT {attempt + 1}/{self.max_repair_attempts}[/bold cyan]")

            # Plan the repair
            plan = repair_planner.plan(diagnosis, pipeline_config, attempt_index=attempt)
            if plan is None:
                console.print("[yellow]No more repair strategies available.[/yellow]")
                break

            # Create sandbox candidate
            candidate = sandbox.create_candidate(pipeline_config, plan)

            if not candidate.repair_result.success:
                result.repairs_attempted.append({
                    "attempt": attempt + 1,
                    "strategy": plan.strategy,
                    "status": "apply_failed",
                    "error": candidate.repair_result.error,
                })
                continue

            # Evaluate candidate in sandbox
            after_eval = sandbox.evaluate_candidate(
                candidate, test_queries, run_pipeline_fn
            )

            # Compare before/after
            verdict = regression_evaluator.compare(before_eval, after_eval)
            result.final_verdict = verdict

            repair_log = {
                "attempt": attempt + 1,
                "strategy": plan.strategy,
                "verdict": verdict.verdict.value,
                "overall_delta": verdict.overall_delta,
                "explanation": verdict.explanation,
            }
            result.repairs_attempted.append(repair_log)

            if verdict.verdict == Verdict.DEPLOY:
                # Promote the candidate to live
                sandbox.promote_candidate(candidate, pipeline_config)
                result.success = True
                result.action_taken = "deployed"
                console.print(
                    f"[bold green]🚀 DEPLOYED: Repair '{plan.strategy}' improved the system[/bold green]"
                )
                break

            elif verdict.verdict == Verdict.ROLLBACK:
                # Discard the candidate, try next strategy
                sandbox.discard_candidate(candidate)
                console.print(
                    f"[bold red]⏪ SKIPPED: Repair '{plan.strategy}' caused regression[/bold red]"
                )
                continue

            else:  # NEUTRAL
                # Accept neutral changes if we're on the last attempt
                if attempt == self.max_repair_attempts - 1:
                    sandbox.promote_candidate(candidate, pipeline_config)
                    result.success = True
                    result.action_taken = "deployed"
                    console.print("[yellow]Accepting neutral repair as last resort[/yellow]")
                else:
                    sandbox.discard_candidate(candidate)
                    console.print("[yellow]Neutral result — trying next strategy[/yellow]")

        # If no repair succeeded
        if not result.success:
            rollback_controller.auto_rollback(pipeline_config)
            result.action_taken = "rolled_back"
            console.print("[bold red]⏪ ROLLED BACK: All repair strategies exhausted[/bold red]")

        result.total_duration_ms = (time.time() - start_time) * 1000

        console.print(f"\n[bold white]{'='*60}[/bold white]")
        console.print(
            f"[bold white]HEAL CYCLE COMPLETE: {result.action_taken.upper()} "
            f"({result.total_duration_ms:.0f}ms)[/bold white]"
        )
        console.print(f"[bold white]{'='*60}[/bold white]\n")

        return result

    def run_continuous(
        self,
        pipeline_config: dict[str, Any],
        run_pipeline_fn: Callable[[str, dict[str, Any]], dict[str, Any]],
        test_queries: list[str],
        interval_seconds: int = 300,
        max_cycles: int | None = None,
    ) -> list[HealCycleResult]:
        """Run continuous monitoring and healing cycles.

        Args:
            pipeline_config: Mutable pipeline configuration.
            run_pipeline_fn: Pipeline execution function.
            test_queries: Queries to use for continuous evaluation.
            interval_seconds: Seconds between heal cycles.
            max_cycles: Maximum number of cycles (None = infinite).

        Returns:
            List of all HealCycleResults.
        """
        results: list[HealCycleResult] = []
        cycle = 0

        console.print(
            f"[bold green]🔄 CONTINUOUS MONITOR: Starting "
            f"(interval={interval_seconds}s, queries={len(test_queries)})[/bold green]"
        )

        while max_cycles is None or cycle < max_cycles:
            cycle += 1
            console.print(f"\n[bold blue]📡 MONITOR CYCLE {cycle}[/bold blue]")

            for query in test_queries:
                result = self.run_once(query, pipeline_config, run_pipeline_fn, test_queries)
                results.append(result)

                if result.failure_detected and result.action_taken == "deployed":
                    console.print(
                        f"[bold green]🔄 Auto-healed during monitoring cycle {cycle}[/bold green]"
                    )

            if max_cycles is not None and cycle >= max_cycles:
                break

            console.print(f"[dim]Sleeping {interval_seconds}s until next cycle...[/dim]")
            time.sleep(interval_seconds)

        return results


# Module-level singleton
heal_loop = HealLoop(
    max_repair_attempts=settings.max_heal_retries + 1,
    latency_sla_ms=10000.0,
)
