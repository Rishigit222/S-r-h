"""Unit tests for the Self-Healing RAG Engine core modules.

Tests:
1. Failure Taxonomy & Classification
2. Diagnosis Engine
3. Repair Planner & Strategy Registry
4. Sandbox Evaluation
5. Regression Evaluator & Verdict Determination
6. Rollback Controller & Checkpoint Restoration
7. Heal Loop Execution
"""

import pytest
from pathlib import Path

from src.engine.failure_taxonomy import (
    FailureType,
    Severity,
    FailureReport,
    classify_failure,
    classify_config_drift,
)
from src.engine.diagnosis_engine import (
    DiagnosisEngine,
    Diagnosis,
    RepairStrategyType,
    diagnosis_engine,
)
from src.engine.repair_planner import (
    RepairPlan,
    RepairPlanner,
    repair_planner,
)
from src.engine.repair_strategies import (
    STRATEGY_REGISTRY,
    BaseRepairStrategy,
    execute_repair,
    rollback_repair,
)
from src.engine.sandbox import (
    Sandbox,
    SandboxCandidate,
    SandboxEvalResult,
    sandbox,
)
from src.engine.regression_evaluator import (
    Verdict,
    ComparisonVerdict,
    MetricDelta,
    RegressionEvaluator,
    regression_evaluator,
)
from src.engine.rollback_controller import (
    Checkpoint,
    RollbackController,
)
from src.engine.heal_loop import (
    HealLoop,
    HealCycleResult,
)


# =====================================================================
# 1. FAILURE TAXONOMY & CLASSIFICATION TESTS
# =====================================================================

def test_classify_retrieval_failure():
    report = classify_failure(
        relevance_passed=False,
        relevance_score=0.15,
        relevance_threshold=0.30,
        query="What is the quantum state?",
    )
    assert report is not None
    assert report.failure_type == FailureType.RETRIEVAL_FAILURE
    assert report.severity == Severity.MEDIUM
    assert report.evidence[0].observed_value == 0.15


def test_classify_grounding_failure():
    report = classify_failure(
        relevance_passed=True,
        relevance_score=0.85,
        faithfulness_passed=False,
        faithfulness_score=0.20,
        faithfulness_threshold=0.60,
        query="Explain photosynthesis.",
    )
    assert report is not None
    assert report.failure_type == FailureType.GROUNDING_FAILURE
    assert report.severity == Severity.HIGH
    assert report.evidence[0].observed_value == 0.20


def test_classify_generation_failure():
    report = classify_failure(
        generation_error="LLM Provider 503 Overloaded",
        query="Generate report",
    )
    assert report is not None
    assert report.failure_type == FailureType.GENERATION_FAILURE
    assert report.severity == Severity.CRITICAL
    assert report.evidence[0].details.get("error") == "LLM Provider 503 Overloaded"


def test_classify_citation_failure():
    report = classify_failure(
        citations_passed=False,
        citations_detail="Cited index [9] exceeds context size 2",
        query="Source check",
    )
    assert report is not None
    assert report.failure_type == FailureType.CITATION_FAILURE
    assert report.severity == Severity.MEDIUM


def test_classify_latency_sla_violation():
    report = classify_failure(
        latency_ms=12500.0,
        latency_sla_ms=10000.0,
        query="Speed check",
    )
    assert report is not None
    assert report.failure_type == FailureType.LATENCY_FAILURE
    assert report.severity == Severity.LOW


def test_classify_no_failure():
    report = classify_failure(
        latency_ms=1200.0,
        latency_sla_ms=10000.0,
        query="Healthy query",
    )
    assert report is None


def test_classify_config_drift():
    drift = classify_config_drift(
        hallucination_rate=0.45,
        refusal_rate=0.05,
        avg_latency_ms=2000.0,
    )
    assert drift is not None
    assert drift.failure_type == FailureType.CONFIG_FAILURE
    assert drift.severity == Severity.HIGH

    drift_lat = classify_config_drift(
        hallucination_rate=0.05,
        refusal_rate=0.05,
        avg_latency_ms=12000.0,
        latency_threshold_ms=8000.0,
    )
    assert drift_lat is not None
    assert drift_lat.failure_type == FailureType.CONFIG_FAILURE


# =====================================================================
# 2. DIAGNOSIS ENGINE TESTS
# =====================================================================

def test_diagnosis_retrieval_failure():
    failure = classify_failure(
        relevance_passed=False,
        relevance_score=0.10,
        query="Neural network training",
    )
    diagnosis = diagnosis_engine.diagnose(failure)
    assert diagnosis.root_cause != ""
    assert diagnosis.confidence >= 0.7
    assert RepairStrategyType.ADJUST_RETRIEVAL in diagnosis.suggested_repairs
    assert RepairStrategyType.REWRITE_QUERY in diagnosis.suggested_repairs


def test_diagnosis_grounding_failure():
    failure = classify_failure(
        relevance_passed=True,
        faithfulness_passed=False,
        faithfulness_score=0.25,
        query="DNA replication",
    )
    diagnosis = diagnosis_engine.diagnose(failure)
    assert RepairStrategyType.REGENERATE in diagnosis.suggested_repairs
    assert RepairStrategyType.REWRITE_QUERY in diagnosis.suggested_repairs


def test_diagnosis_generation_failure():
    failure = classify_failure(
        generation_error="Rate limit reached",
        query="Testing error",
    )
    diagnosis = diagnosis_engine.diagnose(failure)
    assert len(diagnosis.suggested_repairs) > 0


# =====================================================================
# 3. REPAIR PLANNER & STRATEGIES TESTS
# =====================================================================

def test_repair_planner_plan():
    failure = classify_failure(
        relevance_passed=False,
        relevance_score=0.12,
        query="Database indexing",
    )
    diagnosis = diagnosis_engine.diagnose(failure)
    config = {
        "vector_weight": 0.5,
        "bm25_weight": 0.5,
        "top_k_retrieval": 10,
        "relevance_threshold": 0.30,
    }

    plan = repair_planner.plan(
        diagnosis=diagnosis,
        current_config=config,
    )
    assert plan is not None
    assert plan.strategy in diagnosis.suggested_repairs
    assert isinstance(plan.rollback_snapshot, dict)


def test_strategy_registry_completeness():
    expected_strategies = {
        RepairStrategyType.REWRITE_QUERY,
        RepairStrategyType.RECHUNK_DOCUMENTS,
        RepairStrategyType.ADJUST_RETRIEVAL,
        RepairStrategyType.EXPAND_CONTEXT,
        RepairStrategyType.ADJUST_THRESHOLDS,
        RepairStrategyType.SWAP_RERANKER,
        RepairStrategyType.REGENERATE,
    }
    for strategy in expected_strategies:
        assert strategy in STRATEGY_REGISTRY
        assert isinstance(STRATEGY_REGISTRY[strategy], BaseRepairStrategy)


def test_execute_and_rollback_repair():
    config = {
        "vector_weight": 0.5,
        "bm25_weight": 0.5,
        "relevance_threshold": 0.30,
        "top_k_retrieval": 10,
    }
    failure = classify_failure(
        relevance_passed=False,
        relevance_score=0.10,
        query="Test query",
    )
    diagnosis = diagnosis_engine.diagnose(failure)

    plan = RepairPlan(
        strategy=RepairStrategyType.ADJUST_RETRIEVAL,
        parameters={
            "vector_weight": 0.6,
            "bm25_weight": 0.4,
            "top_k_retrieval": 15,
        },
        rollback_snapshot=dict(config),
        estimated_impact="Adjusted retrieval",
        diagnosis=diagnosis,
    )

    # Execute
    res = execute_repair(plan, config)
    assert res.success is True
    assert config["top_k_retrieval"] == 15
    assert config["vector_weight"] == 0.6

    # Rollback
    rollback_repair(plan, config)
    assert config["top_k_retrieval"] == 10
    assert config["vector_weight"] == 0.5


# =====================================================================
# 4. SANDBOX EVALUATION TESTS
# =====================================================================

def test_sandbox_lifecycle():
    box = Sandbox()
    base_config = {"relevance_threshold": 0.30, "top_k_retrieval": 5}
    failure = classify_failure(
        relevance_passed=False,
        relevance_score=0.10,
        query="Test query",
    )
    diagnosis = diagnosis_engine.diagnose(failure)

    plan = RepairPlan(
        strategy=RepairStrategyType.REWRITE_QUERY,
        parameters={"rewritten_query": "better query"},
        rollback_snapshot=dict(base_config),
        estimated_impact="Rewrite query",
        diagnosis=diagnosis,
    )

    candidate = box.create_candidate(pipeline_config=base_config, plan=plan)
    assert candidate.plan.strategy == RepairStrategyType.REWRITE_QUERY
    assert candidate.candidate_config == base_config

    # Dummy evaluation function
    def dummy_pipeline(q, cfg):
        return {
            "faithfulness_score": 0.95,
            "relevance_score": 0.88,
            "citations_valid": True,
            "latency_ms": 120.0,
            "generation_error": None,
        }

    eval_result = box.evaluate_candidate(candidate, ["q1", "q2"], run_pipeline_fn=dummy_pipeline)
    assert eval_result.avg_faithfulness == 0.95
    assert eval_result.overall_score > 0.80


# =====================================================================
# 5. REGRESSION EVALUATOR TESTS
# =====================================================================

def test_regression_evaluator_deploy():
    baseline = SandboxEvalResult(
        faithfulness_scores=[0.50],
        relevance_scores=[0.40],
        citation_pass_rates=[False],
        latencies_ms=[500.0],
    )
    candidate = SandboxEvalResult(
        faithfulness_scores=[0.85],
        relevance_scores=[0.80],
        citation_pass_rates=[True],
        latencies_ms=[450.0],
    )

    verdict = regression_evaluator.compare(baseline, candidate)
    assert verdict.verdict == Verdict.DEPLOY
    assert verdict.overall_delta > 0.0


def test_regression_evaluator_rollback():
    baseline = SandboxEvalResult(
        faithfulness_scores=[0.85],
        relevance_scores=[0.80],
        citation_pass_rates=[True],
        latencies_ms=[300.0],
    )
    candidate = SandboxEvalResult(
        faithfulness_scores=[0.40],
        relevance_scores=[0.40],
        citation_pass_rates=[False],
        latencies_ms=[900.0],
    )

    verdict = regression_evaluator.compare(baseline, candidate)
    assert verdict.verdict == Verdict.ROLLBACK
    assert verdict.overall_delta < 0.0


# =====================================================================
# 6. ROLLBACK CONTROLLER TESTS
# =====================================================================

def test_rollback_controller(tmp_path):
    test_db = tmp_path / "test_rollback.sqlite"
    controller = RollbackController(db_path=test_db)

    config1 = {"relevance_threshold": 0.35, "vector_weight": 0.6}
    cp1 = controller.save_checkpoint(
        pipeline_config=config1,
        reason="Initial stable state",
    )
    assert cp1.checkpoint_id.startswith("cp_")

    config2 = {"relevance_threshold": 0.15, "vector_weight": 0.8}
    cp2 = controller.save_checkpoint(
        pipeline_config=config2,
        reason="Experimental aggressive threshold",
    )

    checkpoints = controller.list_checkpoints(limit=10)
    assert len(checkpoints) == 2
    assert checkpoints[0]["checkpoint_id"] == cp2.checkpoint_id

    # Restore to initial stable state
    restored_config = {}
    success = controller.restore_checkpoint(cp1.checkpoint_id, restored_config)
    assert success is True
    assert restored_config["relevance_threshold"] == 0.35
    assert restored_config["vector_weight"] == 0.6


# =====================================================================
# 7. HEAL LOOP TESTS
# =====================================================================

def test_heal_loop_healthy_query():
    loop = HealLoop(max_repair_attempts=2)
    config = {"relevance_threshold": 0.30}

    def healthy_pipeline(q, cfg):
        return {
            "answer": "This is a factual answer.",
            "faithfulness_score": 0.95,
            "faithfulness_passed": True,
            "faithfulness_reason": "High entailment",
            "relevance_score": 0.85,
            "relevance_passed": True,
            "citations_valid": True,
            "citations_reason": "All sources verified",
            "generation_error": None,
            "latency_ms": 250.0,
            "trace": {},
        }

    res = loop.run_once("Explain quantum tunneling", config, healthy_pipeline)
    assert res.success is True
    assert res.failure_detected is False
    assert res.action_taken == "none"
