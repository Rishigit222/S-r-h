"""RAG Evaluation Runner — benchmarks system against Golden Dataset.

Supports two modes:
1. Standard mode: Runs queries against the API and evaluates responses
2. Engine mode: Runs queries through the heal loop and evaluates self-healing
"""

import json
import time
from pathlib import Path
import requests
from rich.console import Console

from src.evaluation.metrics import compute_citation_coverage, evaluate_refusal, compute_engine_health_score

console = Console()

API_URL = "http://127.0.0.1:8000"
GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"


def run_evaluation(threshold: float = 0.80):
    """Run the standard evaluation suite against the API."""
    console.print("[bold cyan]════════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]       📊 SELF-HEALING RAG ENGINE — EVALUATION SUITE        [/bold cyan]")
    console.print("[bold cyan]════════════════════════════════════════════════════════════[/bold cyan]\n")

    if not GOLDEN_DATASET_PATH.exists():
        console.print(f"[red]Golden dataset not found at {GOLDEN_DATASET_PATH}[/red]")
        return

    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    # First ingest sample docs
    console.print("[cyan]Ingesting sample documents for evaluation...[/cyan]")
    try:
        ingest_resp = requests.post(f"{API_URL}/ingest", timeout=60).json()
        console.print(f"[green][OK] Ingested {ingest_resp['chunks_indexed']} chunks.[/green]\n")
    except Exception as e:
        console.print(f"[red]Failed to connect to API at {API_URL}. Start server first with: make run-api[/red]")
        return

    results = []
    passed_count = 0

    for item in test_cases:
        q_id = item["id"]
        question = item["question"]
        should_answer = item["should_answer"]

        console.print(f"[yellow]Evaluating [{q_id}]: {question}[/yellow]")
        start_time = time.time()

        try:
            resp = requests.post(
                f"{API_URL}/query",
                json={"question": question, "top_k": 5},
                timeout=60,
            ).json()
            latency = time.time() - start_time

            answer = resp.get("answer", "")
            relevance_passed = resp.get("guardrail_passed", False)
            faithfulness = resp.get("faithfulness_score", 0.0)
            heal_attempts = resp.get("heal_attempts", 0)

            refusal_ok = evaluate_refusal(should_answer, answer, relevance_passed)
            citation_cov = compute_citation_coverage(answer) if should_answer else 1.0

            passed = refusal_ok and (faithfulness >= 0.5 or not should_answer)
            if passed:
                passed_count += 1

            results.append({
                "id": q_id,
                "question": question,
                "should_answer": should_answer,
                "passed": passed,
                "refusal_correct": refusal_ok,
                "faithfulness_score": faithfulness,
                "citation_coverage": citation_cov,
                "heal_attempts": heal_attempts,
                "latency_s": round(latency, 2),
            })

            status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
            console.print(f"  Result: {status} | Faithfulness: {faithfulness:.2f} | Heals: {heal_attempts} | Latency: {latency:.2f}s\n")

        except Exception as e:
            console.print(f"  [red]Error testing [{q_id}]: {e}[/red]\n")

    # Aggregate metrics
    total = len(test_cases)
    pass_rate = passed_count / total if total > 0 else 0.0

    console.print(f"[bold cyan]════════════════════════════════════════════════════════════[/bold cyan]")
    console.print(f"[bold green]OVERALL SCORE: {pass_rate:.0%} ({passed_count}/{total} test cases passed)[/bold green]")
    console.print(f"[bold cyan]════════════════════════════════════════════════════════════[/bold cyan]")

    return {
        "pass_rate": pass_rate,
        "passed_count": passed_count,
        "total": total,
        "results": results,
    }


def run_engine_health_check():
    """Run an engine health check via the API's telemetry endpoint."""
    console.print("[bold cyan]════════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]       🏥 SELF-HEALING RAG ENGINE — HEALTH CHECK            [/bold cyan]")
    console.print("[bold cyan]════════════════════════════════════════════════════════════[/bold cyan]\n")

    try:
        # Get telemetry summary
        health_resp = requests.get(f"{API_URL}/health", timeout=10).json()
        telemetry_resp = requests.get(f"{API_URL}/engine/telemetry", timeout=10).json()
        drift_resp = requests.get(f"{API_URL}/engine/drift", timeout=10).json()

        metrics = telemetry_resp.get("summary", {})
        health_score = compute_engine_health_score(metrics)

        console.print(f"[bold]Engine Name:        {health_resp.get('engine_name', 'unknown')}[/bold]")
        console.print(f"[bold]Generation Version: v{health_resp.get('generation_version', '?')}[/bold]")
        console.print(f"[bold]Health Score:        {health_score:.0%}[/bold]")
        console.print(f"[bold]Total Queries:      {metrics.get('total_queries', 0)}[/bold]")
        console.print(f"[bold]Avg Faithfulness:   {metrics.get('avg_faithfulness', 0):.1%}[/bold]")
        console.print(f"[bold]Hallucination Rate: {metrics.get('hallucination_rate_pct', 0):.1f}%[/bold]")
        console.print(f"[bold]Guardrail Pass:     {metrics.get('guardrail_pass_rate_pct', 0):.1f}%[/bold]")
        console.print(f"[bold]Avg Latency:        {metrics.get('avg_latency_ms', 0):.0f}ms[/bold]")
        console.print(f"[bold]Total Heal Events:  {metrics.get('total_heal_events', 0)}[/bold]")
        console.print(f"[bold]Drift Detected:     {drift_resp.get('drift_detected', False)}[/bold]")

        if drift_resp.get("drift_detected"):
            console.print(f"\n[bold red]⚠ DRIFT REPORT:[/bold red]")
            report = drift_resp.get("drift_report", {})
            console.print(f"  Failure Type: {report.get('failure_type', '?')}")
            console.print(f"  Severity: {report.get('severity', '?')}")

        return {
            "health_score": health_score,
            "metrics": metrics,
            "drift_detected": drift_resp.get("drift_detected", False),
        }

    except Exception as e:
        console.print(f"[red]Failed to connect to API: {e}[/red]")
        return None


if __name__ == "__main__":
    run_evaluation()
