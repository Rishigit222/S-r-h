"""RAG Evaluation Runner — benchmarks system against Golden Dataset."""

import json
import time
from pathlib import Path
import requests
from rich.console import Console

from src.evaluation.metrics import compute_citation_coverage, evaluate_refusal

console = Console()

API_URL = "http://127.0.0.1:8000"
GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"


def run_evaluation(threshold: float = 0.80):
    console.print("[bold cyan]════════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]       📊 RAG ENGINE AUTOMATED EVALUATION SUITE             [/bold cyan]")
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
        console.print(f"[green]✓ Ingested {ingest_resp['chunks_indexed']} chunks.[/green]\n")
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
                "latency_s": round(latency, 2),
            })

            status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
            console.print(f"  Result: {status} | Faithfulness: {faithfulness:.2f} | Latency: {latency:.2f}s\n")

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


if __name__ == "__main__":
    run_evaluation()
