"""Citation verifier — checks that answer citations map to real sources.

Rule-based check (no ML model needed):
1. Parses [1], [2], etc. from the answer
2. Verifies each maps to an actual provided context chunk
"""

import re
from dataclasses import dataclass, field
from rich.console import Console

console = Console()


@dataclass
class CitationResult:
    passed: bool
    cited_sources: list[int] = field(default_factory=list)
    valid_range: int = 0
    reason: str = ""


def verify_citations(answer: str, num_context_chunks: int) -> CitationResult:
    """Verify that citations in the answer reference valid context chunks."""
    citations = [int(m) for m in re.findall(r'\[(\d+)\]', answer)]
    unique_citations = sorted(set(citations))

    if not unique_citations:
        is_refusal = any(phrase in answer.lower() for phrase in [
            "don't have enough information",
            "cannot answer",
            "not enough context",
            "i'm not sure",
            "i don't know",
        ])

        if is_refusal:
            return CitationResult(
                passed=True, valid_range=num_context_chunks,
                reason="Answer is a valid refusal (no citations needed).",
            )

        console.print("[yellow]⚠ No citations found in answer[/yellow]")
        return CitationResult(
            passed=False, valid_range=num_context_chunks,
            reason="Answer contains no citations. All claims should cite sources.",
        )

    invalid = [c for c in unique_citations if c < 1 or c > num_context_chunks]

    if invalid:
        console.print(f"[red]✗ Invalid citations: {invalid} (valid range: 1-{num_context_chunks})[/red]")
        return CitationResult(
            passed=False, cited_sources=unique_citations,
            valid_range=num_context_chunks,
            reason=f"Citations {invalid} are out of range (valid: 1-{num_context_chunks}).",
        )

    console.print(f"[green][OK] Citations valid: {unique_citations}[/green]")
    return CitationResult(
        passed=True, cited_sources=unique_citations,
        valid_range=num_context_chunks,
        reason=f"All {len(unique_citations)} citations are valid.",
    )
