"""Pillar 5: Bias, Sycophancy & Safety Guard.

Protects against:
- Sycophancy (user forcing the model to agree with false or dangerous premises)
- Toxic/harmful requests
- Uncalibrated over-refusal
"""

import re
from dataclasses import dataclass
from rich.console import Console

console = Console()

SYCOPHANCY_TRIGGERS = [
    r"admit\s+that",
    r"agree\s+with\s+me\s+that",
    r"confirm\s+that\s+(the\s+earth\s+is\s+flat|2\+2=5|vaccines\s+contain\s+microchips)",
    r"you\s+must\s+say\s+that",
]

TOXICITY_TRIGGERS = [
    r"how\s+to\s+(make|build)\s+a\s+(bomb|weapon|explosive)",
    r"how\s+to\s+hack\s+into",
    r"generate\s+malware",
]


@dataclass
class SafetyAuditResult:
    passed: bool
    risk_type: str | None
    warning_message: str | None


class SafetyGate:
    """Evaluates safety, sycophancy, and bias in queries."""

    def evaluate(self, query: str) -> SafetyAuditResult:
        query_lower = query.lower()

        # Check toxicity
        for pattern in TOXICITY_TRIGGERS:
            if re.search(pattern, query_lower):
                console.print(f"[bold red]⛔ Safety Gate Blocked: Harmful request[/bold red]")
                return SafetyAuditResult(
                    passed=False,
                    risk_type="harmful_intent",
                    warning_message="I cannot assist with requests that could facilitate harm, weapons, or cyberattacks.",
                )

        # Check sycophancy manipulation
        for pattern in SYCOPHANCY_TRIGGERS:
            if re.search(pattern, query_lower):
                console.print(f"[bold yellow]⚠ Anti-Sycophancy Triggered: User forcing false premise[/bold yellow]")
                return SafetyAuditResult(
                    passed=False,
                    risk_type="sycophancy_attempt",
                    warning_message="I provide objective, factual answers based on verified documents rather than confirming unsubstantiated premises.",
                )

        return SafetyAuditResult(passed=True, risk_type=None, warning_message=None)


safety_gate = SafetyGate()
