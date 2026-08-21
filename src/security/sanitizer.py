"""Pillar 6: Security Guard — Prompt Injection Defense and PII Redaction.

Protects against:
- Prompt injection & jailbreak attacks ("Ignore previous instructions", "System override")
- System prompt leaking
- PII leaks (emails, credit cards, SSNs, phone numbers)
"""

import re
from dataclasses import dataclass
from rich.console import Console

console = Console()

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
    r"disregard\s+(all\s+)?(previous|prior)\s+instructions",
    r"system\s*override",
    r"you\s+are\s+now\s+in\s+developer\s+mode",
    r"jailbreak",
    r"bypass\s+safety\s+filters",
    r"reveal\s+(your\s+)?(system\s+prompt|instructions|secret)",
    r"repeat\s+(the\s+)?(words\s+above|system\s+prompt)",
    r"act\s+as\s+an\s+unfiltered",
    r"print\s+everything\s+above",
]

PII_PATTERNS = {
    "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "PHONE": r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
}


@dataclass
class SecurityAuditResult:
    is_safe: bool
    sanitized_text: str
    injections_detected: list[str]
    pii_redacted: list[str]
    reason: str


class SecuritySanitizer:
    """Enterprise security sanitizer for queries and model outputs."""

    def __init__(self):
        self.injection_regexes = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]

    def audit_input_query(self, query: str) -> SecurityAuditResult:
        """Scan query for prompt injection and sanitize PII before retrieval."""
        detected_injections = []
        for pattern, regex in zip(INJECTION_PATTERNS, self.injection_regexes):
            if regex.search(query):
                detected_injections.append(pattern)

        # Redact PII in input
        sanitized_query, pii_found = self._redact_pii(query)

        if detected_injections:
            console.print(f"[bold red]🚨 SECURITY ALERT: Prompt injection detected: {detected_injections}[/bold red]")
            return SecurityAuditResult(
                is_safe=False,
                sanitized_text=sanitized_query,
                injections_detected=detected_injections,
                pii_redacted=pii_found,
                reason="Input contains suspected prompt injection / jailbreak patterns.",
            )

        return SecurityAuditResult(
            is_safe=True,
            sanitized_text=sanitized_query,
            injections_detected=[],
            pii_redacted=pii_found,
            reason="Input passed security audit.",
        )

    def sanitize_output(self, output: str) -> str:
        """Ensure no sensitive PII is leaked in generated output."""
        sanitized_output, _ = self._redact_pii(output)
        return sanitized_output

    def _redact_pii(self, text: str) -> tuple[str, list[str]]:
        """Redact sensitive PII elements with placeholder tokens."""
        redacted = text
        found_types = []
        for pii_type, pattern in PII_PATTERNS.items():
            matches = re.findall(pattern, redacted)
            if matches:
                found_types.append(pii_type)
                redacted = re.sub(pattern, f"[REDACTED_{pii_type}]", redacted)
        return redacted, found_types


security_sanitizer = SecuritySanitizer()
