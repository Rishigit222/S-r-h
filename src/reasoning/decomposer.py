"""Pillar 2: Multi-Hop Reasoning & Query Decomposition.

Handles complex, multi-part, comparative, conversational, or dependent questions by:
1. Extracting core technical questions from conversational preambles
2. Decomposing complex queries into atomic sub-questions
3. Performing targeted sub-retrievals
4. Merging and reconciling context across multiple documents
"""

import re
from dataclasses import dataclass
from rich.console import Console

console = Console()


@dataclass
class DecompositionResult:
    is_complex: bool
    sub_queries: list[str]
    reasoning_type: str  # "comparative" | "multi_part" | "atomic"


class QueryDecomposer:
    """Decomposes multi-hop or comparative questions into parallel sub-queries."""

    def decompose(self, query: str) -> DecompositionResult:
        query_lower = query.lower()

        # 1. Clean conversational preambles if multiple sentences exist
        sentences = [s.strip() for s in re.split(r'[.!?\n]+', query) if len(s.strip()) > 3]
        target_query = query
        if len(sentences) > 1:
            # Find the sentence that is an actual question or command
            q_sents = [s for s in sentences if any(w in s.lower() for w in ["what", "how", "why", "explain", "compare", "describe", "is", "can", "define"])]
            if q_sents:
                target_query = q_sents[-1]

        target_lower = target_query.lower()

        # 2. Check for comparative queries
        if any(w in target_lower for w in ["compare", "difference between", "versus", " vs "]):
            sub_queries = [
                f"What is {target_query}?",
                f"Key differences and features in {target_query}",
            ]
            console.print(f"[cyan]🧠 Multi-Hop Decomposer: Detected comparative query → 2 sub-queries[/cyan]")
            return DecompositionResult(
                is_complex=True,
                sub_queries=sub_queries,
                reasoning_type="comparative",
            )

        # 3. Check for compound queries with 'and', 'also', 'as well as'
        if " and " in target_lower and len(target_query.split()) > 6:
            parts = re.split(r'\s+and\s+|\s+as well as\s+', target_query, flags=re.IGNORECASE)
            sub_queries = [p.strip() for p in parts if len(p.strip()) > 5]
            if len(sub_queries) > 1:
                console.print(f"[cyan]🧠 Multi-Hop Decomposer: Split into {len(sub_queries)} sub-queries[/cyan]")
                return DecompositionResult(
                    is_complex=True,
                    sub_queries=sub_queries,
                    reasoning_type="multi_part",
                )

        return DecompositionResult(
            is_complex=(target_query != query),
            sub_queries=[target_query],
            reasoning_type="atomic",
        )


query_decomposer = QueryDecomposer()
