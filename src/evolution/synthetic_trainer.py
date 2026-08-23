"""Self-Supervised Synthetic QA Generator and Continuous Trainer for s@r@h.

Mines document chunks to automatically generate synthetic question-answer pairs
and benchmarks retrieval & guardrail accuracy without requiring human-labeled data.
"""

import re
import random
from dataclasses import dataclass
from rich.console import Console

from src.ingestion.loader import load_directory
from src.ingestion.chunker import chunk_documents
from src.config import settings

console = Console()


@dataclass
class SyntheticQAPair:
    question: str
    ground_truth_context: str
    target_entity: str
    difficulty: str  # "easy" | "medium" | "hard"


class SyntheticTrainer:
    """Generates synthetic QA pairs from raw document chunks for continuous self-training."""

    def generate_synthetic_dataset(self, max_pairs: int = 12) -> list[SyntheticQAPair]:
        """Extract key concepts from knowledge documents and construct QA pairs."""
        docs = load_directory(settings.documents_dir)
        if not docs:
            return []

        chunks = chunk_documents(docs, chunk_size=400, chunk_overlap=40)
        qa_pairs = []

        patterns = [
            # Pattern: X is Y -> What is X?
            (r"([A-Z][A-Za-z0-9_\s]{2,25})\s+is\s+(?:a|an)\s+([^.\n]+)", "What is {entity}?", "easy"),
            # Pattern: X extends Y -> How does X extend Y?
            (r"([A-Z][A-Za-z0-9_]+)\s+extends\s+([^.\n]+)", "How does {entity} extend functionality?", "medium"),
            # Pattern: X uses Y -> What does X use?
            (r"([A-Z][A-Za-z0-9_]+)\s+uses\s+([^.\n]+)", "What mechanism does {entity} use?", "medium"),
            # Pattern: X reduces Y -> How does X reduce Y?
            (r"([A-Z][A-Za-z0-9_]+)\s+reduces\s+([^.\n]+)", "How does {entity} optimize performance or reduce overhead?", "hard"),
        ]

        for chunk in chunks:
            text = chunk.content
            for pat, q_template, diff in patterns:
                matches = re.findall(pat, text)
                for m in matches:
                    entity = m[0].strip()
                    if len(entity) > 2 and len(entity) < 30 and not entity.startswith("The "):
                        q_text = q_template.format(entity=entity)
                        qa_pairs.append(
                            SyntheticQAPair(
                                question=q_text,
                                ground_truth_context=text[:300],
                                target_entity=entity,
                                difficulty=diff,
                            )
                        )
                        if len(qa_pairs) >= max_pairs:
                            break
            if len(qa_pairs) >= max_pairs:
                break

        # Deduplicate by question
        seen = set()
        deduped = []
        for pair in qa_pairs:
            if pair.question not in seen:
                seen.add(pair.question)
                deduped.append(pair)

        console.print(f"[green]✓ Generated {len(deduped)} synthetic QA pairs for autonomous training[/green]")
        return deduped


synthetic_trainer = SyntheticTrainer()
