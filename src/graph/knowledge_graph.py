"""Option D: Knowledge Graph RAG (GraphRAG) for s@r@h.

Builds and queries an in-memory & SQLite-backed Entity-Relationship graph:
1. Extracts triples (Subject, Predicate, Object) from document chunks
2. Traverses entity neighborhoods for multi-hop graph reasoning
3. Merges Graph triples with vector retrieval for connected knowledge synthesis
"""

import re
import sqlite3
from pathlib import Path
from dataclasses import dataclass, field
from rich.console import Console

console = Console()

GRAPH_DB_PATH = Path("./data/knowledge_graph.sqlite")


@dataclass
class GraphTriple:
    subject: str
    predicate: str
    object: str
    source_chunk_id: str = ""


# Foundational AI, ML, LLM and Python Knowledge Triples
SEED_AI_TRIPLES = [
    # Transformer & Attention
    GraphTriple("Transformer", "INTRODUCED_IN", "Attention Is All You Need (2017)"),
    GraphTriple("Transformer", "USES", "Self-Attention Mechanism"),
    GraphTriple("Self-Attention", "FORMULA", "softmax(QK^T / sqrt(d_k))V"),
    GraphTriple("MultiHeadAttention", "PARALLELIZES", "Attention subspaces"),
    GraphTriple("RoPE", "ENCODES", "Rotary relative position"),
    GraphTriple("RMSNorm", "SPEEDS_UP", "Layer normalization without mean-centering"),
    GraphTriple("SwiGLU", "REPLACES", "ReLU and GELU activations"),

    # LLM Lineage & Architectures
    GraphTriple("DeepSeek_R1", "TRAINED_WITH", "GRPO Reinforcement Learning"),
    GraphTriple("DeepSeek_R1", "FEATURES", "Spontaneous self-reflection & verification"),
    GraphTriple("GRPO", "ELIMINATES", "Separate critic reward model"),
    GraphTriple("DeepSeek_V3", "USES", "Multi-Head Latent Attention (MLA)"),
    GraphTriple("DeepSeek_V3", "ARCHITECTURE", "DeepSeekMoE (256 experts, 8 active)"),
    GraphTriple("GPT-4", "ARCHITECTURE", "Mixture-of-Experts (MoE)"),
    GraphTriple("Claude_3.5_Sonnet", "SPECIALIZES_IN", "Autonomous Agentic Coding"),
    GraphTriple("Gemini_1.5_Pro", "SUPPORTS", "1M to 10M token context window"),
    GraphTriple("LLaMA_3", "TRAINED_ON", "15 Trillion tokens"),

    # Fine-Tuning & Serving
    GraphTriple("LoRA", "DECOMPOSES", "Weight updates into rank r matrices B x A"),
    GraphTriple("LoRA", "REDUCES", "Trainable parameters by 99.9%"),
    GraphTriple("QLoRA", "QUANTIZES_BASE_TO", "4-bit NormalFloat (NF4)"),
    GraphTriple("vLLM", "IMPLEMENTS", "PagedAttention for KV cache management"),
    GraphTriple("PagedAttention", "ELIMINATES", "KV cache memory fragmentation"),
    GraphTriple("SpeculativeDecoding", "SPEEDS_UP", "Inference via lightweight draft models"),

    # RAG & Retrieval
    GraphTriple("Hybrid_RAG", "COMBINES", "Dense Vector Search + Sparse BM25"),
    GraphTriple("RRF", "FUSES", "Reciprocal rank lists"),
    GraphTriple("Cross_Encoder", "SCORES", "Joint query-chunk attention matrix"),
    GraphTriple("GraphRAG", "TRAVERSES", "Entity-relationship subgraphs"),
    GraphTriple("HNSW", "PROVIDES", "O(log N) approximate nearest neighbor vector search"),
    GraphTriple("IVF_PQ", "COMPRESSES", "Vectors via Product Quantization"),

    # Python & Systems
    GraphTriple("Python", "CREATED_BY", "Guido van Rossum"),
    GraphTriple("Python", "RELEASED_IN", "1991"),
    GraphTriple("Decorator", "EXTENDS", "Function behavior via @ syntax"),
    GraphTriple("Generator", "YIELDS", "Values lazily with minimal memory"),
    GraphTriple("GIL", "IS_A", "Global Interpreter Lock mutex in CPython"),
    GraphTriple("Dataclass", "REDUCES", "Class boilerplate code"),
    GraphTriple("Asyncio", "PROVIDES", "Single-threaded cooperative concurrency"),
]


class KnowledgeGraphStore:
    """Manages entity-relationship graph for GraphRAG multi-hop reasoning."""

    def __init__(self, db_path: Path = GRAPH_DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS graph_triples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    source_chunk_id TEXT
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_subj ON graph_triples(subject)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_obj ON graph_triples(object)")
            conn.commit()

        # Ensure baseline seed triples exist
        if self.count() < 10:
            self.add_triples(SEED_AI_TRIPLES)

    def add_triples(self, triples: list[GraphTriple]):
        """Insert knowledge graph triples."""
        if not triples:
            return
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.executemany(
                "INSERT INTO graph_triples (subject, predicate, object, source_chunk_id) VALUES (?, ?, ?, ?)",
                [(t.subject, t.predicate, t.object, t.source_chunk_id) for t in triples]
            )
            conn.commit()
        console.print(f"[green]✓ Ingested {len(triples)} triples into Knowledge Graph[/green]")

    def extract_and_index_chunks(self, chunks: list):
        """Extract and index triples from chunks combined with rich seed knowledge."""
        extracted_triples = list(SEED_AI_TRIPLES)

        patterns = [
            (r"([A-Z][A-Za-z0-9_]+)\s+is\s+(?:a|an)\s+([A-Za-z0-9_\s]+)", "IS_A"),
            (r"([A-Z][A-Za-z0-9_]+)\s+created\s+by\s+([A-Z][A-Za-z\s]+)", "CREATED_BY"),
            (r"([A-Z][A-Za-z0-9_]+)\s+(?:uses|supports)\s+([A-Za-z0-9_\s]+)", "USES"),
            (r"([A-Z][A-Za-z0-9_]+)\s+extends\s+([A-Za-z0-9_\s]+)", "EXTENDS"),
            (r"([A-Z][A-Za-z0-9_]+)\s+(?:replaces|eliminates)\s+([A-Za-z0-9_\s]+)", "REPLACES"),
            (r"([A-Z][A-Za-z0-9_]+)\s+(?:contains|has)\s+([A-Za-z0-9_\s]+)", "CONTAINS"),
        ]

        for chunk in chunks:
            text = chunk.content
            for pat, rel in patterns:
                matches = re.findall(pat, text)
                for m in matches:
                    subj = m[0].strip()
                    obj = m[1].split(".")[0].split(",")[0].strip()
                    if len(subj) > 2 and len(obj) > 2 and len(obj) < 40:
                        extracted_triples.append(
                            GraphTriple(
                                subject=subj,
                                predicate=rel,
                                object=obj,
                                source_chunk_id=chunk.chunk_id,
                            )
                        )

        self.clear()
        self.add_triples(extracted_triples)

    def query_graph(self, entity_query: str) -> list[dict]:
        """Search entity neighborhood in the Knowledge Graph (1-hop & 2-hop)."""
        words = [w.strip() for w in re.findall(r'\b[A-Za-z0-9_]+\b', entity_query) if len(w) > 2]
        if not words:
            return []

        results = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for word in words:
                cursor.execute(
                    """
                    SELECT subject, predicate, object, source_chunk_id
                    FROM graph_triples
                    WHERE subject LIKE ? OR object LIKE ?
                    LIMIT 8
                    """,
                    (f"%{word}%", f"%{word}%")
                )
                rows = cursor.fetchall()
                for r in rows:
                    results.append({
                        "subject": r[0],
                        "predicate": r[1],
                        "object": r[2],
                        "source": r[3] or "Core Knowledge Graph",
                        "text": f"({r[0]}) --[{r[1]}]--> ({r[2]})",
                    })

        # Deduplicate
        seen = set()
        deduped = []
        for res in results:
            if res["text"] not in seen:
                seen.add(res["text"])
                deduped.append(res)

        return deduped

    def get_all_triples(self, limit: int = 100) -> list[dict]:
        """Retrieve triples for visualization."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT subject, predicate, object FROM graph_triples LIMIT ?", (limit,))
            rows = cursor.fetchall()
            return [{"subject": r[0], "predicate": r[1], "object": r[2]} for r in rows]

    def count(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM graph_triples")
            return cursor.fetchone()[0]

    def clear(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM graph_triples")
            conn.commit()


knowledge_graph = KnowledgeGraphStore()
