# Advanced RAG Architectures: Hybrid Search, GraphRAG & Self-Healing Guardrails

## 1. Why Naive Vector RAG Fails in Production

Research across enterprise deployments indicates that **~73% of RAG failures occur in the retrieval stage**, caused by:
1. **Semantic Mismatch**: Cosine similarity matches broad topical relevance rather than direct factual answers.
2. **Keyword & Identifier Blindness**: Vector models struggle with specific alphanumeric strings, error codes, and function names (e.g. `lru_cache`, `ZeroDivisionError`).
3. **Lossy Bi-Encoder Compression**: Encoding an entire 500-token chunk into a single 384- or 1536-dimensional vector loses fine-grained relational nuance.

---

## 2. The 3-Stage Hybrid Retrieval & Reranking Paradigm

### Stage 1: Dual-Path Retrieval
- **Dense Vector Search**: ChromaDB / Qdrant with cosine metric for semantic concept matching.
- **Sparse Keyword Search**: Okapi BM25 statistical term weighting ($k_1=1.5, b=0.75$) for exact token matching.

### Stage 2: Reciprocal Rank Fusion (RRF)
Combines candidate lists without needing arbitrary score calibration:
$$\text{RRF Score}(d) = \sum_{m \in M} \frac{w_m}{k + \text{rank}_m(d)}$$
Where $k \approx 60$ is a smoothing constant and $w_m$ is the retriever weight.

### Stage 3: Cross-Encoder Precision Reranking
A Cross-Encoder (e.g., `cross-encoder/ms-marco-MiniLM-L6-v2`) jointly attends to the `[CLS] Query [SEP] Chunk [SEP]` sequence with all-to-all attention across all token layers. This produces calibrated relevance logits where $>0.3$ represents high answering confidence.

---

## 3. GraphRAG: Entity-Relationship Knowledge Graphs

While standard RAG retrieves isolated document chunks, **GraphRAG** builds a connected graph of entity nodes $(E_1, E_2)$ and directed relational edges $(R)$:
$$(E_{\text{Subject}}) \xrightarrow{R_{\text{Predicate}}} (E_{\text{Object}})$$

### Graph Traversal for Multi-Hop Reasoning:
1. **Entity Extraction**: Identifies key entities in the query (e.g., *Transformer*, *LoRA*, *DeepSeek*).
2. **Neighborhood Subgraph Traversal**: Performs 1-hop and 2-hop graph expansions to connect facts across disparate documents.
3. **Context Fusion**: Injects explicit relational triples alongside hybrid text chunks into the LLM context prompt.

---

## 4. Tri-Guardrail Self-Healing Loop

1. **Pre-Generation Gate**: Filters candidate chunks whose Cross-Encoder rerank score falls below the confidence threshold ($< 0.3$), preventing ungrounded hallucination before the LLM is invoked.
2. **Post-Generation NLI Faithfulness**: Uses a Natural Language Inference (NLI) model to check if the generated answer is strictly entailed by the context premise ($P(\text{Entailment}) \ge 0.70$).
3. **Citation Verifier**: Validates that every citation marker `[1]`, `[2]` maps to a valid retrieved chunk and supports the claim.
4. **Self-Healing Loop**: If verification fails, s@r@h rewrites the query, expands the search neighborhood, and retries up to 2 times before gracefully refusing.
