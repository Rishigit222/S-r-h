# Machine Learning System Design: Serving, Vector Indexing & Algorithms

## 1. Vector Search Indexing: HNSW vs. IVF-PQ

Fast nearest neighbor search over millions of dense vectors requires approximate nearest neighbor (ANN) graph algorithms:

### Hierarchical Navigable Small World (HNSW):
- Constructs a multi-layer graph where upper layers contain sparse long-range highway links and lower layers contain dense local clustering.
- **Search Complexity**: $\mathcal{O}(\log N)$ logarithmic time traversal with high recall (>98%).
- **Parameters**:
  - `M`: Maximum number of bidirectional links per node in layers $>0$.
  - `efConstruction`: Size of dynamic candidate list during graph construction.
  - `efSearch`: Size of dynamic candidate list during query time.

### Inverted File with Product Quantization (IVF-PQ):
- Partitions vector space into Voronoi cells via k-means clustering (Inverted File).
- Decomposes high-dimensional vectors into $m$ low-dimensional sub-vectors, quantizing each to a centroid index (Product Quantization).
- Reduces memory consumption by 90% at the cost of slight recall reduction.

---

## 2. High-Throughput LLM Inference: vLLM, PagedAttention & KV Caching

### The KV Cache Memory Bottleneck:
During autoregressive token generation, the Key ($K$) and Value ($V$) tensors for all past tokens must be preserved to compute self-attention for subsequent tokens. Memory usage scales linearly with batch size and context length:
$$\text{KV Cache Size} = 2 \times 2 \times n_{\text{layers}} \times n_{\text{heads}} \times d_{\text{head}} \times L_{\text{seq}} \times \text{batch\_size} \times \text{precision (bytes)}$$

### PagedAttention (vLLM):
- Inspired by virtual memory paging in operating systems.
- Allocates KV cache memory in fixed-size blocks (pages) rather than contiguous memory chunks.
- Completely eliminates internal and external memory fragmentation, allowing dynamic batch sizes to increase by **2x–4x** and boosting serving throughput.

---

## 3. Speculative Decoding & Medusa Multi-Head Verification

- **Speculative Decoding**: Employs a lightweight draft model (e.g. 1B model) to generate $K$ candidate tokens in parallel, which are then verified in a single forward pass of the larger target model (e.g. 70B model).
- Provides a **2x–3x** end-to-end latency speedup without changing the mathematical output distribution.
