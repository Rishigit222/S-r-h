# AI & Machine Learning Foundations: Architectures, Mathematics & Optimization

## 1. The Transformer Architecture & Self-Attention Mechanics

Introduced by Vaswani et al. in the 2017 paper *"Attention Is All You Need"*, the Transformer replaced recurrent architectures (RNNs, LSTMs) with pure self-attention mechanisms, allowing massive parallelization across GPU clusters.

### Mathematical Formulation of Scaled Dot-Product Attention:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- $Q \in \mathbb{R}^{n \times d_k}$: Query matrix representing the tokens searching for information.
- $K \in \mathbb{R}^{m \times d_k}$: Key matrix representing the tokens indexing their relevance.
- $V \in \mathbb{R}^{m \times d_v}$: Value matrix containing the actual token content.
- $\sqrt{d_k}$: Scaling factor preventing dot products from growing excessively large in high dimensions, which would cause vanishing gradients in the softmax.

### Multi-Head Attention (MHA):
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$$
$$\text{where } \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.

---

## 2. Modern Positional Encodings & Normalization Layers

1. **Rotary Position Embedding (RoPE)**:
   - RoPE encodes relative position by multiplying the Query and Key representations with an orthogonal rotation matrix in complex space.
   - Enables extrapolation to sequence lengths beyond the training context window. Used in LLaMA, Mistral, Qwen, and DeepSeek.

2. **RMSNorm (Root Mean Square Layer Normalization)**:
   - Replaces standard LayerNorm by removing the mean-centering step:
     $$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma$$
   - Provides 10%–20% training speedup with zero loss in convergence stability.

3. **SwiGLU (Swish Gated Linear Unit)**:
   - Replaces standard ReLU/GELU in feed-forward networks:
     $$\text{SwiGLU}(x) = (\text{Swish}(xW) \odot xV)W_2$$
   - Delivers significantly higher parameter efficiency.

---

## 3. Parameter-Efficient Fine-Tuning (PEFT) & LoRA

### Low-Rank Adaptation (LoRA):
Instead of updating all billions of model weights $W_0 \in \mathbb{R}^{d \times k}$, LoRA decomposes the weight update $\Delta W$ into two low-rank matrices $A \in \mathbb{R}^{r \times k}$ and $B \in \mathbb{R}^{d \times r}$ with rank $r \ll \min(d, k)$:
$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r}(B \times A)$$

- **Advantages**:
  - Reduces trainable parameters by **99.9%** (e.g., training only 4M parameters instead of 7B).
  - VRAM footprint during training is drastically reduced since optimizer states (AdamW momentum and variance) are only stored for low-rank matrices.
  - Zero inference latency overhead: $B \times A$ can be merged back into $W_0$ at deployment time.

### QLoRA (Quantized Low-Rank Adaptation):
- Quantizes the base model weights to 4-bit NormalFloat (NF4) while maintaining 16-bit brain float (bfloat16) for the LoRA adapters.
- Uses Double Quantization and Paged Optimizers to prevent out-of-memory errors on consumer GPUs.

---

## 4. Reinforcement Learning Alignment: RLHF, DPO & GRPO

1. **RLHF (Reinforcement Learning from Human Feedback)**:
   - Trains a separate Reward Model on pairwise human preferences, then optimizes the policy using PPO (Proximal Policy Optimization).
2. **DPO (Direct Preference Optimization)**:
   - Mathematically eliminates the separate reward model and PPO reinforcement learning loop by directly optimizing the language model on preference pairs $(y_w, y_l)$ using implicit reward formulation.
3. **GRPO (Group Relative Policy Optimization)**:
   - Introduced by DeepSeek in DeepSeek-Math and DeepSeek-R1.
   - Samples a group of outputs for each prompt and optimizes relative performance within the group, completely removing the need for a separate critic model and cutting training compute requirements in half.
