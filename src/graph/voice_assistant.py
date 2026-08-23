"""3D Knowledge Graph AI Voice Assistant & Multi-Segment Task Decomposer.

Breaks user queries into 4 execution stages:
1. Conceptual Architecture & Math Logic
2. 3D Graph Entity-Relational Traversal (Node Illumination)
3. Step-by-Step Production Code Blueprint
4. Developer Resources, Official Links & Research Papers
"""

from dataclasses import dataclass, field, asdict
from pydantic import BaseModel, Field

from src.graph.knowledge_graph import knowledge_graph


@dataclass
class DeveloperLink:
    title: str
    url: str
    category: str  # "github" | "paper" | "docs" | "tutorial"
    description: str


@dataclass
class VoiceReasoningSegment:
    segment_id: int
    title: str
    narration_speech: str  # Spoken aloud by AI voice assistant
    detailed_text: str     # Rendered in right intelligence panel
    highlight_nodes: list[str] = field(default_factory=list)
    code_snippet: str | None = None
    links: list[DeveloperLink] = field(default_factory=list)


@dataclass
class VoiceAssistantResponse:
    query: str
    intent: str
    total_segments: int
    segments: list[VoiceReasoningSegment]
    overall_summary: str


class VoiceAssistRequest(BaseModel):
    query: str = Field(..., description="Spoken voice query or command transcript")
    session_id: str = Field(default="user_session_01", description="User session ID")


class GraphVoiceAssistant:
    """Decomposes queries into 4-step progressive voice narration segments with 3D node targets and developer links."""

    def process_voice_query(self, query: str) -> VoiceAssistantResponse:
        q_lower = query.lower()

        # 1. DeepSeek R1 / GRPO Reasoning Topic
        if "deepseek" in q_lower or "grpo" in q_lower or "reasoning" in q_lower:
            return self._build_deepseek_reasoning_pipeline(query)

        # 2. LoRA / Fine-Tuning / PEFT Topic
        elif "lora" in q_lower or "peft" in q_lower or "fine-tun" in q_lower:
            return self._build_lora_peft_pipeline(query)

        # 3. Transformer / Self-Attention Topic
        elif "transformer" in q_lower or "attention" in q_lower or "rope" in q_lower or "rmsnorm" in q_lower:
            return self._build_transformer_pipeline(query)

        # 4. GraphRAG / Hybrid RAG / Self-Healing Architecture
        elif "rag" in q_lower or "graph" in q_lower or "hybrid" in q_lower or "retriev" in q_lower:
            return self._build_graphrag_pipeline(query)

        # 5. Generic / System Design & Fast Serving
        else:
            return self._build_system_design_pipeline(query)

    def _build_deepseek_reasoning_pipeline(self, query: str) -> VoiceAssistantResponse:
        segments = [
            VoiceReasoningSegment(
                segment_id=1,
                title="Conceptual Architecture & GRPO Reinforcement Learning",
                narration_speech="DeepSeek R1 pioneers open-weight reasoning without a separate critic model, using Group Relative Policy Optimization to reinforce spontaneous self-verification.",
                detailed_text="DeepSeek R1 eliminates the memory-heavy Critic model in traditional PPO. It samples a group of candidate reasoning paths and optimizes policy based on relative group rewards with verifiable rule-based outcomes.",
                highlight_nodes=["DeepSeek_R1", "GRPO", "Transformer"],
                links=[
                    DeveloperLink("DeepSeek-R1 Official Research Paper", "https://arxiv.org/abs/2501.12948", "paper", "Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"),
                    DeveloperLink("DeepSeek-R1 GitHub Repository", "https://github.com/deepseek-ai/DeepSeek-R1", "github", "Official open-weights and replication codebase"),
                ],
            ),
            VoiceReasoningSegment(
                segment_id=2,
                title="3D Knowledge Graph Relational Traversal",
                narration_speech="In our 3D Knowledge Graph, notice how DeepSeek R1 connects directly to GRPO Reinforcement Learning and Multi-Head Latent Attention.",
                detailed_text="The 3D physics graph traverses 1-hop and 2-hop entity triples: (DeepSeek_R1) -> [TRAINED_WITH] -> (GRPO) and (DeepSeek_V3) -> [USES] -> (Multi-Head Latent Attention MLA).",
                highlight_nodes=["DeepSeek_R1", "DeepSeek_V3", "GRPO", "Self-Attention"],
            ),
            VoiceReasoningSegment(
                segment_id=3,
                title="Step-by-Step Production Code Blueprint",
                narration_speech="Here is the production implementation blueprint using vLLM and Hugging Face Transformers for running DeepSeek R1 reasoning pipelines.",
                detailed_text="Run DeepSeek R1 with speculative decoding and KV cache paging using vLLM:",
                highlight_nodes=["PagedAttention", "DeepSeek_R1"],
                code_snippet="""# Production DeepSeek-R1 Reasoning Pipeline
from vllm import LLM, SamplingParams

llm = LLM(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.90,
    max_model_len=16384,
)

sampling_params = SamplingParams(
    temperature=0.6,
    top_p=0.95,
    max_tokens=4096,
)

outputs = llm.generate(["Solve step by step: Prove sqrt(2) is irrational."], sampling_params)
print(outputs[0].outputs[0].text)""",
            ),
            VoiceReasoningSegment(
                segment_id=4,
                title="Developer Resources & Where to Build",
                narration_speech="You can build and deploy this using Hugging Face Transformers, vLLM for high-throughput serving, and Ollama for local CPU execution.",
                detailed_text="Curated official documentation and deployment guides:",
                highlight_nodes=["DeepSeek_R1", "vLLM", "Python"],
                links=[
                    DeveloperLink("vLLM High-Throughput Serving Engine", "https://github.com/vllm-project/vllm", "github", "PagedAttention and fast distributed LLM inference"),
                    DeveloperLink("Hugging Face DeepSeek Hub", "https://huggingface.co/deepseek-ai", "docs", "Model weights, tokenizers, and GGUF quants"),
                    DeveloperLink("Ollama Local CPU Runner", "https://ollama.com/library/deepseek-r1", "tutorial", "Run DeepSeek-R1 locally with 1-click install"),
                ],
            ),
        ]
        return VoiceAssistantResponse(
            query=query,
            intent="deepseek_reasoning_architecture",
            total_segments=len(segments),
            segments=segments,
            overall_summary="DeepSeek R1 reasoning architecture, GRPO reinforcement learning, and production deployment blueprint.",
        )

    def _build_lora_peft_pipeline(self, query: str) -> VoiceAssistantResponse:
        segments = [
            VoiceReasoningSegment(
                segment_id=1,
                title="LoRA Mathematical Foundation & Weight Decomposition",
                narration_speech="LoRA freezes the pre-trained weight matrix W-zero and decomposes the weight update delta-W into two low-rank matrices B and A.",
                detailed_text="LoRA formula: W = W_0 + (alpha / r) * (B @ A), where A is Gaussian initialized and B is zero initialized, reducing trainable parameters by over 99%.",
                highlight_nodes=["LoRA", "TrainableParameters", "Transformer"],
                links=[
                    DeveloperLink("LoRA: Low-Rank Adaptation of Large Language Models", "https://arxiv.org/abs/2106.09685", "paper", "Original Microsoft Research LoRA paper"),
                    DeveloperLink("Hugging Face PEFT Library", "https://github.com/huggingface/peft", "github", "State-of-the-art Parameter-Efficient Fine-Tuning methods"),
                ],
            ),
            VoiceReasoningSegment(
                segment_id=2,
                title="3D Knowledge Graph Entity Connections",
                narration_speech="Observe in the graph how LoRA connects to Trainable Parameters, QLoRA 4-bit NormalFloat quantization, and Transformer Multi-Head Attention.",
                detailed_text="Relational path: (LoRA) -> [REDUCES] -> (TrainableParameters) and (QLoRA) -> [QUANTIZES_TO] -> (NF4 4-bit Base Weights).",
                highlight_nodes=["LoRA", "Transformer", "Self-Attention"],
            ),
            VoiceReasoningSegment(
                segment_id=3,
                title="Production LoRA Fine-Tuning Blueprint",
                narration_speech="Here is the PyTorch and Hugging Face PEFT code to attach LoRA adapters to any attention layer in 5 lines.",
                detailed_text="Apply LoRA to Linear Attention Projection layers:",
                highlight_nodes=["LoRA", "Python"],
                code_snippet="""# Attach LoRA to Transformer Attention Layers
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()""",
            ),
            VoiceReasoningSegment(
                segment_id=4,
                title="Developer Resources & Implementation Links",
                narration_speech="Everything you need to train LoRA models with Hugging Face PEFT, Unsloth, and PyTorch is linked below.",
                detailed_text="Official documentation and high-speed training repos:",
                highlight_nodes=["LoRA", "Python"],
                links=[
                    DeveloperLink("Unsloth 5x Faster LoRA Training", "https://github.com/unslothai/unsloth", "github", "Memory-optimized manual backprop LoRA engine"),
                    DeveloperLink("Hugging Face SFT Trainer Docs", "https://huggingface.co/docs/trl/sft_trainer", "docs", "Supervised fine-tuning guide"),
                ],
            ),
        ]
        return VoiceAssistantResponse(
            query=query,
            intent="lora_peft_architecture",
            total_segments=len(segments),
            segments=segments,
            overall_summary="LoRA parameter-efficient fine tuning mathematical foundation, graph path, and production PEFT code.",
        )

    def _build_transformer_pipeline(self, query: str) -> VoiceAssistantResponse:
        segments = [
            VoiceReasoningSegment(
                segment_id=1,
                title="Transformer Attention Math & Rotary Embeddings",
                narration_speech="The core of modern LLMs is Multi-Head Scaled Dot-Product Attention combined with RoPE rotary positional embeddings and RMSNorm normalization.",
                detailed_text="Scaled Dot-Product Attention: Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V. RoPE applies complex plane 2D rotation matrices to query and key vectors for relative distance encoding.",
                highlight_nodes=["Transformer", "Self-Attention", "RoPE", "RMSNorm"],
                links=[
                    DeveloperLink("Attention Is All You Need", "https://arxiv.org/abs/1706.03762", "paper", "The foundational Transformer paper by Vaswani et al."),
                    DeveloperLink("RoFormer: Enhanced Transformer with Rotary Position Embedding", "https://arxiv.org/abs/2104.09864", "paper", "Official RoPE mathematical formulation"),
                ],
            ),
            VoiceReasoningSegment(
                segment_id=2,
                title="3D Graph Traversal: Transformer to Attention & RoPE",
                narration_speech="Look at the 3D graph: Transformer links directly to Self-Attention, RMSNorm for mean-centering-free normalization, and SwiGLU activation.",
                detailed_text="Triples traversed: (Transformer) -> [USES] -> (Self-Attention) -> [INTEGRATES] -> (RoPE).",
                highlight_nodes=["Transformer", "Self-Attention", "RMSNorm", "RoPE"],
            ),
            VoiceReasoningSegment(
                segment_id=3,
                title="Production PyTorch Self-Attention Implementation",
                narration_speech="Here is the complete PyTorch implementation of Multi-Head Scaled Dot-Product Attention with FlashAttention scaling.",
                detailed_text="Multi-Head Attention PyTorch Module:",
                highlight_nodes=["Transformer", "Python"],
                code_snippet="""# PyTorch Scaled Dot-Product Multi-Head Attention
import torch
import torch.nn as nn
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model=4096, n_heads=32):
        super().__init__()
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)""",
            ),
            VoiceReasoningSegment(
                segment_id=4,
                title="Developer Resources & Official Repositories",
                narration_speech="You can inspect the complete open source implementations in Hugging Face Transformers and PyTorch Core linked below.",
                detailed_text="Official repositories and learning documentation:",
                highlight_nodes=["Transformer", "Python"],
                links=[
                    DeveloperLink("Hugging Face Transformers GitHub", "https://github.com/huggingface/transformers", "github", "State-of-the-art ML models in PyTorch & JAX"),
                    DeveloperLink("PyTorch Native Scaled Dot Product Attention", "https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html", "docs", "FlashAttention C++ kernel in PyTorch"),
                ],
            ),
        ]
        return VoiceAssistantResponse(
            query=query,
            intent="transformer_attention_architecture",
            total_segments=len(segments),
            segments=segments,
            overall_summary="Transformer multi-head attention, RoPE rotary position embeddings, and PyTorch implementation.",
        )

    def _build_graphrag_pipeline(self, query: str) -> VoiceAssistantResponse:
        segments = [
            VoiceReasoningSegment(
                segment_id=1,
                title="GraphRAG & Hybrid Search Architecture",
                narration_speech="GraphRAG combines vector semantic retrieval with entity-relation knowledge graphs, eliminating the 73 percent retrieval failure of naive RAG.",
                detailed_text="Hybrid Dual-Path Search merges dense ChromaDB vector embeddings with sparse BM25Okapi keyword scores via Reciprocal Rank Fusion, followed by Cross-Encoder reranking.",
                highlight_nodes=["GraphRAG", "Transformer", "Self-Attention"],
                links=[
                    DeveloperLink("Microsoft GraphRAG Research Paper", "https://arxiv.org/abs/2404.16130", "paper", "From Local to Global: A Graph RAG Approach to Query-Focused Summarization"),
                    DeveloperLink("Microsoft GraphRAG GitHub", "https://github.com/microsoft/graphrag", "github", "Official modular graph retrieval pipeline"),
                ],
            ),
            VoiceReasoningSegment(
                segment_id=2,
                title="3D Knowledge Graph Entity Traversal",
                narration_speech="In our 3D graph, notice how entity triples interconnect concepts across Transformers, DeepSeek, LoRA, and Serving Systems.",
                detailed_text="1-hop and 2-hop subgraph expansion extracts relational context to ground LLM generation and prevent hallucinations.",
                highlight_nodes=["GraphRAG", "DeepSeek_R1", "LoRA", "PagedAttention"],
            ),
            VoiceReasoningSegment(
                segment_id=3,
                title="Step-by-Step Self-Healing RAG Code Blueprint",
                narration_speech="Here is the complete Python implementation of Dual-Path Hybrid Search with Reciprocal Rank Fusion.",
                detailed_text="Reciprocal Rank Fusion (RRF) algorithm:",
                highlight_nodes=["GraphRAG", "Python"],
                code_snippet="""# Dual-Path Hybrid Search with Reciprocal Rank Fusion (RRF)
def reciprocal_rank_fusion(vector_results, bm25_results, k=60, vector_weight=0.5, bm25_weight=0.5):
    scores = {}
    for rank, doc in enumerate(vector_results):
        scores[doc["chunk_id"]] = scores.get(doc["chunk_id"], 0.0) + (vector_weight / (k + rank + 1))
    for rank, doc in enumerate(bm25_results):
        scores[doc["chunk_id"]] = scores.get(doc["chunk_id"], 0.0) + (bm25_weight / (k + rank + 1))
    
    sorted_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_chunks""",
            ),
            VoiceReasoningSegment(
                segment_id=4,
                title="Developer Resources & Where to Build",
                narration_speech="You can build production GraphRAG and hybrid search using ChromaDB, rank-bm25, and Sentence-Transformers.",
                detailed_text="Curated official libraries and implementation links:",
                highlight_nodes=["GraphRAG", "Python"],
                links=[
                    DeveloperLink("ChromaDB Vector Database", "https://github.com/chroma-core/chroma", "github", "Open-source embedding database for AI applications"),
                    DeveloperLink("Sentence-Transformers Cross-Encoder", "https://www.sbert.net/docs/pretrained_cross-encoders.html", "docs", "Pre-trained MS-MARCO Cross-Encoder rerankers"),
                ],
            ),
        ]
        return VoiceAssistantResponse(
            query=query,
            intent="graphrag_hybrid_architecture",
            total_segments=len(segments),
            segments=segments,
            overall_summary="GraphRAG entity traversal, hybrid Reciprocal Rank Fusion search, and self-healing guardrail loops.",
        )

    def _build_system_design_pipeline(self, query: str) -> VoiceAssistantResponse:
        segments = [
            VoiceReasoningSegment(
                segment_id=1,
                title="High-Throughput LLM Serving & PagedAttention",
                narration_speech="High-throughput LLM serving solves GPU memory fragmentation using PagedAttention, which allocates Key-Value cache memory in virtual pages like an operating system.",
                detailed_text="PagedAttention eliminates up to 96% of wasted KV-cache memory, enabling near-optimal batching and 2x to 4x throughput improvements.",
                highlight_nodes=["PagedAttention", "Transformer"],
                links=[
                    DeveloperLink("vLLM: Efficient Memory Management for LLM Serving with PagedAttention", "https://arxiv.org/abs/2309.06180", "paper", "UC Berkeley PagedAttention paper"),
                    DeveloperLink("vLLM Project GitHub", "https://github.com/vllm-project/vllm", "github", "High-throughput and low-latency LLM serving engine"),
                ],
            ),
            VoiceReasoningSegment(
                segment_id=2,
                title="3D Knowledge Graph Connections",
                narration_speech="Notice in the 3D graph how PagedAttention connects to Transformers, Speculative Decoding, and Distributed Inference.",
                detailed_text="Triples traversed: (PagedAttention) -> [MANAGES] -> (KVCache) and (PagedAttention) -> [POWERS] -> (vLLM Serving).",
                highlight_nodes=["PagedAttention", "Transformer"],
            ),
            VoiceReasoningSegment(
                segment_id=3,
                title="Production Serving Blueprint",
                narration_speech="Here is the asynchronous FastAPI serving blueprint for low-latency speculative decoding.",
                detailed_text="FastAPI asynchronous serving setup:",
                highlight_nodes=["PagedAttention", "Python"],
                code_snippet="""# Async High-Throughput Serving Engine
from fastapi import FastAPI
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs

app = FastAPI()
engine_args = AsyncEngineArgs(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
engine = AsyncLLMEngine.from_engine_args(engine_args)

@app.post("/v1/chat")
async def chat(prompt: str):
    results_generator = engine.generate(prompt, sampling_params, request_id="req_01")
    final_output = None
    async for request_output in results_generator:
        final_output = request_output
    return {"text": final_output.outputs[0].text}""",
            ),
            VoiceReasoningSegment(
                segment_id=4,
                title="Developer Resources & Official Repositories",
                narration_speech="Explore the official vLLM, TensorRT-LLM, and TGI repositories linked below to build production LLM serving clusters.",
                detailed_text="Official documentation and links:",
                highlight_nodes=["PagedAttention", "Python"],
                links=[
                    DeveloperLink("vLLM Documentation", "https://docs.vllm.ai", "docs", "Official vLLM deployment and benchmark guide"),
                    DeveloperLink("NVIDIA TensorRT-LLM", "https://github.com/NVIDIA/TensorRT-LLM", "github", "Optimized LLM inference engine for NVIDIA GPUs"),
                ],
            ),
        ]
        return VoiceAssistantResponse(
            query=query,
            intent="system_design_serving",
            total_segments=len(segments),
            segments=segments,
            overall_summary="High-throughput LLM serving, PagedAttention memory paging, and asynchronous production deployment.",
        )


graph_voice_assistant = GraphVoiceAssistant()
