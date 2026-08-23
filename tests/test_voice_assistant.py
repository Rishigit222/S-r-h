"""Tests for 3D Knowledge Graph AI Voice Assistant & Multi-Segment Task Decomposer."""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.graph.voice_assistant import graph_voice_assistant

client = TestClient(app)


def test_voice_assistant_deepseek_decomposition():
    resp = graph_voice_assistant.process_voice_query("Explain DeepSeek R1 reasoning with GRPO")
    assert resp.total_segments == 4
    assert len(resp.segments) == 4
    assert resp.intent == "deepseek_reasoning_architecture"

    # Segment 1: Conceptual
    assert "DeepSeek R1" in resp.segments[0].title or "GRPO" in resp.segments[0].title
    assert len(resp.segments[0].highlight_nodes) > 0

    # Segment 2: Graph Relational
    assert "DeepSeek_R1" in resp.segments[1].highlight_nodes or "GRPO" in resp.segments[1].highlight_nodes

    # Segment 3: Code Blueprint
    assert resp.segments[2].code_snippet is not None
    assert "vllm" in resp.segments[2].code_snippet.lower() or "deepseek" in resp.segments[2].code_snippet.lower()

    # Segment 4: Developer Links
    assert len(resp.segments[3].links) > 0
    assert any("github.com" in l.url or "arxiv.org" in l.url or "docs" in l.url for l in resp.segments[3].links)


def test_voice_assistant_lora_decomposition():
    resp = graph_voice_assistant.process_voice_query("How to fine tune with LoRA?")
    assert resp.total_segments == 4
    assert resp.intent == "lora_peft_architecture"
    assert "LoRA" in resp.segments[0].highlight_nodes
    assert resp.segments[2].code_snippet is not None
    assert "peft" in resp.segments[2].code_snippet.lower()


def test_voice_assistant_transformer_decomposition():
    resp = graph_voice_assistant.process_voice_query("Explain RoPE and Multi-Head Attention in Transformers")
    assert resp.total_segments == 4
    assert resp.intent == "transformer_attention_architecture"
    assert "Transformer" in resp.segments[0].highlight_nodes
    assert resp.segments[2].code_snippet is not None
    assert "torch" in resp.segments[2].code_snippet.lower()


def test_voice_assistant_graphrag_decomposition():
    resp = graph_voice_assistant.process_voice_query("How does GraphRAG and hybrid search work?")
    assert resp.total_segments == 4
    assert resp.intent == "graphrag_hybrid_architecture"
    assert "GraphRAG" in resp.segments[0].highlight_nodes
    assert "reciprocal_rank_fusion" in resp.segments[2].code_snippet


def test_api_voice_assistant_endpoint():
    res = client.post("/graph/voice-assist", json={"query": "Explain DeepSeek R1 reasoning architecture and GRPO"})
    assert res.status_code == 200
    data = res.json()
    assert "segments" in data
    assert data["total_segments"] == 4
    assert len(data["segments"]) == 4
    assert data["segments"][0]["narration_speech"] is not None
    assert len(data["segments"][3]["links"]) > 0
