"""Streamlit frontend for s@r@h — Self-Adaptive Reasoning & Retrieval Autonomous Host."""

import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="s@r@h — Autonomous Knowledge Host",
    page_icon="🌟",
    layout="wide",
)

st.title("🌟 s@r@h")
st.caption("**Self-Adaptive Reasoning & Retrieval Autonomous Host** — Hybrid RAG + GraphRAG + Memory Agent + Real-Time Telemetry")

if "session_id" not in st.session_state:
    st.session_state.session_id = "user_session_01"

# Sidebar
with st.sidebar:
    st.header("⚡ System Control")
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success(f"s@r@h Engine: {health['status'].upper()}")
        st.metric("Vector Chunks", health.get("vector_store_count", 0))
        st.metric("Graph Triples", health.get("graph_triples_count", 0))
        st.metric("Active LLM Backend", health.get("llm_provider", "local_fallback"))

        c_stats = health.get("cache_stats", {})
        st.divider()
        st.subheader("⚡ Semantic Cache")
        st.metric("Hit Rate", f"{c_stats.get('hit_rate_pct', 0)}%")
        st.caption(f"Hits: {c_stats.get('hits', 0)} | Misses: {c_stats.get('misses', 0)}")
    except Exception:
        st.error("s@r@h backend offline. Run: make run-api")

    st.divider()
    st.header("📥 Knowledge Ingestion")
    if st.button("Ingest Knowledge Base", use_container_width=True):
        with st.spinner("Indexing vector embeddings and building Knowledge Graph..."):
            try:
                resp = requests.post(f"{API_URL}/ingest", timeout=120).json()
                st.success(
                    f"[OK] Loaded {resp['documents_loaded']} docs → "
                    f"{resp['chunks_created']} chunks | {resp['graph_triples_indexed']} graph triples"
                )
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

# Main Tabs
tab_chat, tab_graph, tab_telemetry, tab_memory = st.tabs([
    "💬 Chat with s@r@h",
    "🕸️ Knowledge Graph (GraphRAG)",
    "📊 Real-Time Quality & Drift",
    "🧠 Memory Agent & Profile",
])

# --- TAB 1: Chat ---
with tab_chat:
    query = st.text_input(
        "Ask s@r@h any question:",
        placeholder="e.g., Compare Python generators and list comprehensions, or What is a decorator?",
    )

    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        enable_web = st.checkbox("🌐 Enable Dynamic Web Grounding Fallback", value=True)
    with col_opt2:
        enable_graph = st.checkbox("🕸️ Enable GraphRAG Relational Context", value=True)

    if query:
        with st.spinner("s@r@h is processing (Security → Cache → GraphRAG → Hybrid Search → Guardrails)..."):
            try:
                resp = requests.post(
                    f"{API_URL}/query",
                    json={
                        "question": query,
                        "session_id": st.session_state.session_id,
                        "top_k": 5,
                        "enable_web_fallback": enable_web,
                        "enable_graph_rag": enable_graph,
                    },
                    timeout=120,
                ).json()
            except Exception as e:
                st.error(f"Query failed: {e}")
                st.stop()

        # Badges
        badge_cols = st.columns(5)
        with badge_cols[0]:
            if resp.get("cache_hit"):
                st.success("⚡ Cache: HIT (<5ms)")
            else:
                st.info("⚡ Cache: MISS")
        with badge_cols[1]:
            if resp.get("multi_hop_used"):
                st.info("🧠 Multi-Hop: ACTIVE")
            else:
                st.caption("🧠 Single-Hop")
        with badge_cols[2]:
            if resp.get("web_grounded"):
                st.warning("🌐 Web Grounded")
            else:
                st.success("📁 Local Corpus")
        with badge_cols[3]:
            if resp.get("guardrail_passed"):
                st.success("🛡️ Guardrails: PASS")
            else:
                st.error("🛡️ Guardrails: REFUSED")
        with badge_cols[4]:
            score = resp.get("faithfulness_score", 0)
            st.metric("Faithfulness", f"{score:.0%}")

        # Answer
        st.subheader("💬 Verified Answer")
        st.markdown(resp.get("answer", ""))

        # Graph Context
        if resp.get("graph_relations"):
            with st.expander(f"🕸️ GraphRAG Injected Triples ({len(resp['graph_relations'])})", expanded=False):
                for rel in resp["graph_relations"]:
                    st.code(rel, language="text")

        # Sources
        if resp.get("sources"):
            with st.expander(f"📄 Grounded Context & Sources ({len(resp['sources'])} chunks)", expanded=False):
                for i, src in enumerate(resp["sources"], 1):
                    meta = src.get("metadata", {})
                    st.markdown(f"**[{i}] Source:** `{meta.get('source', 'unknown')}`")
                    st.text(src.get("content", "")[:350])
                    st.divider()

        # Observability Trace
        if resp.get("trace"):
            with st.expander("🔍 Observability & Decision Trace", expanded=False):
                trace = resp["trace"]
                st.write(f"**Total Latency:** {trace.get('total_duration_ms', 0)} ms | **Trace ID:** `{trace.get('trace_id')}`")
                st.json(trace)

# --- TAB 2: Knowledge Graph ---
with tab_graph:
    st.subheader("🕸️ s@r@h Knowledge Graph (Entity-Relationship Network)")
    st.caption("GraphRAG extracts semantic triples from documents to answer multi-hop relational questions.")
    try:
        g_data = requests.get(f"{API_URL}/graph/triples", timeout=3).json()
        triples = g_data.get("triples", [])
        if triples:
            st.dataframe(triples, use_container_width=True)
        else:
            st.info("No triples in graph yet. Click 'Ingest Knowledge Base' in the sidebar.")
    except Exception as e:
        st.warning(f"Could not load graph: {e}")

# --- TAB 3: Telemetry & Drift ---
with tab_telemetry:
    st.subheader("📊 Real-Time LLM Quality & Hallucination Drift Platform")
    st.caption("Continuous production telemetry tracking faithfulness, latency percentiles, and cache efficiency.")
    try:
        t_data = requests.get(f"{API_URL}/metrics/telemetry", timeout=3).json()
        summary = t_data.get("summary", {})
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("Total Queries", summary.get("total_queries", 0))
        with m_col2:
            st.metric("Avg Faithfulness", f"{summary.get('avg_faithfulness', 1.0):.1%}")
        with m_col3:
            st.metric("Hallucination Rate", f"{summary.get('hallucination_rate_pct', 0)}%")
        with m_col4:
            st.metric("Avg Latency", f"{summary.get('avg_latency_ms', 0)} ms")

        st.divider()
        st.subheader("📜 Recent Inference Telemetry Logs")
        recent = t_data.get("recent_logs", [])
        if recent:
            st.dataframe(recent, use_container_width=True)
        else:
            st.info("No telemetry logs yet. Ask queries in the chat tab to see real-time data.")
    except Exception as e:
        st.warning(f"Could not load telemetry: {e}")

# --- TAB 4: Memory & Profile ---
with tab_memory:
    st.subheader("🧠 s@r@h Memory Agent & User Profile")
    st.caption("Persistent episodic memory stores conversation turns and automatically remembers user preferences.")
    try:
        m_data = requests.get(f"{API_URL}/memory/facts/{st.session_state.session_id}", timeout=3).json()
        facts = m_data.get("facts", {})
        st.write("#### 👤 Extracted User Facts & Preferences")
        if facts:
            for k, v in facts.items():
                st.write(f"- **{k.replace('_', ' ').title()}**: `{v}`")
        else:
            st.info("No user facts extracted yet. Try saying: *'My name is Rishi and I like Python'* in the chat tab!")

        st.divider()
        st.write("#### 💬 Session Conversation History")
        hist = m_data.get("history", [])
        if hist:
            for item in hist:
                role_icon = "👤" if item["role"] == "user" else "🌟"
                st.markdown(f"**{role_icon} {item['role'].title()}:** {item['content']}")
        else:
            st.info("Session history is empty.")
    except Exception as e:
        st.warning(f"Could not load memory: {e}")
