/* Main Application Controller for s@r@h Glassmorphic UI */

let currentSession = {
  sessionId: localStorage.getItem("sarah_session_id") || "user_session_" + Math.floor(Math.random() * 10000),
  username: localStorage.getItem("sarah_username") || "Rishi",
  role: "Lead AI Engineer",
};

let graphVisualizer = null;

document.addEventListener("DOMContentLoaded", () => {
  localStorage.setItem("sarah_session_id", currentSession.sessionId);
  initNavigation();
  initChat();
  initIngestion();
  fetchHealth();
  fetchTelemetry();
  fetchMemory();
});

// View Navigation
function initNavigation() {
  const navItems = document.querySelectorAll(".nav-item");
  navItems.forEach((item) => {
    item.addEventListener("click", () => {
      navItems.forEach((i) => i.classList.remove("active"));
      item.classList.add("active");

      const viewName = item.getAttribute("data-view");
      document.querySelectorAll(".view-section").forEach((sec) => sec.classList.remove("active"));
      const targetSec = document.getElementById(`view-${viewName}`);
      if (targetSec) targetSec.classList.add("active");

      if (viewName === "graph" && !graphVisualizer) {
        setTimeout(() => {
          graphVisualizer = new KnowledgeGraph3D("graph-canvas");
        }, 150);
      } else if (viewName === "telemetry") {
        fetchTelemetry();
      } else if (viewName === "memory") {
        fetchMemory();
      }
    });
  });
}

// System Health Polling
async function fetchHealth() {
  try {
    const res = await fetch("/health");
    const data = await res.json();
    document.getElementById("top-chunks-count").innerText = data.vector_store_count || 0;
    document.getElementById("top-triples-count").innerText = data.graph_triples_count || 0;
    document.getElementById("top-cache-rate").innerText = `${data.cache_stats?.hit_rate_pct || 0}%`;
    document.getElementById("top-llm-provider").innerText = data.llm_provider || "local";
  } catch (e) {
    console.warn("Health check error:", e);
  }
}

// Chat Functionality
function initChat() {
  const sendBtn = document.getElementById("btn-chat-send");
  const input = document.getElementById("chat-query-input");
  const msgList = document.getElementById("chat-messages-list");

  const sendQuery = async () => {
    const text = input.value.trim();
    if (!text) return;
    input.value = "";

    // Append User Message
    appendMessage("user", text);

    // Show loading placeholder
    const loadingId = "loading-" + Date.now();
    const loadingDiv = document.createElement("div");
    loadingDiv.className = "msg-row bot";
    loadingDiv.id = loadingId;
    loadingDiv.innerHTML = `
      <div class="msg-avatar">🌟</div>
      <div class="msg-content" style="display: flex; gap: 8px; align-items: center;">
        <span class="status-dot"></span>
        <span style="color: var(--accent-cyan); font-size: 13.5px;">s@r@h is processing (Security ➔ Cache ➔ GraphRAG ➔ Hybrid Search ➔ Guardrails)...</span>
      </div>
    `;
    msgList.appendChild(loadingDiv);
    msgList.scrollTop = msgList.scrollHeight;

    try {
      const resp = await fetch("/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: text,
          session_id: currentSession.sessionId,
          top_k: 5,
          enable_web_fallback: true,
          enable_graph_rag: true,
        }),
      });
      const data = await resp.json();

      document.getElementById(loadingId)?.remove();
      renderBotResponse(data);
      fetchHealth();
    } catch (err) {
      document.getElementById(loadingId)?.remove();
      appendMessage("bot", `⚠️ Error processing query: ${err.message}`);
    }
  };

  sendBtn.addEventListener("click", sendQuery);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendQuery();
  });
}

function appendMessage(role, text) {
  const msgList = document.getElementById("chat-messages-list");
  const row = document.createElement("div");
  row.className = `msg-row ${role}`;
  row.innerHTML = `
    <div class="msg-avatar">${role === "user" ? "👤" : "🌟"}</div>
    <div class="msg-content">${escapeHtml(text)}</div>
  `;
  msgList.appendChild(row);
  msgList.scrollTop = msgList.scrollHeight;
}

function renderBotResponse(data) {
  const msgList = document.getElementById("chat-messages-list");
  const row = document.createElement("div");
  row.className = "msg-row bot";

  const cacheBadge = data.cache_hit
    ? `<span class="badge cache">⚡ Cache: HIT (<10ms)</span>`
    : `<span class="badge" style="background: rgba(255,255,255,0.06); color: var(--text-secondary)">⚡ Cache: MISS</span>`;

  const guardBadge = data.guardrail_passed
    ? `<span class="badge pass">🛡️ Guardrails: PASS (${Math.round((data.faithfulness_score || 1) * 100)}%)</span>`
    : `<span class="badge" style="background: rgba(244,63,94,0.15); color: var(--accent-rose)">🛡️ Guardrails: REFUSED</span>`;

  const multiBadge = data.multi_hop_used
    ? `<span class="badge multi">🧠 Multi-Hop Reasoning</span>`
    : "";

  const graphBadge = data.graph_relations?.length > 0
    ? `<span class="badge" style="background: rgba(168,85,247,0.15); color: var(--accent-violet)">🕸️ ${data.graph_relations.length} Graph Triples</span>`
    : "";

  let sourcesHtml = "";
  if (data.sources?.length > 0) {
    sourcesHtml = `
      <details style="margin-top: 12px; font-size: 12px; border-top: 1px solid var(--border-glass); padding-top: 8px;">
        <summary style="color: var(--accent-cyan); cursor: pointer; font-weight: 500;">📄 View ${data.sources.length} Grounded Source Chunks</summary>
        <div style="margin-top: 8px; display: flex; flex-direction: column; gap: 8px;">
          ${data.sources.map((s, idx) => `
            <div style="background: rgba(0,0,0,0.3); padding: 8px 12px; border-radius: 8px; border-left: 2px solid var(--accent-cyan);">
              <strong>[${idx + 1}] ${s.metadata?.source || "Source"}</strong>
              <p style="margin-top: 4px; color: var(--text-secondary);">${escapeHtml(s.content)}</p>
            </div>
          `).join("")}
        </div>
      </details>
    `;
  }

  let traceHtml = "";
  if (data.trace?.total_duration_ms) {
    traceHtml = `
      <details style="margin-top: 8px; font-size: 11px; color: var(--text-muted);">
        <summary style="cursor: pointer;">🔍 Decision Trace & Latency (${data.trace.total_duration_ms} ms)</summary>
        <pre style="margin-top: 6px; padding: 8px; background: rgba(0,0,0,0.4); border-radius: 6px; overflow-x: auto;">${JSON.stringify(data.trace, null, 2)}</pre>
      </details>
    `;
  }

  row.innerHTML = `
    <div class="msg-avatar">🌟</div>
    <div class="msg-content">
      <div class="badge-row">
        ${cacheBadge}
        ${guardBadge}
        ${multiBadge}
        ${graphBadge}
      </div>
      <div>${formatMarkdown(data.answer || "")}</div>
      ${sourcesHtml}
      ${traceHtml}
    </div>
  `;
  msgList.appendChild(row);
  msgList.scrollTop = msgList.scrollHeight;
}

// Ingestion Trigger
function initIngestion() {
  const btn = document.getElementById("btn-ingest-corpus");
  if (!btn) return;
  btn.addEventListener("click", async () => {
    btn.innerText = "⏳ Ingesting AI/ML Knowledge Base...";
    btn.disabled = true;
    try {
      const res = await fetch("/ingest", { method: "POST" });
      const data = await res.json();
      alert(`✅ Ingestion Complete!\n- Loaded: ${data.documents_loaded} docs\n- Chunks: ${data.chunks_created}\n- Graph Triples: ${data.graph_triples_indexed}`);
      fetchHealth();
      if (graphVisualizer) graphVisualizer.loadGraphData();
    } catch (e) {
      alert(`Ingestion failed: ${e.message}`);
    } finally {
      btn.innerText = "📥 Ingest Entire AI/ML Corpus";
      btn.disabled = false;
    }
  });
}

// Telemetry & Drift Fetcher
async function fetchTelemetry() {
  try {
    const res = await fetch("/metrics/telemetry");
    const data = await res.json();
    const sum = data.summary || {};

    document.getElementById("telemetry-total-q").innerText = sum.total_queries || 0;
    document.getElementById("telemetry-avg-faith").innerText = `${Math.round((sum.avg_faithfulness || 1) * 100)}%`;
    document.getElementById("telemetry-drift-rate").innerText = `${sum.hallucination_rate_pct || 0}%`;
    document.getElementById("telemetry-avg-latency").innerText = `${sum.avg_latency_ms || 0} ms`;

    const logsContainer = document.getElementById("telemetry-logs-table");
    if (logsContainer && data.recent_logs) {
      logsContainer.innerHTML = data.recent_logs.map(log => `
        <tr>
          <td style="padding: 10px; border-bottom: 1px solid var(--border-glass); color: var(--accent-cyan);">${log.time}</td>
          <td style="padding: 10px; border-bottom: 1px solid var(--border-glass);">${escapeHtml(log.query)}</td>
          <td style="padding: 10px; border-bottom: 1px solid var(--border-glass); color: var(--accent-emerald);">${log.faithfulness}</td>
          <td style="padding: 10px; border-bottom: 1px solid var(--border-glass);">${log.passed ? "✅ Pass" : "⛔ Refused"}</td>
          <td style="padding: 10px; border-bottom: 1px solid var(--border-glass); color: var(--accent-violet);">${log.cache_hit ? "⚡ Hit" : "Miss"}</td>
          <td style="padding: 10px; border-bottom: 1px solid var(--border-glass);">${log.latency_ms}</td>
        </tr>
      `).join("");
    }
  } catch (e) {
    console.warn("Telemetry fetch error:", e);
  }
}

// Memory Vault Fetcher
async function fetchMemory() {
  try {
    const res = await fetch(`/memory/facts/${currentSession.sessionId}`);
    const data = await res.json();

    const factsContainer = document.getElementById("memory-facts-list");
    if (factsContainer) {
      const facts = data.facts || {};
      const keys = Object.keys(facts);
      if (keys.length === 0) {
        factsContainer.innerHTML = `<p style="color: var(--text-muted); font-size: 13.5px;">No facts remembered yet. In chat, say: <em>"My name is Rishi and I prefer PyTorch"</em></p>`;
      } else {
        factsContainer.innerHTML = keys.map(k => `
          <div style="display: flex; justify-content: space-between; padding: 10px 14px; background: rgba(255,255,255,0.04); border-radius: 8px; border: 1px solid var(--border-glass);">
            <strong style="color: var(--accent-cyan);">${k.replace('_', ' ').toUpperCase()}:</strong>
            <span>${escapeHtml(facts[k])}</span>
          </div>
        `).join("");
      }
    }
  } catch (e) {
    console.warn("Memory fetch error:", e);
  }
}

// Simple Markdown Formatter
function formatMarkdown(text) {
  let html = escapeHtml(text);
  // Bold
  html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
  // Code blocks
  html = html.replace(/```([a-z]*)\n([\s\S]*?)```/g, "<pre style='background: rgba(0,0,0,0.5); padding: 12px; border-radius: 8px; margin: 8px 0; overflow-x: auto;'><code>$2</code></pre>");
  // Inline code
  html = html.replace(/`([^`]+)`/g, "<code style='background: rgba(255,255,255,0.1); padding: 2px 5px; border-radius: 4px; color: var(--accent-cyan);'>$1</code>");
  // Citations [1], [2]
  html = html.replace(/\[(\d+)\]/g, "<span style='color: var(--accent-cyan); font-weight: 700; background: rgba(0,240,255,0.15); padding: 1px 5px; border-radius: 4px;'>[$1]</span>");
  // Newlines
  html = html.replace(/\n/g, "<br/>");
  return html;
}

function escapeHtml(string) {
  return String(string).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
