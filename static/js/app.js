/**
 * s@r@h Front-End Application Controller
 * Handles Chat streaming, 3D Graph integration, Telemetry, Memory, Meta-RAG, and Settings Studio.
 */

document.addEventListener("DOMContentLoaded", () => {
  const sessionId = "user_session_01";

  // --- View Switching ---
  const navItems = document.querySelectorAll(".nav-item");
  const viewSections = document.querySelectorAll(".view-section");

  navItems.forEach((item) => {
    item.addEventListener("click", () => {
      navItems.forEach((n) => n.classList.remove("active"));
      viewSections.forEach((s) => s.classList.remove("active"));

      item.classList.add("active");
      const targetView = item.getAttribute("data-view");
      const targetSec = document.getElementById(`view-${targetView}`);
      if (targetSec) targetSec.classList.add("active");

      // Trigger refreshes on view switch
      if (targetView === "telemetry") loadTelemetry();
      if (targetView === "memory") loadMemory();
      if (targetView === "evolution") loadEvolutionStatus();
      if (targetView === "graph" && window.initGraph3D) {
        setTimeout(window.initGraph3D, 100);
      }
    });
  });

  // --- Settings Modal Controller ---
  const settingsModal = document.getElementById("settings-modal-backdrop");
  const btnOpenSettingsHeader = document.getElementById("btn-open-settings-header");
  const btnOpenSettingsSidebar = document.getElementById("btn-open-settings-sidebar");
  const btnCloseSettings = document.getElementById("btn-close-settings");

  function openSettings() {
    loadUserSettings();
    settingsModal.classList.add("open");
  }

  function closeSettings() {
    settingsModal.classList.remove("open");
  }

  if (btnOpenSettingsHeader) btnOpenSettingsHeader.addEventListener("click", openSettings);
  if (btnOpenSettingsSidebar) btnOpenSettingsSidebar.addEventListener("click", openSettings);
  if (btnCloseSettings) btnCloseSettings.addEventListener("click", closeSettings);

  settingsModal.addEventListener("click", (e) => {
    if (e.target === settingsModal) closeSettings();
  });

  // Settings Tab Switching
  const settingsNavBtns = document.querySelectorAll(".settings-nav-btn");
  const settingsPanes = document.querySelectorAll(".settings-pane");

  settingsNavBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      settingsNavBtns.forEach((b) => b.classList.remove("active"));
      settingsPanes.forEach((p) => p.classList.remove("active"));

      btn.classList.add("active");
      const tabId = btn.getAttribute("data-tab");
      const targetPane = document.getElementById(`pane-${tabId}`);
      if (targetPane) targetPane.classList.add("active");
    });
  });

  // Load User Settings from Backend
  async function loadUserSettings() {
    try {
      const res = await fetch(`/settings/profile?session_id=${sessionId}`);
      const data = await res.json();
      if (data) {
        // Update inputs
        document.getElementById("settings-user-info").value = data.custom_instructions_user || "";
        document.getElementById("settings-style-info").value = data.custom_instructions_style || "";
        document.getElementById("settings-reasoning-effort").value = data.reasoning_effort || "high";
        document.getElementById("settings-toggle-autorefine").checked = data.auto_refine_enabled !== false;
        document.getElementById("settings-toggle-particles").checked = data.particles_enabled !== false;

        // Update Tier UI
        updateTierUI(data.subscription_tier);
        if (data.theme_accent) applyTheme(data.theme_accent);
      }
    } catch (e) {
      console.warn("Failed to load user settings:", e);
    }
  }

  function updateTierUI(tier) {
    const sidebarBadge = document.getElementById("sidebar-tier-badge");
    const topBadge = document.getElementById("top-plan-badge");
    const btnFree = document.getElementById("btn-plan-free");
    const btnPro = document.getElementById("btn-plan-pro");

    if (tier === "pro" || tier === "enterprise") {
      sidebarBadge.innerText = "🌟 Pro Tier";
      sidebarBadge.className = "brand-badge pro";
      topBadge.innerText = "Pro Plan";
      topBadge.style.color = "var(--accent-cyan)";
      btnFree.innerText = "Downgrade to Free";
      btnFree.className = "btn-tier";
      btnPro.innerText = "✓ Active Plan";
      btnPro.className = "btn-tier active-plan";
    } else {
      sidebarBadge.innerText = "Free Tier";
      sidebarBadge.className = "brand-badge";
      topBadge.innerText = "Free";
      topBadge.style.color = "var(--accent-amber)";
      btnFree.innerText = "Current Plan";
      btnFree.className = "btn-tier active-plan";
      btnPro.innerText = "Upgrade to Pro ➔";
      btnPro.className = "btn-tier upgrade";
    }
  }

  // Switch Subscription Tier (Global function)
  window.switchPlan = async function (targetTier) {
    try {
      const res = await fetch("/settings/tier/upgrade", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target_tier: targetTier }),
      });
      const data = await res.json();
      updateTierUI(data.current_tier);
      alert(data.message);
      refreshSystemHealth();
    } catch (e) {
      alert("Error updating plan: " + e.message);
    }
  };

  // Save Custom Persona Instructions
  const btnSaveCustom = document.getElementById("btn-save-custom-instructions");
  if (btnSaveCustom) {
    btnSaveCustom.addEventListener("click", async () => {
      btnSaveCustom.innerText = "Saving...";
      try {
        const userInfo = document.getElementById("settings-user-info").value.trim();
        const styleInfo = document.getElementById("settings-style-info").value.trim();
        const effort = document.getElementById("settings-reasoning-effort").value;
        const autoRefine = document.getElementById("settings-toggle-autorefine").checked;

        await fetch("/settings/profile", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            custom_instructions_user: userInfo,
            custom_instructions_style: styleInfo,
            reasoning_effort: effort,
            auto_refine_enabled: autoRefine,
          }),
        });
        alert("✓ Custom instructions and preferences saved successfully!");
      } catch (e) {
        alert("Error saving settings: " + e.message);
      } finally {
        btnSaveCustom.innerText = "Save Preferences";
      }
    });
  }

  // Save API Keys Vault
  const btnSaveKeys = document.getElementById("btn-save-api-keys");
  if (btnSaveKeys) {
    btnSaveKeys.addEventListener("click", async () => {
      btnSaveKeys.innerText = "Encrypting & Storing...";
      try {
        const openai = document.getElementById("vault-openai-key").value.trim();
        const anthropic = document.getElementById("vault-anthropic-key").value.trim();
        const gemini = document.getElementById("vault-gemini-key").value.trim();
        const groq = document.getElementById("vault-groq-key").value.trim();

        await fetch("/auth/keys", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            openai_api_key: openai || undefined,
            anthropic_api_key: anthropic || undefined,
            google_api_key: gemini || undefined,
            groq_api_key: groq || undefined,
          }),
        });
        alert("✓ API keys securely stored in local auth vault!");
      } catch (e) {
        alert("Error saving keys: " + e.message);
      } finally {
        btnSaveKeys.innerText = "Save Keys to Vault";
      }
    });
  }

  // Clear Cache Action
  window.clearCacheAction = async function () {
    if (!confirm("Are you sure you want to purge the semantic cache?")) return;
    try {
      const res = await fetch("/settings/cache/clear", { method: "POST" });
      const data = await res.json();
      alert(data.message);
      refreshSystemHealth();
    } catch (e) {
      alert("Error clearing cache: " + e.message);
    }
  };

  // Clear Memory Action
  window.clearMemoryAction = async function () {
    if (!confirm("Are you sure you want to clear episodic session history?")) return;
    try {
      const res = await fetch("/settings/memory/clear", { method: "POST" });
      const data = await res.json();
      alert(data.message);
      loadMemory();
    } catch (e) {
      alert("Error clearing memory: " + e.message);
    }
  };

  // Theme Accent Switcher
  window.setThemeAccent = async function (accent) {
    applyTheme(accent);
    try {
      await fetch("/settings/profile", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ theme_accent: accent }),
      });
    } catch (e) {}
  };

  function applyTheme(accent) {
    const root = document.documentElement;
    const colorBtns = document.querySelectorAll(".theme-color-btn");
    colorBtns.forEach((b) => b.classList.remove("active"));

    if (accent === "violet") {
      root.style.setProperty("--accent-cyan", "#c084fc");
      root.style.setProperty("--border-glass-glow", "rgba(192, 132, 252, 0.35)");
      root.style.setProperty("--glow-cyan", "0 0 25px rgba(192, 132, 252, 0.25)");
    } else if (accent === "emerald") {
      root.style.setProperty("--accent-cyan", "#10b981");
      root.style.setProperty("--border-glass-glow", "rgba(16, 185, 129, 0.35)");
      root.style.setProperty("--glow-cyan", "0 0 25px rgba(16, 185, 129, 0.25)");
    } else if (accent === "amber") {
      root.style.setProperty("--accent-cyan", "#f59e0b");
      root.style.setProperty("--border-glass-glow", "rgba(245, 158, 11, 0.35)");
      root.style.setProperty("--glow-cyan", "0 0 25px rgba(245, 158, 11, 0.25)");
    } else {
      root.style.setProperty("--accent-cyan", "#00f0ff");
      root.style.setProperty("--border-glass-glow", "rgba(0, 240, 255, 0.35)");
      root.style.setProperty("--glow-cyan", "0 0 25px rgba(0, 240, 255, 0.25)");
    }
  }

  // Toggle Particles
  window.toggleParticles = function (enabled) {
    const canvas = document.getElementById("bg-canvas");
    if (canvas) canvas.style.display = enabled ? "block" : "none";
  };

  // --- System Health Header Check ---
  async function refreshSystemHealth() {
    try {
      const res = await fetch("/health");
      const data = await res.json();
      if (data) {
        document.getElementById("top-llm-provider").innerText = data.llm_provider || "local";
        if (data.generation_version) {
          document.getElementById("top-gen-version").innerText = `v${data.generation_version}`;
        }
        if (data.subscription_tier) {
          updateTierUI(data.subscription_tier);
        }
        if (data.cache_stats) {
          document.getElementById("top-cache-rate").innerText = `${data.cache_stats.hit_rate_pct}%`;
        }
      }
    } catch (e) {
      console.warn("Health check failed:", e);
    }
  }
  refreshSystemHealth();
  setInterval(refreshSystemHealth, 15000);

  // --- Chat Stream & Querying ---
  const chatInput = document.getElementById("chat-query-input");
  const btnChatSend = document.getElementById("btn-chat-send");
  const chatMessagesList = document.getElementById("chat-messages-list");

  async function handleSendChat() {
    const query = chatInput.value.trim();
    if (!query) return;

    // Append User Message
    appendMessage("user", query);
    chatInput.value = "";

    // Show Loading Bot Row
    const loadingRow = appendLoadingMessage();

    try {
      const res = await fetch("/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: query, session_id: sessionId }),
      });
      const data = await res.json();

      loadingRow.remove();
      appendBotResponse(data);
      refreshSystemHealth();
    } catch (err) {
      loadingRow.remove();
      appendMessage("bot", `❌ Error querying s@r@h: ${err.message}`);
    }
  }

  btnChatSend.addEventListener("click", handleSendChat);
  chatInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") handleSendChat();
  });

  function appendMessage(role, text) {
    const row = document.createElement("div");
    row.className = `msg-row ${role}`;
    row.innerHTML = `
      <div class="msg-avatar">${role === "user" ? "R" : "🌟"}</div>
      <div class="msg-content">
        <p>${escapeHtml(text)}</p>
      </div>
    `;
    chatMessagesList.appendChild(row);
    chatMessagesList.scrollTop = chatMessagesList.scrollHeight;
    return row;
  }

  function appendLoadingMessage() {
    const row = document.createElement("div");
    row.className = "msg-row bot";
    row.innerHTML = `
      <div class="msg-avatar">🌟</div>
      <div class="msg-content">
        <div style="display: flex; gap: 6px; align-items: center; color: var(--accent-cyan); font-size: 13px;">
          <span class="status-dot"></span> s@r@h is reasoning across Hybrid Search, GraphRAG & Guardrails...
        </div>
      </div>
    `;
    chatMessagesList.appendChild(row);
    chatMessagesList.scrollTop = chatMessagesList.scrollHeight;
    return row;
  }

  function appendBotResponse(data) {
    const row = document.createElement("div");
    row.className = "msg-row bot";

    const isGuardPassed = data.guardrail_passed !== false;
    const isCacheHit = data.cache_hit === true;
    const isWebGrounded = data.web_grounded === true;
    const isMultiHop = data.multi_hop_used === true;

    let badgesHtml = "";
    if (isCacheHit) {
      badgesHtml += `<span class="badge" style="background: rgba(0,240,255,0.2); color: var(--accent-cyan);">⚡ Cache: HIT (&lt;10ms)</span>`;
    } else {
      badgesHtml += `<span class="badge ${isGuardPassed ? "pass" : "fail"}">${isGuardPassed ? "🛡️ Guardrail: PASSED" : "⚠️ Guardrail: REWRITTEN"}</span>`;
    }
    if (isWebGrounded) {
      badgesHtml += `<span class="badge" style="background: rgba(16,185,129,0.15); color: var(--accent-emerald);">🌐 Live Web Grounded</span>`;
    }
    if (isMultiHop) {
      badgesHtml += `<span class="badge" style="background: rgba(168,85,247,0.15); color: var(--accent-violet);">🧩 Multi-Hop Decomposed</span>`;
    }
    if (data.faithfulness_score) {
      badgesHtml += `<span class="badge" style="background: rgba(255,255,255,0.06);">Faithfulness: ${Math.round(data.faithfulness_score * 100)}%</span>`;
    }

    let sourcesHtml = "";
    if (data.sources && data.sources.length > 0) {
      sourcesHtml = `
        <div style="margin-top: 12px; padding-top: 10px; border-top: 1px solid var(--border-glass); font-size: 12.5px;">
          <strong style="color: var(--accent-cyan);">📚 Verified Sources & Citations:</strong>
          <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-top: 6px;">
            ${data.sources
              .slice(0, 3)
              .map(
                (s, i) =>
                  `<span style="padding: 4px 8px; border-radius: 6px; background: rgba(0,0,0,0.3); border: 1px solid var(--border-glass); color: var(--text-secondary);">[${i + 1}] ${escapeHtml(s.metadata.source || "Knowledge Base")}</span>`
              )
              .join("")}
          </div>
        </div>
      `;
    }

    let traceHtml = "";
    if (data.trace && data.trace.steps) {
      traceHtml = `
        <details style="margin-top: 10px; font-size: 12px; color: var(--text-muted);">
          <summary style="cursor: pointer; color: var(--accent-cyan);">🔍 Inspect Decision Trace (${data.trace.total_duration_ms || 0} ms)</summary>
          <div style="margin-top: 8px; background: rgba(0,0,0,0.4); padding: 10px; border-radius: 8px; font-family: monospace;">
            ${data.trace.steps
              .map((st) => `<div>• [${st.status.toUpperCase()}] <strong>${st.step_name}</strong>: ${st.duration_ms}ms</div>`)
              .join("")}
          </div>
        </details>
      `;
    }

    row.innerHTML = `
      <div class="msg-avatar">🌟</div>
      <div class="msg-content">
        <div class="badge-row">${badgesHtml}</div>
        <p style="white-space: pre-wrap; line-height: 1.6;">${escapeHtml(data.answer)}</p>
        ${sourcesHtml}
        ${traceHtml}
      </div>
    `;
    chatMessagesList.appendChild(row);
    chatMessagesList.scrollTop = chatMessagesList.scrollHeight;
  }

  // --- Telemetry Loader ---
  async function loadTelemetry() {
    try {
      const res = await fetch("/metrics/telemetry");
      const data = await res.json();
      if (data && data.summary) {
        document.getElementById("telemetry-total-q").innerText = data.summary.total_queries || "0";
        document.getElementById("telemetry-avg-faith").innerText = `${Math.round((data.summary.avg_faithfulness || 1.0) * 100)}%`;
        document.getElementById("telemetry-drift-rate").innerText = `${data.summary.hallucination_rate_pct || 0}%`;
        document.getElementById("telemetry-avg-latency").innerText = `${Math.round(data.summary.avg_latency_ms || 0)} ms`;
      }

      if (data && data.recent_logs) {
        const tableBody = document.getElementById("telemetry-logs-table");
        if (data.recent_logs.length === 0) {
          tableBody.innerHTML = `<tr><td colspan="6" style="padding: 20px; text-align: center; color: var(--text-muted);">No queries logged yet.</td></tr>`;
        } else {
          tableBody.innerHTML = data.recent_logs
            .map(
              (log) => `
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.04);">
              <td style="padding: 8px 10px; color: var(--text-muted);">${log.timestamp}</td>
              <td style="padding: 8px 10px; max-width: 260px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${escapeHtml(log.query)}</td>
              <td style="padding: 8px 10px; color: var(--accent-emerald); font-weight: 600;">${Math.round(log.faithfulness * 100)}%</td>
              <td style="padding: 8px 10px;"><span class="badge ${log.guardrail_passed ? "pass" : "fail"}">${log.guardrail_passed ? "PASS" : "HEAL"}</span></td>
              <td style="padding: 8px 10px;">${log.cache_hit ? "⚡ HIT" : "MISS"}</td>
              <td style="padding: 8px 10px; color: var(--accent-violet);">${Math.round(log.latency_ms)} ms</td>
            </tr>
          `
            )
            .join("");
        }
      }
    } catch (e) {
      console.warn("Failed to load telemetry:", e);
    }
  }

  // --- Memory Loader ---
  async function loadMemory() {
    try {
      const res = await fetch(`/memory/facts/${sessionId}`);
      const data = await res.json();
      const list = document.getElementById("memory-facts-list");
      if (data && data.facts && Object.keys(data.facts).length > 0) {
        list.innerHTML = Object.entries(data.facts)
          .map(
            ([k, v]) => `
            <div style="display: flex; justify-content: space-between; padding: 10px 14px; border-radius: 8px; background: rgba(255,255,255,0.03); border: 1px solid var(--border-glass);">
              <span style="color: var(--accent-cyan); font-weight: 500;">${escapeHtml(k)}</span>
              <span style="color: var(--text-primary); font-weight: 600;">${escapeHtml(v)}</span>
            </div>
          `
          )
          .join("");
      } else {
        list.innerHTML = `<p style="color: var(--text-muted); font-size: 13.5px;">No persistent facts saved yet. Chat with s@r@h (e.g. "My name is Rishi and I like Python") to auto-extract entities!</p>`;
      }
    } catch (e) {
      console.warn("Failed to load memory:", e);
    }
  }

  // --- Meta-RAG Evolution Loader & Controls ---
  async function loadEvolutionStatus() {
    try {
      const res = await fetch("/evolution/status");
      const data = await res.json();
      if (data && data.current_parameters) {
        const p = data.current_parameters;
        document.getElementById("evo-version-tag").innerText = `${p.version} Active`;
        document.getElementById("evo-v-weight").innerText = p.vector_weight;
        document.getElementById("evo-b-weight").innerText = p.bm25_weight;
        document.getElementById("evo-rel-thresh").innerText = p.relevance_threshold;
        document.getElementById("evo-rrf-k").innerText = p.rrf_k;
      }
    } catch (e) {
      console.warn("Failed to load evolution status:", e);
    }
  }

  // Auto-Tune Button
  const btnAutoTune = document.getElementById("btn-trigger-autotune");
  if (btnAutoTune) {
    btnAutoTune.addEventListener("click", async () => {
      btnAutoTune.innerText = "⏳ Self-Tuning In Progress...";
      try {
        const res = await fetch("/evolution/auto-tune", { method: "POST" });
        const data = await res.json();
        alert(`🧬 Meta-RAG Optimization Result:\n\n` + data.mutations.join("\n"));
        loadEvolutionStatus();
        refreshSystemHealth();
      } catch (e) {
        alert("Self-tuning error: " + e.message);
      } finally {
        btnAutoTune.innerText = "🔄 Run Autonomous Self-Tuning";
      }
    });
  }

  // Synthetic Train Button
  const btnSyntheticTrain = document.getElementById("btn-run-synthetic-train");
  const synthStatus = document.getElementById("synthetic-train-status");
  if (btnSyntheticTrain) {
    btnSyntheticTrain.addEventListener("click", async () => {
      btnSyntheticTrain.innerText = "⏳ Generating QA & Training...";
      synthStatus.innerText = "Mining concepts & running self-evaluation benchmark...";
      try {
        const res = await fetch("/evolution/synthetic-train", { method: "POST" });
        const data = await res.json();
        synthStatus.innerHTML = `
          <strong style="color: var(--accent-emerald);">✓ Self-Trained Accuracy: ${data.self_trained_accuracy}</strong>
          (${data.passed}/${data.dataset_size} QA pairs passed retrieval gates).
        `;
      } catch (e) {
        synthStatus.innerText = "Error running synthetic trainer: " + e.message;
      } finally {
        btnSyntheticTrain.innerText = "🚀 Generate Synthetic QA & Train";
      }
    });
  }

  // External RAG Modifier Button
  const btnAuditRag = document.getElementById("btn-audit-external-rag");
  const ragInput = document.getElementById("external-rag-input");
  const ragResults = document.getElementById("external-rag-results");

  if (btnAuditRag) {
    btnAuditRag.addEventListener("click", async () => {
      const code = ragInput.value.trim();
      if (!code) {
        alert("Please paste external RAG code or configuration to audit.");
        return;
      }

      btnAuditRag.innerText = "🔍 Auditing & Generating Patch...";
      try {
        const res = await fetch("/evolution/audit-rag", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ config_or_code: code }),
        });
        const data = await res.json();

        ragResults.style.display = "block";
        ragResults.innerHTML = `
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <h4 style="font-size: 16px; color: var(--accent-cyan);">Diagnostic Health Score: ${data.overall_health_score}/100</h4>
            <span class="badge ${data.overall_health_score >= 70 ? "pass" : "fail"}">${data.retrieval_architecture}</span>
          </div>

          <div style="margin-bottom: 12px;">
            <strong style="color: #ef4444; font-size: 13px;">🚨 Vulnerabilities Detected:</strong>
            <ul style="margin: 6px 0 0 18px; color: var(--text-secondary); font-size: 13px;">
              ${data.vulnerabilities_detected.map((v) => `<li>${escapeHtml(v)}</li>`).join("")}
            </ul>
          </div>

          <div style="margin-bottom: 14px;">
            <strong style="color: var(--accent-emerald); font-size: 13px;">💡 Optimization Recommendations:</strong>
            <ul style="margin: 6px 0 0 18px; color: var(--text-secondary); font-size: 13px;">
              ${data.optimization_recommendations.map((r) => `<li>${escapeHtml(r)}</li>`).join("")}
            </ul>
          </div>

          <div>
            <strong style="color: var(--accent-violet); font-size: 13px;">🛠️ s@r@h Auto-Generated Upgraded RAG Implementation:</strong>
            <pre style="margin-top: 8px; background: rgba(0,0,0,0.5); padding: 12px; border-radius: 8px; border: 1px solid var(--border-glass); color: #a5f3fc; font-size: 12px; overflow-x: auto;"><code>${escapeHtml(data.generated_code_patch)}</code></pre>
          </div>
        `;
      } catch (e) {
        alert("Error auditing RAG: " + e.message);
      } finally {
        btnAuditRag.innerText = "🔍 Audit & Auto-Modify External RAG";
      }
    });
  }

  // --- Ingest Corpus Button ---
  const btnIngest = document.getElementById("btn-ingest-corpus");
  if (btnIngest) {
    btnIngest.addEventListener("click", async () => {
      btnIngest.innerText = "⏳ Ingesting AI/ML Textbooks...";
      try {
        const res = await fetch("/ingest", { method: "POST" });
        const data = await res.json();
        alert(`✓ Ingestion Complete!\n\nDocuments Loaded: ${data.documents_loaded}\nChunks Indexed: ${data.chunks_indexed}\n3D Graph Triples: ${data.graph_triples_indexed}`);
        refreshSystemHealth();
      } catch (e) {
        alert("Ingestion error: " + e.message);
      } finally {
        btnIngest.innerText = "📥 Ingest Entire AI/ML Corpus";
      }
    });
  }

  function escapeHtml(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
});
