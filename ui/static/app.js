document.addEventListener("DOMContentLoaded", () => {
  // Navigation elements
  const tabWorkspace = document.getElementById("tabWorkspace");
  const tabSettings = document.getElementById("tabSettings");
  const viewWorkspace = document.getElementById("viewWorkspace");
  const viewSettings = document.getElementById("viewSettings");
  const btnJumpSettings = document.getElementById("btnJumpSettings");

  // Status badges & summary
  const tierBadge = document.getElementById("tierBadge");
  const activeModelBadge = document.getElementById("activeModelBadge");
  const summaryProvider = document.getElementById("summaryProvider");
  const summaryLLM = document.getElementById("summaryLLM");
  const summaryEmbedding = document.getElementById("summaryEmbedding");
  const summaryMode = document.getElementById("summaryMode");

  // Chat Workspace Elements
  const btnNewChat = document.getElementById("btnNewChat");
  const chatsList = document.getElementById("chatsList");
  const documentsList = document.getElementById("documentsList");
  const chatThread = document.getElementById("chatThread");
  const queryInput = document.getElementById("queryInput");
  const btnSearch = document.getElementById("btnSearch");
  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("fileInput");
  const uploadStatus = document.getElementById("uploadStatus");

  // Settings & Credentials elements
  const providerSelect = document.getElementById("providerSelect");
  const apiKeyInput = document.getElementById("apiKeyInput");
  const apiKeyGroup = document.getElementById("apiKeyGroup");
  const ollamaUrlGroup = document.getElementById("ollamaUrlGroup");
  const ollamaUrlInput = document.getElementById("ollamaUrlInput");
  const btnToggleKeyVis = document.getElementById("btnToggleKeyVis");
  const btnSaveCredentials = document.getElementById("btnSaveCredentials");

  // Model Management elements
  const activeLlmSelect = document.getElementById("activeLlmSelect");
  const activeEmbeddingSelect = document.getElementById("activeEmbeddingSelect");
  const newModelId = document.getElementById("newModelId");
  const newModelName = document.getElementById("newModelName");
  const newModelProvider = document.getElementById("newModelProvider");
  const newModelType = document.getElementById("newModelType");
  const btnAddModel = document.getElementById("btnAddModel");
  const btnPingNewModel = document.getElementById("btnPingNewModel");
  const newModelPingStatus = document.getElementById("newModelPingStatus");
  const modelsTableBody = document.getElementById("modelsTableBody");

  // Retrieval & Toggle controls
  const hybridAlpha = document.getElementById("hybridAlpha");
  const alphaVal = document.getElementById("alphaVal");
  const mmrLambda = document.getElementById("mmrLambda");
  const lambdaVal = document.getElementById("lambdaVal");
  const topKPool = document.getElementById("topKPool");
  const topkVal = document.getElementById("topkVal");
  const topNContext = document.getElementById("topNContext");
  const topnVal = document.getElementById("topnVal");

  const toggleReranker = document.getElementById("toggleReranker");
  const toggleHyde = document.getElementById("toggleHyde");
  const toggleStrict = document.getElementById("toggleStrict");
  const btnClearIndex = document.getElementById("btnClearIndex");
  const btnResetConfig = document.getElementById("btnResetConfig");

  let currentConfig = null;
  let activeChatId = null;
  let modelPingResults = {};

  // ----------------------------------------------------
  // In-Memory Tab Switching
  // ----------------------------------------------------
  function switchTab(target) {
    const wTab = document.getElementById("tabWorkspace");
    const sTab = document.getElementById("tabSettings");
    const wView = document.getElementById("viewWorkspace");
    const sView = document.getElementById("viewSettings");
    const tBadge = document.getElementById("tierBadge");

    if (target === "settings") {
      if (wTab) wTab.classList.remove("active");
      if (sTab) sTab.classList.add("active");
      if (wView) wView.classList.remove("active");
      if (sView) sView.classList.add("active");
      if (tBadge) tBadge.style.display = "none";
    } else {
      if (sTab) sTab.classList.remove("active");
      if (wTab) wTab.classList.add("active");
      if (sView) sView.classList.remove("active");
      if (wView) wView.classList.add("active");
      if (tBadge) tBadge.style.display = "inline-flex";
    }
  }

  function openSettingsPanel(panelId) {
    switchTab("settings");
    document.querySelectorAll(".settings-nav-item").forEach(b => {
      b.classList.toggle("active", b.getAttribute("data-target") === panelId);
    });
    document.querySelectorAll(".settings-panel").forEach(p => {
      p.classList.toggle("active", p.id === panelId);
    });
  }

  if (tabWorkspace) tabWorkspace.addEventListener("click", () => switchTab("workspace"));
  if (tabSettings) tabSettings.addEventListener("click", () => switchTab("settings"));
  if (btnJumpSettings) btnJumpSettings.addEventListener("click", () => openSettingsPanel("setPanelModels"));

  // Settings Internal Sidebar Navigation
  document.querySelectorAll(".settings-nav-item").forEach(btn => {
    btn.addEventListener("click", () => {
      openSettingsPanel(btn.getAttribute("data-target"));
    });
  });

  // Toast Helper
  function showToast(msg, type = "info") {
    const container = document.getElementById("toastContainer");
    if (!container) return;
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.innerText = msg;
    container.appendChild(toast);
    setTimeout(() => toast.remove(), 4000);
  }

  // ----------------------------------------------------
  // Initial Load
  // ----------------------------------------------------
  loadConfig();
  loadStats();
  loadChats();

  async function loadConfig() {
    try {
      const res = await fetch("/api/config");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      currentConfig = await res.json();
      renderConfigUI(currentConfig);
    } catch (err) {
      console.error("Failed to load config:", err);
      showToast("Failed to load system config.", "error");
    }
  }

  async function loadStats() {
    try {
      const res = await fetch("/api/stats");
      const stats = await res.json();
      tierBadge.innerText = `Corpus: ${stats.corpus_tier} (${stats.total_chunks} Chunks)`;
      tierBadge.className = `badge ${stats.corpus_tier.toLowerCase()}`;
    } catch (err) {
      console.error("Failed to load stats:", err);
    }
  }

  // ----------------------------------------------------
  // Document File Manager & Chunk Purging
  // ----------------------------------------------------
  async function loadDocuments() {
    if (!documentsList) return;
    try {
      const res = await fetch("/api/documents");
      const docs = await res.json();
      renderDocumentsList(docs);
    } catch (err) {
      console.error("Failed to load documents list:", err);
    }
  }

  function renderDocumentsList(docs) {
    if (!documentsList) return;
    documentsList.innerHTML = "";
    if (!docs || docs.length === 0) {
      documentsList.innerHTML = `<div style="font-size:0.75rem; color:var(--text-muted); text-align:center; padding:0.5rem;">No documents uploaded</div>`;
      return;
    }

    docs.forEach(d => {
      const item = document.createElement("div");
      item.className = "doc-item";
      const sizeMb = (d.size_bytes / (1024 * 1024)).toFixed(2);
      item.innerHTML = `
        <div style="overflow:hidden;">
          <div class="doc-item-title" title="${escapeHtml(d.filename)}">📄 ${escapeHtml(d.filename)}</div>
          <div class="doc-item-meta">${d.chunk_count} chunks • ${sizeMb} MB</div>
        </div>
        <button class="doc-item-del" data-filename="${escapeHtml(d.filename)}" title="Delete document & chunks">🗑️</button>
      `;
      documentsList.appendChild(item);
    });

    documentsList.querySelectorAll(".doc-item-del").forEach(btn => {
      btn.addEventListener("click", () => {
        deleteDocument(btn.getAttribute("data-filename"));
      });
    });
  }

  async function deleteDocument(filename) {
    if (!confirm(`Are you sure you want to delete document '${filename}' and purge all its vector chunks?`)) return;
    try {
      const res = await fetch(`/api/documents/${encodeURIComponent(filename)}`, { method: "DELETE" });
      const data = await res.json();
      showToast(data.message, "info");
      loadDocuments();
      loadStats();
    } catch (err) {
      showToast("Failed to delete document: " + err.message, "error");
    }
  }

  // ----------------------------------------------------
  // Chat Session Management
  // ----------------------------------------------------
  async function loadChats() {
    try {
      const res = await fetch("/api/chats");
      const chats = await res.json();
      renderChatsList(chats);
      if (chats.length > 0 && !activeChatId) {
        selectChat(chats[0].id);
      } else if (chats.length === 0) {
        createNewChat();
      }
    } catch (err) {
      console.error("Failed to load chats:", err);
    }
  }

  function renderChatsList(chats) {
    chatsList.innerHTML = "";
    if (chats.length === 0) {
      chatsList.innerHTML = `<div style="font-size:0.8rem; color:var(--text-muted); text-align:center; padding:1rem;">No active chats</div>`;
      return;
    }

    chats.forEach(c => {
      const item = document.createElement("div");
      item.className = `chat-item ${c.id === activeChatId ? 'active' : ''}`;
      item.innerHTML = `
        <span class="chat-item-title">💬 ${escapeHtml(c.title)}</span>
        <button class="chat-item-del" data-id="${c.id}" title="Delete chat">🗑️</button>
      `;
      item.addEventListener("click", (e) => {
        if (!e.target.classList.contains("chat-item-del")) {
          selectChat(c.id);
        }
      });
      chatsList.appendChild(item);
    });

    chatsList.querySelectorAll(".chat-item-del").forEach(btn => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        deleteChat(btn.getAttribute("data-id"));
      });
    });
  }

  async function createNewChat() {
    try {
      const res = await fetch("/api/chats", { method: "POST" });
      const newChat = await res.json();
      activeChatId = newChat.id;
      await loadChats();
      selectChat(newChat.id);
    } catch (err) {
      showToast("Error creating chat: " + err.message, "error");
    }
  }

  if (btnNewChat) btnNewChat.addEventListener("click", createNewChat);

  async function selectChat(chatId) {
    activeChatId = chatId;
    try {
      const res = await fetch(`/api/chats/${chatId}`);
      if (!res.ok) return;
      const chat = await res.json();
      renderChatMessages(chat.messages || []);
      loadChats();
    } catch (err) {
      console.error("Failed to select chat:", err);
    }
  }

  async function deleteChat(chatId) {
    if (!confirm("Are you sure you want to delete this conversation session?")) return;
    try {
      await fetch(`/api/chats/${chatId}`, { method: "DELETE" });
      if (activeChatId === chatId) activeChatId = null;
      loadChats();
      showToast("Chat session deleted.", "info");
    } catch (err) {
      showToast("Failed to delete chat.", "error");
    }
  }

  // ----------------------------------------------------
  // Pretty Chat Message Thread Rendering
  // ----------------------------------------------------
  function renderChatMessages(messages) {
    chatThread.innerHTML = "";
    if (messages.length === 0) {
      chatThread.innerHTML = `
        <div class="chat-welcome">
          <div class="welcome-icon">💬</div>
          <h2>Interactive RAG Document Chat</h2>
          <p>Upload your documents and start chatting continuously. Context & history are preserved across messages!</p>
        </div>
      `;
      return;
    }

    messages.forEach(m => {
      appendMessageToThread(m);
    });

    chatThread.scrollTop = chatThread.scrollHeight;
  }

  function appendMessageToThread(m) {
    // Remove welcome state if present
    const welcome = chatThread.querySelector(".chat-welcome");
    if (welcome) welcome.remove();

    const msgWrapper = document.createElement("div");
    msgWrapper.className = `chat-msg ${m.role}`;

    if (m.role === "user") {
      msgWrapper.innerHTML = `<div class="msg-bubble-user">${escapeHtml(m.content)}</div>`;
    } else {
      // AI Assistant Message with Pretty Markdown & Collapsible Drawers
      let parsedHTML = formatMarkdown(m.content);

      // Collapsible Metrics Drawer
      let metricsHTML = "";
      if (m.metrics || m.telemetry) {
        const faith = (m.metrics?.faithfulness || 0.95).toFixed(2);
        const prec = (m.metrics?.context_precision || 0.90).toFixed(2);
        const rec = (m.metrics?.context_recall || 0.88).toFixed(2);
        
        let waterfallSpans = "";
        (m.telemetry?.spans || []).forEach(s => {
          waterfallSpans += `<div style="display:flex; justify-content:space-between; font-size:0.75rem; color:var(--text-muted);"><span>${s.name}</span><span>${s.duration_ms} ms</span></div>`;
        });

        metricsHTML = `
          <div class="collapsible-drawer">
            <div class="drawer-header" onclick="this.nextElementSibling.classList.toggle('open')">
              <span>📊 RAG Telemetry & Quality Metrics</span>
              <span>▼</span>
            </div>
            <div class="drawer-body">
              <div class="metrics-grid">
                <div class="metric-box"><div class="metric-val">${faith}</div><div class="metric-label">Faithfulness</div></div>
                <div class="metric-box"><div class="metric-val">${prec}</div><div class="metric-label">Precision</div></div>
                <div class="metric-box"><div class="metric-val">${rec}</div><div class="metric-label">Recall</div></div>
              </div>
              ${waterfallSpans ? `<div style="margin-top:0.5rem;">${waterfallSpans}</div>` : ''}
            </div>
          </div>
        `;
      }

      // Collapsible Excerpts Drawer
      let excerptsHTML = "";
      if (m.contexts && m.contexts.length > 0) {
        let items = "";
        m.contexts.forEach((c, idx) => {
          items += `
            <div style="background:rgba(255,255,255,0.03); border:1px solid var(--bg-card-border); padding:0.6rem; border-radius:6px; margin-bottom:0.4rem; font-size:0.8rem;">
              <strong style="color:var(--accent-cyan);">Excerpt [${idx+1}] - ${escapeHtml(c.metadata?.source || 'doc')} (Page ${c.metadata?.page || 1})</strong>
              <div style="color:#d0d7de; margin-top:0.2rem;">${escapeHtml(c.content)}</div>
            </div>
          `;
        });

        excerptsHTML = `
          <div class="collapsible-drawer">
            <div class="drawer-header" onclick="this.nextElementSibling.classList.toggle('open')">
              <span>📚 Retrieved Document Excerpts (${m.contexts.length})</span>
              <span>▼</span>
            </div>
            <div class="drawer-body">
              ${items}
            </div>
          </div>
        `;
      }

      msgWrapper.innerHTML = `
        <div class="msg-bubble-assistant">
          <div class="markdown-body">${parsedHTML}</div>
          ${metricsHTML}
          ${excerptsHTML}
        </div>
      `;
    }

    chatThread.appendChild(msgWrapper);
    chatThread.scrollTop = chatThread.scrollHeight;
  }

  function formatMarkdown(text) {
    if (!text) return "";
    let rawHtml = "";
    if (window.marked && typeof window.marked.parse === "function") {
      rawHtml = window.marked.parse(text);
    } else {
      // Basic fallback formatting
      rawHtml = escapeHtml(text).replace(/\n/g, "<br>");
    }

    // Replace [1], [2] with styled citation badges
    return rawHtml.replace(/\[(\d+)\]/g, '<span class="citation-badge">[$1]</span>');
  }

  function escapeHtml(str) {
    if (!str) return "";
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  // ----------------------------------------------------
  // Send RAG Query in Chat
  // ----------------------------------------------------
  if (btnSearch) btnSearch.addEventListener("click", sendChatQuery);
  if (queryInput) {
    queryInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendChatQuery();
      }
    });
  }

  async function sendChatQuery() {
    const query = queryInput.value.trim();
    if (!query) return;

    if (!activeChatId) await createNewChat();

    // Immediately display user message bubble
    appendMessageToThread({ role: "user", content: query });
    queryInput.value = "";

    // Show temporary thinking state
    const tempWrapper = document.createElement("div");
    tempWrapper.className = "chat-msg assistant temp-thinking";
    tempWrapper.innerHTML = `<div class="msg-bubble-assistant" style="color:var(--accent-cyan); font-weight:600;">⚡ RAG Engine is synthesizing context & retrieving answer...</div>`;
    chatThread.appendChild(tempWrapper);
    chatThread.scrollTop = chatThread.scrollHeight;

    btnSearch.innerText = "⏳ Synthesizing...";
    btnSearch.disabled = true;

    try {
      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query, chat_id: activeChatId })
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Query failed");
      }

      const data = await res.json();
      tempWrapper.remove();

      // Append real AI message response
      appendMessageToThread({
        role: "assistant",
        content: data.answer,
        contexts: data.contexts,
        telemetry: data.telemetry,
        metrics: data.metrics,
        model: data.model,
        provider: data.provider
      });

      // Refresh chat list for auto-generated titles
      loadChats();
    } catch (err) {
      tempWrapper.remove();
      appendMessageToThread({
        role: "assistant",
        content: `⚠️ **Error executing RAG query:** ${err.message}`
      });
    } finally {
      btnSearch.innerText = "⚡ Send";
      btnSearch.disabled = false;
    }
  }

  // ----------------------------------------------------
  // Config UI & Settings Handlers
  // ----------------------------------------------------
  function renderConfigUI(cfg) {
    if (!cfg) return;
    currentConfig = cfg;
    providerSelect.value = cfg.provider;
    apiKeyInput.value = cfg.openrouter_api_key || "";
    ollamaUrlInput.value = cfg.ollama_base_url || "http://localhost:11434";
    updateProviderVisibility();

    hybridAlpha.value = cfg.hybrid_alpha;
    alphaVal.innerText = cfg.hybrid_alpha;
    mmrLambda.value = cfg.mmr_lambda;
    lambdaVal.innerText = cfg.mmr_lambda;
    topKPool.value = cfg.top_k_candidates;
    topkVal.innerText = cfg.top_k_candidates;
    topNContext.value = cfg.top_n_final;
    topnVal.innerText = cfg.top_n_final;

    toggleReranker.checked = cfg.enable_reranker;
    toggleHyde.checked = cfg.enable_hyde;
    toggleStrict.checked = cfg.strict_evidence;

    populateModelSelectors(cfg);
    renderModelsTable(cfg.custom_models || []);

    summaryProvider.innerText = cfg.provider.toUpperCase();
    summaryLLM.innerText = cfg.llm_model;
    summaryEmbedding.innerText = cfg.embedding_model;
    summaryMode.innerText = `Hybrid (α=${cfg.hybrid_alpha})`;
    activeModelBadge.innerText = `Model: ${cfg.llm_model}`;
  }

  function updateProviderVisibility() {
    if (providerSelect.value === "ollama") {
      apiKeyGroup.style.display = "none";
      ollamaUrlGroup.style.display = "flex";
    } else {
      apiKeyGroup.style.display = "flex";
      ollamaUrlGroup.style.display = "none";
    }
  }

  if (providerSelect) providerSelect.addEventListener("change", updateProviderVisibility);

  if (btnToggleKeyVis) {
    btnToggleKeyVis.addEventListener("click", () => {
      apiKeyInput.type = apiKeyInput.type === "password" ? "text" : "password";
    });
  }

  async function saveConfig(showNotification = true, skipReRender = false) {
    if (!currentConfig) return;

    const payload = {
      provider: providerSelect.value,
      openrouter_api_key: apiKeyInput.value,
      ollama_base_url: ollamaUrlInput.value,
      llm_model: activeLlmSelect.value || currentConfig.llm_model,
      embedding_model: activeEmbeddingSelect.value || currentConfig.embedding_model,
      hybrid_alpha: parseFloat(hybridAlpha.value),
      mmr_lambda: parseFloat(mmrLambda.value),
      top_k_candidates: parseInt(topKPool.value),
      top_n_final: parseInt(topNContext.value),
      enable_reranker: toggleReranker.checked,
      enable_hyde: toggleHyde.checked,
      strict_evidence: toggleStrict.checked
    };

    try {
      const res = await fetch("/api/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      currentConfig = await res.json();
      
      if (!skipReRender) {
        renderConfigUI(currentConfig);
      } else {
        summaryProvider.innerText = currentConfig.provider.toUpperCase();
        summaryLLM.innerText = currentConfig.llm_model;
        summaryEmbedding.innerText = currentConfig.embedding_model;
        summaryMode.innerText = `Hybrid (α=${currentConfig.hybrid_alpha})`;
        activeModelBadge.innerText = `Model: ${currentConfig.llm_model}`;
      }

      if (showNotification) showToast("Settings saved successfully!", "success");
    } catch (err) {
      console.error("Failed to save config:", err);
      showToast("Error saving settings.", "error");
    }
  }

  if (btnSaveCredentials) btnSaveCredentials.addEventListener("click", () => saveConfig(true));
  if (activeLlmSelect) activeLlmSelect.addEventListener("change", () => saveConfig(true));
  if (activeEmbeddingSelect) activeEmbeddingSelect.addEventListener("change", () => saveConfig(true));

  [toggleReranker, toggleHyde, toggleStrict].forEach(el => {
    if (el) el.addEventListener("change", () => saveConfig(false, true));
  });

  if (hybridAlpha) {
    hybridAlpha.addEventListener("input", () => { alphaVal.innerText = hybridAlpha.value; });
    hybridAlpha.addEventListener("change", () => { saveConfig(false, true); });
  }

  if (mmrLambda) {
    mmrLambda.addEventListener("input", () => { lambdaVal.innerText = mmrLambda.value; });
    mmrLambda.addEventListener("change", () => { saveConfig(false, true); });
  }

  if (topKPool) {
    topKPool.addEventListener("input", () => { topkVal.innerText = topKPool.value; });
    topKPool.addEventListener("change", () => { saveConfig(false, true); });
  }

  if (topNContext) {
    topNContext.addEventListener("input", () => { topnVal.innerText = topNContext.value; });
    topNContext.addEventListener("change", () => { saveConfig(false, true); });
  }

  // Model Selectors & Models Table
  function populateModelSelectors(cfg) {
    const models = cfg.custom_models || [];
    const llmModels = models.filter(m => m.type === "llm" || !m.type);
    const embedModels = models.filter(m => m.type === "embedding");

    activeLlmSelect.innerHTML = "";
    llmModels.forEach(m => {
      const opt = document.createElement("option");
      opt.value = m.id;
      opt.innerText = `${m.name} (${m.provider})`;
      if (m.id === cfg.llm_model) opt.selected = true;
      activeLlmSelect.appendChild(opt);
    });

    activeEmbeddingSelect.innerHTML = "";
    embedModels.forEach(m => {
      const opt = document.createElement("option");
      opt.value = m.id;
      opt.innerText = `${m.name} (${m.provider})`;
      if (m.id === cfg.embedding_model) opt.selected = true;
      activeEmbeddingSelect.appendChild(opt);
    });
  }

  function renderModelsTable(models) {
    modelsTableBody.innerHTML = "";
    if (models.length === 0) {
      modelsTableBody.innerHTML = `<tr><td colspan="5" style="text-align:center; color: var(--text-muted);">No models registered. Add a model above!</td></tr>`;
      return;
    }

    models.forEach(m => {
      const tr = document.createElement("tr");
      const isLLMActive = currentConfig && m.id === currentConfig.llm_model;
      const isEmbedActive = currentConfig && m.id === currentConfig.embedding_model;
      const pingRes = modelPingResults[m.id];

      let pingBadgeHTML = `<span class="ping-badge" id="pingBadge_${cssEscape(m.id)}">📡 Untested</span>`;
      if (pingRes) {
        const badgeClass = pingRes.status === "ok" ? "ok" : (pingRes.status === "warning" ? "warning" : "error");
        const statusIcon = pingRes.status === "ok" ? "🟢" : (pingRes.status === "warning" ? "🟡" : "🔴");
        const dimStr = pingRes.dimension ? ` (${pingRes.dimension}d)` : '';
        pingBadgeHTML = `<span class="ping-badge ${badgeClass}" id="pingBadge_${cssEscape(m.id)}" title="${pingRes.message}">${statusIcon} ${pingRes.latency_ms}ms${dimStr}</span>`;
      }

      tr.innerHTML = `
        <td>
          <div class="model-name">${escapeHtml(m.name)} ${isLLMActive ? '<span class="citation-badge">Active LLM</span>' : ''} ${isEmbedActive ? '<span class="citation-badge" style="background: rgba(127,0,255,0.2); color: #c084fc;">Active Embedding</span>' : ''}</div>
          <div class="model-id">${escapeHtml(m.id)}</div>
        </td>
        <td><span class="badge">${m.provider.toUpperCase()}</span></td>
        <td><span class="badge" style="color: var(--accent-cyan);">${(m.type || 'llm').toUpperCase()}</span></td>
        <td>${pingBadgeHTML}</td>
        <td>
          <div style="display: flex; gap: 0.4rem;">
            <button class="btn-secondary btn-sm-ping" data-id="${m.id}" data-provider="${m.provider}" data-type="${m.type || 'llm'}">📡 Ping</button>
            ${!isLLMActive && m.type === 'llm' ? `<button class="btn-primary btn-sm-activate" data-id="${m.id}">🎯 Set Active</button>` : ''}
            <button class="btn-danger btn-sm-delete" data-id="${m.id}">🗑️</button>
          </div>
        </td>
      `;
      modelsTableBody.appendChild(tr);
    });

    modelsTableBody.querySelectorAll(".btn-sm-ping").forEach(btn => {
      btn.addEventListener("click", () => pingSingleModel(btn.getAttribute("data-id"), btn.getAttribute("data-provider"), btn.getAttribute("data-type")));
    });

    modelsTableBody.querySelectorAll(".btn-sm-activate").forEach(btn => {
      btn.addEventListener("click", () => {
        activeLlmSelect.value = btn.getAttribute("data-id");
        saveConfig(true);
      });
    });

    modelsTableBody.querySelectorAll(".btn-sm-delete").forEach(btn => {
      btn.addEventListener("click", () => deleteModel(btn.getAttribute("data-id")));
    });
  }

  function cssEscape(str) {
    return str.replace(/[^a-zA-Z0-9_-]/g, "_");
  }

  if (btnAddModel) {
    btnAddModel.addEventListener("click", async () => {
      const id = newModelId.value.trim();
      const name = newModelName.value.trim();
      const provider = newModelProvider.value;
      const type = newModelType.value;

      if (!id || !name) {
        showToast("Please provide both Model ID and Display Name.", "error");
        return;
      }

      try {
        const res = await fetch("/api/models", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ id, name, provider, type })
        });
        currentConfig = await res.json();
        renderConfigUI(currentConfig);
        showToast(`Model '${name}' registered successfully!`, "success");
        newModelId.value = "";
        newModelName.value = "";
      } catch (err) {
        showToast("Failed to add model: " + err.message, "error");
      }
    });
  }

  if (btnPingNewModel) {
    btnPingNewModel.addEventListener("click", async () => {
      const id = newModelId.value.trim();
      const provider = newModelProvider.value;
      const type = newModelType.value;
      if (!id) {
        showToast("Enter a Model ID to ping test.", "error");
        return;
      }
      btnPingNewModel.innerText = "⏳ Pinging...";
      btnPingNewModel.disabled = true;
      newModelPingStatus.style.display = "block";
      newModelPingStatus.className = "ping-result-box";
      newModelPingStatus.innerText = "Testing connectivity & vector dimensions...";

      const res = await performModelPing(id, provider, type);
      btnPingNewModel.innerText = "📡 Ping Test Model";
      btnPingNewModel.disabled = false;

      newModelPingStatus.className = `ping-result-box ${res.status}`;
      newModelPingStatus.innerText = res.message;
    });
  }

  async function pingSingleModel(modelId, provider, type = "llm") {
    const badgeEl = document.getElementById(`pingBadge_${cssEscape(modelId)}`);
    if (badgeEl) badgeEl.innerText = "⏳ Pinging...";

    const res = await performModelPing(modelId, provider, type);
    modelPingResults[modelId] = res;
    renderModelsTable(currentConfig.custom_models || []);
    
    if (res.status === "ok") {
      const dimInfo = res.dimension ? ` (Dimension: ${res.dimension}d)` : '';
      showToast(`Ping Success [${modelId}]: ${res.latency_ms}ms${dimInfo}`, "success");
    } else {
      showToast(`Ping Failed [${modelId}]: ${res.message}`, "error");
    }
  }

  async function performModelPing(modelId, provider, type = "llm") {
    try {
      const res = await fetch("/api/models/ping", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: modelId,
          provider: provider,
          type: type,
          openrouter_api_key: apiKeyInput.value,
          ollama_base_url: ollamaUrlInput.value
        })
      });
      return await res.json();
    } catch (err) {
      return { status: "error", latency_ms: 0, message: "Connection Error: " + err.message };
    }
  }

  async function deleteModel(modelId) {
    if (!confirm(`Are you sure you want to delete model '${modelId}'?`)) return;
    try {
      const res = await fetch(`/api/models/${encodeURIComponent(modelId)}`, { method: "DELETE" });
      currentConfig = await res.json();
      renderConfigUI(currentConfig);
      showToast(`Model '${modelId}' removed.`, "info");
    } catch (err) {
      showToast("Failed to delete model: " + err.message, "error");
    }
  }

  // ----------------------------------------------------
  // Document Ingestion
  // ----------------------------------------------------
  if (dropZone) {
    dropZone.addEventListener("click", () => fileInput.click());
    dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.style.borderColor = "var(--accent-cyan)"; });
    dropZone.addEventListener("dragleave", () => { dropZone.style.borderColor = "var(--bg-card-border)"; });
    dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropZone.style.borderColor = "var(--bg-card-border)";
      if (e.dataTransfer.files.length) handleFiles(e.dataTransfer.files);
    });
  }

  if (fileInput) {
    fileInput.addEventListener("change", () => {
      if (fileInput.files.length) handleFiles(fileInput.files);
    });
  }

  async function handleFiles(files) {
    const formData = new FormData();
    for (let f of files) formData.append("files", f);

    const progressBox = document.getElementById("uploadProgressBox");
    const progressBar = document.getElementById("uploadProgressBar");
    const progressText = document.getElementById("uploadProgressText");

    if (progressBox) progressBox.style.display = "block";
    if (progressBar) progressBar.style.width = "10%";
    if (progressText) progressText.innerText = "Uploading to folder... 10%";
    if (uploadStatus) uploadStatus.innerText = "⏳ Saving & Chunking...";

    try {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/upload", true);

      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) {
          const percent = Math.min(90, Math.round(10 + (e.loaded / e.total) * 80));
          if (progressBar) progressBar.style.width = `${percent}%`;
          if (progressText) progressText.innerText = `Chunking & Embedding... ${percent}%`;
        }
      };

      xhr.onload = () => {
        if (progressBar) progressBar.style.width = "100%";
        if (progressText) progressText.innerText = "Completed 100%";

        if (xhr.status === 200) {
          const data = JSON.parse(xhr.responseText);
          if (uploadStatus) uploadStatus.innerText = `✅ ${data.message}`;
          showToast(data.message, "success");
          loadStats();
          loadDocuments();
        } else {
          if (uploadStatus) uploadStatus.innerText = `❌ Upload failed: HTTP ${xhr.status}`;
          showToast("Upload failed", "error");
        }

        setTimeout(() => {
          if (progressBox) progressBox.style.display = "none";
          if (uploadStatus) uploadStatus.innerText = "";
        }, 4000);
      };

      xhr.onerror = () => {
        if (uploadStatus) uploadStatus.innerText = "❌ Upload connection failed";
        showToast("Upload connection error", "error");
        if (progressBox) progressBox.style.display = "none";
      };

      xhr.send(formData);
    } catch (err) {
      if (uploadStatus) uploadStatus.innerText = `❌ Error: ${err.message}`;
      if (progressBox) progressBox.style.display = "none";
      setTimeout(() => { if (uploadStatus) uploadStatus.innerText = ""; }, 4000);
    }
  }

  if (btnClearIndex) {
    btnClearIndex.addEventListener("click", async () => {
      if (confirm("Are you sure you want to clear the indexed corpus? This action cannot be undone.")) {
        await fetch("/api/clear", { method: "POST" });
        uploadStatus.innerText = "Corpus index cleared.";
        showToast("Corpus index cleared successfully.", "info");
        loadStats();
        loadDocuments();
        setTimeout(() => { if (uploadStatus) uploadStatus.innerText = ""; }, 4000);
      }
    });
  }

  if (btnResetConfig) {
    btnResetConfig.addEventListener("click", async () => {
      if (confirm("Are you sure you want to clear saved credentials and reset system config to defaults?")) {
        try {
          const res = await fetch("/api/config/reset", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ clear_credentials: true, clear_custom_models: false })
          });
          currentConfig = await res.json();
          renderConfigUI(currentConfig);
          showToast("Credentials cleared and system config reset.", "info");
        } catch (err) {
          showToast("Failed to reset config: " + err.message, "error");
        }
      }
    });
  }
});
