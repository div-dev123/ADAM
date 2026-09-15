/**
 * ADAM - Adaptive Memory Management Research Prototype Client
 * Manages Chat, Pipeline Visualization, Memory Tiers, Lifecycle Audit, and Metrics.
 */

(() => {
    'use strict';

    // Application State
    const state = {
        userId: 'user-1',
        activeView: 'chat',
        chatHistory: [],
        turnCount: 0,
        memories: [],
        selectedMemoryId: null,
        metrics: null,
        systemStatus: null,
        isProcessing: false,
    };

    // DOM Elements
    const elements = {
        userSelect: document.getElementById('user-select'),
        navTabs: document.querySelectorAll('.nav-tab'),
        viewPanels: document.querySelectorAll('.view-panel'),
        headerMemoryCount: document.getElementById('header-memory-count'),
        ollamaStatusDot: document.getElementById('ollama-status-dot'),
        ollamaStatusText: document.getElementById('ollama-status-text'),
        btnResetDb: document.getElementById('btn-reset-db'),

        // Chat View
        chatMessages: document.getElementById('chat-messages'),
        chatForm: document.getElementById('chat-form'),
        chatTextarea: document.getElementById('chat-textarea'),
        btnSendMessage: document.getElementById('btn-send-message'),
        chatTurnTiming: document.getElementById('chat-turn-timing'),
        turnInspectorContent: document.getElementById('turn-inspector-content'),
        traceTurnId: document.getElementById('trace-turn-id'),
        pipelineBanner: document.getElementById('pipeline-banner'),
        pipelineOverallStatus: document.getElementById('pipeline-overall-status'),
        stepCards: {
            extract: document.getElementById('step-extract'),
            consolidation: document.getElementById('step-consolidation'),
            retrieval: document.getElementById('step-retrieval'),
            llm: document.getElementById('step-llm'),
            response: document.getElementById('step-response'),
        },

        // Dashboard View
        memorySearchInput: document.getElementById('memory-search-input'),
        btnClearSearch: document.getElementById('btn-clear-search'),
        filterTier: document.getElementById('filter-tier'),
        filterImportance: document.getElementById('filter-importance'),
        btnRefreshMemories: document.getElementById('btn-refresh-memories'),
        btnAddMemoryManual: document.getElementById('btn-add-memory-manual'),
        btnEmptyDbDashboard: document.getElementById('btn-empty-db-dashboard'),
        cardsContainers: {
            WORKING: document.getElementById('cards-working'),
            SHORT_TERM: document.getElementById('cards-short-term'),
            LONG_TERM: document.getElementById('cards-long-term'),
            ARCHIVE: document.getElementById('cards-archive'),
        },
        tierCounts: {
            WORKING: document.getElementById('count-working'),
            SHORT_TERM: document.getElementById('count-short-term'),
            LONG_TERM: document.getElementById('count-long-term'),
            ARCHIVE: document.getElementById('count-archive'),
        },

        // Lifecycle Explorer
        tracerMemorySelect: document.getElementById('tracer-memory-select'),
        tracerTimelineContainer: document.getElementById('tracer-timeline-container'),

        // Consolidation Feed
        auditFeedList: document.getElementById('audit-feed-list'),
        btnRefreshHistory: document.getElementById('btn-refresh-history'),

        // Metrics Panel
        metricTotalMemories: document.getElementById('metric-total-memories'),
        metricAvgImportance: document.getElementById('metric-avg-importance'),
        metricTotalConsolidations: document.getElementById('metric-total-consolidations'),
        metricCompressedCount: document.getElementById('metric-compressed-count'),
        barWorking: document.getElementById('bar-working'),
        valWorking: document.getElementById('val-working'),
        barShortTerm: document.getElementById('bar-short-term'),
        valShortTerm: document.getElementById('val-short-term'),
        barLongTerm: document.getElementById('bar-long-term'),
        valLongTerm: document.getElementById('val-long-term'),
        barArchive: document.getElementById('bar-archive'),
        valArchive: document.getElementById('val-archive'),
        countDuplicate: document.getElementById('count-duplicate'),
        countRelated: document.getElementById('count-related'),
        countContradictory: document.getElementById('count-contradictory'),
        countCompressedOp: document.getElementById('count-compressed-op'),
        sysOllamaHost: document.getElementById('sys-ollama-host'),
        sysOllamaModel: document.getElementById('sys-ollama-model'),
        sysEmbeddingModel: document.getElementById('sys-embedding-model'),
        sysDatabasePath: document.getElementById('sys-database-path'),
        sysSimThreshold: document.getElementById('sys-sim-threshold'),
        sysCandidateLimit: document.getElementById('sys-candidate-limit'),

        // Modals
        modalAddMemory: document.getElementById('modal-add-memory'),
        manualMemoryUser: document.getElementById('manual-memory-user'),
        manualMemoryContent: document.getElementById('manual-memory-content'),
        btnSubmitManualMemory: document.getElementById('btn-submit-manual-memory'),
        modalMemoryDetails: document.getElementById('modal-memory-details'),
        memoryDetailsModalBody: document.getElementById('memory-details-modal-body'),
        modalTierTransition: document.getElementById('modal-tier-transition'),
        transitionMemorySummary: document.getElementById('transition-memory-summary'),
        transitionMemoryId: document.getElementById('transition-memory-id'),
        targetTierSelect: document.getElementById('target-tier-select'),
        btnSubmitTransition: document.getElementById('btn-submit-transition'),
        modalResetConfirm: document.getElementById('modal-reset-confirm'),
        btnConfirmReset: document.getElementById('btn-confirm-reset'),
        toastContainer: document.getElementById('toast-container'),
    };

    // =========================================================================
    // API Helper Functions
    // =========================================================================

    async function apiRequest(endpoint, options = {}) {
        try {
            const response = await fetch(endpoint, {
                headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
                ...options,
            });
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: response.statusText }));
                throw new Error(errorData.detail || `HTTP ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error(`API Error on ${endpoint}:`, error);
            throw error;
        }
    }

    // =========================================================================
    // UI Helpers & Formatters
    // =========================================================================

    function showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.textContent = message;
        elements.toastContainer.appendChild(toast);
        setTimeout(() => {
            toast.style.opacity = '0';
            setTimeout(() => toast.remove(), 300);
        }, 3500);
    }

    function scrollToBottom() {
        if (elements.chatMessages) {
            requestAnimationFrame(() => {
                elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
            });
        }
    }

    function formatTime(isoString) {
        if (!isoString) return '-';
        const date = new Date(isoString);
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    }

    function formatDate(isoString) {
        if (!isoString) return '-';
        const date = new Date(isoString);
        return date.toLocaleDateString([], { month: 'short', day: 'numeric', year: 'numeric' }) + ' ' + formatTime(isoString);
    }

    function getImportanceCategory(score) {
        if (score >= 0.70) return { label: 'High', class: 'badge-imp-high', fillClass: 'fill-high' };
        if (score > 0.30) return { label: 'Medium', class: 'badge-imp-med', fillClass: 'fill-med' };
        return { label: 'Low', class: 'badge-imp-low', fillClass: 'fill-low' };
    }

    function getTierBadge(tier) {
        const tierNorm = (tier || 'WORKING').toUpperCase();
        switch (tierNorm) {
            case 'WORKING': return { label: 'WORKING', class: 'badge-working' };
            case 'SHORT_TERM': return { label: 'SHORT-TERM', class: 'badge-short-term' };
            case 'LONG_TERM': return { label: 'LONG-TERM', class: 'badge-long-term' };
            case 'ARCHIVE': return { label: 'ARCHIVE', class: 'badge-archive' };
            default: return { label: tierNorm, class: 'badge-subtle' };
        }
    }

    function getActionBadge(action) {
        const act = (action || 'NEW').toUpperCase();
        switch (act) {
            case 'NEW': return { label: 'NEW (Saved)', class: 'badge-new' };
            case 'DUPLICATE': return { label: 'DUPLICATE (Touch)', class: 'badge-duplicate' };
            case 'RELATED': return { label: 'RELATED (Merged)', class: 'badge-related' };
            case 'CONTRADICTORY': return { label: 'CONTRADICTORY (Updated)', class: 'badge-contradictory' };
            case 'COMPRESSED': return { label: 'COMPRESSED', class: 'badge-compression' };
            case 'EVALUATED': return { label: 'EVALUATED', class: 'badge-subtle' };
            case 'FILLER': return { label: 'IGNORED (Filler)', class: 'badge-subtle' };
            default: return { label: act, class: 'badge-subtle' };
        }
    }

    // =========================================================================
    // View Navigation & State
    // =========================================================================

    function switchView(viewName) {
        state.activeView = viewName;
        elements.navTabs.forEach(tab => {
            tab.classList.toggle('active', tab.dataset.view === viewName);
        });
        elements.viewPanels.forEach(panel => {
            panel.classList.toggle('active', panel.id === `view-${viewName}`);
        });

        // Trigger view-specific loads
        if (viewName === 'chat') scrollToBottom();
        if (viewName === 'dashboard') loadMemories();
        if (viewName === 'lifecycle') loadLifecycleTracer();
        if (viewName === 'consolidation') loadConsolidationFeed();
        if (viewName === 'metrics') loadMetrics();
    }

    // =========================================================================
    // System Status & Health
    // =========================================================================

    async function checkSystemStatus() {
        try {
            const status = await apiRequest('/system/status');
            state.systemStatus = status;

            if (status.ollama && status.ollama.connected) {
                elements.ollamaStatusDot.className = 'status-dot online';
                elements.ollamaStatusText.textContent = `Ollama Online (${status.ollama.model})`;
            } else {
                elements.ollamaStatusDot.className = 'status-dot offline';
                elements.ollamaStatusText.textContent = `Ollama Offline (${status.ollama.model})`;
            }

            elements.headerMemoryCount.textContent = status.database.total_memories;

            // System settings in metrics view
            if (elements.sysOllamaHost) elements.sysOllamaHost.textContent = status.ollama.host;
            if (elements.sysOllamaModel) elements.sysOllamaModel.textContent = status.ollama.model;
            if (elements.sysEmbeddingModel) elements.sysEmbeddingModel.textContent = status.embedding_model;
            if (elements.sysDatabasePath) elements.sysDatabasePath.textContent = status.database.path;
            if (elements.sysSimThreshold) elements.sysSimThreshold.textContent = `≥ ${status.config.consolidation_min_similarity}`;
            if (elements.sysCandidateLimit) elements.sysCandidateLimit.textContent = `${status.config.consolidation_candidate_limit} candidates max`;
        } catch (error) {
            elements.ollamaStatusDot.className = 'status-dot offline';
            elements.ollamaStatusText.textContent = 'Backend Offline';
        }
    }

    // =========================================================================
    // View 1: Chat Pipeline & Inspector
    // =========================================================================

    function setPipelineSteps(activeStepName = null, completedSteps = []) {
        Object.keys(elements.stepCards).forEach(key => {
            const card = elements.stepCards[key];
            card.classList.remove('active-step', 'completed-step');
            if (completedSteps.includes(key)) {
                card.classList.add('completed-step');
            } else if (key === activeStepName) {
                card.classList.add('active-step');
            }
        });
    }

    async function handleChatSubmit() {
        const message = elements.chatTextarea.value.trim();
        if (!message || state.isProcessing) return;

        state.isProcessing = true;
        elements.btnSendMessage.disabled = true;
        elements.chatTextarea.value = '';
        state.turnCount += 1;
        const currentTurn = state.turnCount;

        const startTime = performance.now();
        elements.pipelineOverallStatus.textContent = 'Running Memory Write & Retrieval Pipeline...';
        setPipelineSteps('extract', []);

        // Append Temporary User Message to UI
        const turnGroup = document.createElement('div');
        turnGroup.className = 'chat-turn-group';
        turnGroup.id = `turn-group-${currentTurn}`;

        const userRow = document.createElement('div');
        userRow.className = 'chat-message-row user-row';
        userRow.innerHTML = `
            <div class="avatar user-avatar">U</div>
            <div class="message-bubble-wrapper">
                <div class="message-bubble">${escapeHtml(message)}</div>
                <div class="message-badges-bar" id="user-badges-${currentTurn}">
                    <span class="badge badge-subtle">Processing write path...</span>
                </div>
            </div>
        `;
        turnGroup.appendChild(userRow);
        elements.chatMessages.appendChild(turnGroup);
        scrollToBottom();

        try {
            setPipelineSteps('consolidation', ['extract']);

            const turnResult = await apiRequest('/chat', {
                method: 'POST',
                body: JSON.stringify({
                    user_id: state.userId,
                    message: message,
                    top_k: 5,
                    chat_history: state.chatHistory,
                }),
            });

            const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);
            elements.chatTurnTiming.textContent = `Turn #${currentTurn} completed in ${elapsed}s`;
            elements.pipelineOverallStatus.textContent = `Turn #${currentTurn} complete (${elapsed}s)`;
            setPipelineSteps(null, ['extract', 'consolidation', 'retrieval', 'llm', 'response']);

            // Update User Badges
            renderUserMessageBadges(currentTurn, turnResult.user_memory);

            // Append Assistant Message
            renderAssistantMessage(turnGroup, currentTurn, turnResult.response, turnResult.assistant_memory, turnResult.retrieved_memories);

            // Update Turn Inspector on the right
            renderTurnInspector(currentTurn, message, turnResult);

            // Update conversation history
            state.chatHistory.push({ role: 'user', content: message });
            state.chatHistory.push({ role: 'assistant', content: turnResult.response });

            // Refresh background metrics
            checkSystemStatus();
            if (state.activeView === 'dashboard') loadMemories();
        } catch (error) {
            console.error('Chat Turn Error:', error);
            elements.pipelineOverallStatus.textContent = `Error: ${error.message}`;
            setPipelineSteps(null, []);

            const errorRow = document.createElement('div');
            errorRow.className = 'chat-message-row assistant-row';
            errorRow.innerHTML = `
                <div class="avatar assistant-avatar">!</div>
                <div class="message-bubble-wrapper">
                    <div class="message-bubble" style="border-color: #f43f5e; color: #fb7185;">
                        <strong>ADAM Pipeline Error:</strong> ${escapeHtml(error.message)}
                    </div>
                </div>
            `;
            turnGroup.appendChild(errorRow);
            showToast(`Turn error: ${error.message}`, 'danger');
        } finally {
            state.isProcessing = false;
            elements.btnSendMessage.disabled = false;
            scrollToBottom();
        }
    }

    function renderUserMessageBadges(turnId, userMemory) {
        const container = document.getElementById(`user-badges-${turnId}`);
        if (!container || !userMemory) return;

        if (!userMemory.is_stored || userMemory.action === 'FILLER') {
            container.innerHTML = `
                <span class="badge badge-subtle" title="Trivial greeting or conversational filler was not stored in memory">
                    🚫 Greeting / Filler (Ignored)
                </span>
                <button class="btn-details-toggle" onclick="window.ADAM.toggleDetails('user-details-${turnId}')">
                    <span>Details</span> ▾
                </button>
                <div class="adam-details-drawer" id="user-details-${turnId}">
                    <div class="details-section-title">Filter Reason</div>
                    <p style="color: #cbd5e1; margin-bottom: 0.5rem;">${escapeHtml(userMemory.decision_reason || 'Greeting or conversational small talk was excluded from storage.')}</p>
                </div>
            `;
            return;
        }

        const impCat = getImportanceCategory(userMemory.importance_score);
        const tierBadge = getTierBadge(userMemory.tier);
        const actBadge = getActionBadge(userMemory.action);
        const compLevel = userMemory.compression_level || 0;

        container.innerHTML = `
            <span class="badge ${impCat.class}" title="Intrinsic Information Value [0-1]">
                ⭐ Importance: ${userMemory.importance_score.toFixed(2)} (${impCat.label})
            </span>
            <span class="badge ${tierBadge.class}" title="Storage Tier">
                🗄️ ${tierBadge.label}
            </span>
            <span class="badge ${actBadge.class}" title="Consolidation Action">
                ⚖️ ${actBadge.label}
            </span>
            ${compLevel > 0 ? `<span class="badge badge-compression">📦 L${compLevel} Compressed</span>` : ''}
            <button class="btn-details-toggle" onclick="window.ADAM.toggleDetails('user-details-${turnId}')">
                <span>Details</span> ▾
            </button>
            <div class="adam-details-drawer" id="user-details-${turnId}">
                <div class="details-section-title">Consolidation Reasoning</div>
                <p style="color: #cbd5e1; margin-bottom: 0.5rem;">${escapeHtml(userMemory.decision_reason || 'Standard storage classification')}</p>
                
                ${userMemory.score_breakdown && Object.keys(userMemory.score_breakdown).length > 0 ? `
                    <div class="details-section-title">Multi-Signal Importance Breakdown</div>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 0.35rem; margin-bottom: 0.5rem;">
                        ${Object.entries(userMemory.score_breakdown).map(([name, s]) => `
                            <div style="background: rgba(15, 23, 42, 0.6); padding: 0.3rem 0.5rem; border-radius: 4px; border-left: 2px solid #38bdf8; font-size: 0.72rem;">
                                <div style="font-weight: 600; text-transform: capitalize; color: #94a3b8;">${name} (${(s.weight * 100).toFixed(0)}%)</div>
                                <div style="color: #38bdf8; font-family: monospace; font-size: 0.8rem;">Val: ${s.value} &rarr; +${s.contribution}</div>
                                <div style="color: #64748b; font-size: 0.65rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="${escapeHtml(s.reason || '')}">${escapeHtml(s.reason || '')}</div>
                            </div>
                        `).join('')}
                    </div>
                ` : ''}

                ${userMemory.old_content ? `
                    <div class="details-section-title">Content Evolution</div>
                    <div class="timeline-event-diff" style="margin-bottom: 0.5rem;">
                        <div class="diff-old">- Prior: ${escapeHtml(userMemory.old_content)}</div>
                        <div class="diff-new">+ Updated: ${escapeHtml(userMemory.merged_content || userMemory.memory?.content || '')}</div>
                    </div>
                ` : ''}

                <div class="details-section-title">Similar Candidates Evaluated (${userMemory.candidates.length})</div>
                ${userMemory.candidates.length > 0 ? userMemory.candidates.map(c => `
                    <div class="details-candidate-item">
                        <span>${escapeHtml(c.content)}</span>
                        <span style="color:#38bdf8;">Sim: ${c.similarity}</span>
                    </div>
                `).join('') : '<div style="color:var(--text-muted);">No candidates exceeded cosine similarity threshold.</div>'}
            </div>
        `;
    }

    function renderAssistantMessage(turnGroup, turnId, responseText, assistantMemory, retrievedMemories) {
        const assistantRow = document.createElement('div');
        assistantRow.className = 'chat-message-row assistant-row';

        const retrievedCount = retrievedMemories ? retrievedMemories.length : 0;
        const isStored = assistantMemory && assistantMemory.is_stored;
        const impScore = assistantMemory?.importance_score || 0;
        const impCat = getImportanceCategory(impScore);
        const tierBadge = getTierBadge(assistantMemory?.tier || 'WORKING');
        const actBadge = getActionBadge(assistantMemory?.action || 'NEW');

        assistantRow.innerHTML = `
            <div class="avatar assistant-avatar">A</div>
            <div class="message-bubble-wrapper">
                <div class="message-bubble">${escapeHtml(responseText)}</div>
                <div class="message-badges-bar">
                    ${isStored ? `
                        <span class="badge ${impCat.class}">
                            ⭐ Stored: ${impScore.toFixed(2)} (${impCat.label})
                        </span>
                        <span class="badge ${tierBadge.class}">
                            🗄️ ${tierBadge.label}
                        </span>
                        <span class="badge ${actBadge.class}">
                            ⚖️ ${actBadge.label}
                        </span>
                    ` : `
                        <span class="badge badge-subtle" title="LLM pleasantry or boilerplate was not stored">
                            🚫 Boilerplate (Ignored)
                        </span>
                    `}
                    <span class="badge badge-subtle" title="Retrieved memories injected into context">
                        🔍 ${retrievedCount} Injected ${retrievedCount === 1 ? 'Memory' : 'Memories'}
                    </span>
                    <button class="btn-details-toggle" onclick="window.ADAM.toggleDetails('asst-details-${turnId}')">
                        <span>Context & Storage</span> ▾
                    </button>
                    <div class="adam-details-drawer" id="asst-details-${turnId}">
                        <div class="details-section-title">Assistant Memory Decision</div>
                        <p style="color: #cbd5e1; margin-bottom: 0.5rem;">${escapeHtml(assistantMemory?.decision_reason || (isStored ? 'Response contained substantive information and was stored.' : 'Boilerplate filtered.'))}</p>
                        <div class="details-section-title">Retrieved Memories for Query Context (${retrievedCount})</div>
                        ${retrievedCount > 0 ? retrievedMemories.map(r => `
                            <div class="details-candidate-item">
                                <div>
                                    <span class="badge ${getTierBadge(r.memory?.tier).class}" style="font-size:0.6rem; padding:0.1rem 0.3rem;">${r.memory?.tier}</span>
                                    <span>${escapeHtml(r.memory?.content || '')}</span>
                                </div>
                                <span style="color:#a855f7;">Sim: ${r.similarity}</span>
                            </div>
                        `).join('') : '<div style="color:var(--text-muted);">No relevant memories retrieved.</div>'}
                    </div>
                </div>
            </div>
        `;
        turnGroup.appendChild(assistantRow);
        scrollToBottom();
    }

    function renderTurnInspector(turnId, userQuery, turnResult) {
        elements.traceTurnId.textContent = `Turn #${turnId}`;
        const uMem = turnResult.user_memory;
        const aMem = turnResult.assistant_memory;
        const retrieved = turnResult.retrieved_memories || [];

        elements.turnInspectorContent.innerHTML = `
            <!-- User Extraction & Scoring Block -->
            <div style="margin-bottom: 1rem;">
                <div class="details-section-title">1. User Message Write Path</div>
                <div style="background: rgba(0,0,0,0.25); padding: 0.65rem; border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
                    ${uMem.is_stored ? `
                        <div style="display:flex; justify-content:space-between; margin-bottom:0.25rem;">
                            <span style="font-size:0.75rem; color:var(--text-muted);">Importance:</span>
                            <span style="font-size:0.8rem; font-weight:700; color:#c084fc;">${uMem.importance_score.toFixed(4)}</span>
                        </div>
                        <div style="display:flex; justify-content:space-between; margin-bottom:0.25rem;">
                            <span style="font-size:0.75rem; color:var(--text-muted);">Storage Tier:</span>
                            <span class="badge ${getTierBadge(uMem.tier).class}">${uMem.tier}</span>
                        </div>
                        <div style="display:flex; justify-content:space-between;">
                            <span style="font-size:0.75rem; color:var(--text-muted);">Action:</span>
                            <span class="badge ${getActionBadge(uMem.action).class}">${uMem.action}</span>
                        </div>
                    ` : `
                        <div style="font-size:0.75rem; color:#94a3b8;">
                            <strong>Filtered:</strong> ${escapeHtml(uMem.decision_reason || 'Greeting or conversational noise')}
                        </div>
                    `}
                </div>
            </div>

            <!-- Assistant Memory Storage Block -->
            <div style="margin-bottom: 1rem;">
                <div class="details-section-title">2. Assistant Response Write Path</div>
                <div style="background: rgba(0,0,0,0.25); padding: 0.65rem; border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
                    ${aMem && aMem.is_stored ? `
                        <div style="display:flex; justify-content:space-between; margin-bottom:0.25rem;">
                            <span style="font-size:0.75rem; color:var(--text-muted);">Importance:</span>
                            <span style="font-size:0.8rem; font-weight:700; color:#38bdf8;">${aMem.importance_score.toFixed(4)}</span>
                        </div>
                        <div style="display:flex; justify-content:space-between; margin-bottom:0.25rem;">
                            <span style="font-size:0.75rem; color:var(--text-muted);">Storage Tier:</span>
                            <span class="badge ${getTierBadge(aMem.tier).class}">${aMem.tier}</span>
                        </div>
                        <div style="display:flex; justify-content:space-between;">
                            <span style="font-size:0.75rem; color:var(--text-muted);">Action:</span>
                            <span class="badge ${getActionBadge(aMem.action).class}">${aMem.action}</span>
                        </div>
                    ` : `
                        <div style="font-size:0.75rem; color:#94a3b8;">
                            <strong>Boilerplate Ignored:</strong> ${escapeHtml(aMem?.decision_reason || 'No substantive information to persist')}
                        </div>
                    `}
                </div>
            </div>

            <!-- Retrieval Context Block -->
            <div style="margin-bottom: 1rem;">
                <div class="details-section-title">3. Retrieved Memories for LLM Context (${retrieved.length})</div>
                <div style="display:flex; flex-direction:column; gap:0.4rem;">
                    ${retrieved.length > 0 ? retrieved.map(r => `
                        <div style="background: rgba(0,0,0,0.25); padding: 0.5rem; border-radius: var(--radius-sm); border: 1px solid var(--border-subtle); font-size:0.75rem;">
                            <div style="display:flex; justify-content:space-between; margin-bottom:0.2rem;">
                                <span class="badge ${getTierBadge(r.memory?.tier).class}" style="font-size:0.65rem;">${r.memory?.tier}</span>
                                <span style="color:#a855f7; font-family:var(--font-mono); font-size:0.7rem;">Cosine Sim: ${r.similarity}</span>
                            </div>
                            <div style="color:var(--text-primary); font-size:0.78rem;">${escapeHtml(r.memory?.content || '')}</div>
                        </div>
                    `).join('') : '<div style="color:var(--text-muted); font-size:0.75rem;">No memory context needed.</div>'}
                </div>
            </div>
        `;
    }

    // =========================================================================
    // View 2: Memory Dashboard (4 Tiers)
    // =========================================================================

    async function loadMemories() {
        const query = elements.memorySearchInput ? elements.memorySearchInput.value.trim() : '';
        const tier = elements.filterTier ? elements.filterTier.value : '';
        const impFilter = elements.filterImportance ? elements.filterImportance.value : '';

        try {
            let url = `/memories?user_id=${encodeURIComponent(state.userId)}`;
            if (tier) url += `&tier=${encodeURIComponent(tier)}`;
            if (query) url += `&search=${encodeURIComponent(query)}`;

            const memories = await apiRequest(url);
            state.memories = memories;

            // Apply client importance filter if set
            let filtered = memories;
            if (impFilter === 'high') filtered = memories.filter(m => m.importance_score >= 0.70);
            if (impFilter === 'medium') filtered = memories.filter(m => m.importance_score > 0.30 && m.importance_score < 0.70);
            if (impFilter === 'low') filtered = memories.filter(m => m.importance_score <= 0.30);

            renderMemoryDashboard(filtered);
            elements.headerMemoryCount.textContent = memories.length;
        } catch (error) {
            console.error('Failed to load memories:', error);
            showToast(`Could not load memories: ${error.message}`, 'danger');
        }
    }

    function renderMemoryDashboard(memories) {
        // Group by tier
        const grouped = {
            WORKING: [],
            SHORT_TERM: [],
            LONG_TERM: [],
            ARCHIVE: [],
        };

        memories.forEach(m => {
            const t = (m.tier || 'WORKING').toUpperCase();
            if (grouped[t]) grouped[t].push(m);
            else grouped.WORKING.push(m);
        });

        // Update counts and render cards in each column
        Object.keys(grouped).forEach(tierKey => {
            const list = grouped[tierKey];
            if (elements.tierCounts[tierKey]) {
                elements.tierCounts[tierKey].textContent = list.length;
            }
            const container = elements.cardsContainers[tierKey];
            if (!container) return;

            if (list.length === 0) {
                container.innerHTML = `<div class="tier-empty-slot">No memories in ${tierKey} tier</div>`;
            } else {
                container.innerHTML = list.map(m => createMemoryCardHtml(m)).join('');
            }
        });
    }

    function createMemoryCardHtml(memory) {
        const impCat = getImportanceCategory(memory.importance_score);
        const tierBadge = getTierBadge(memory.tier);
        const compLevel = memory.compression_level || 0;

        return `
            <div class="memory-card" onclick="window.ADAM.inspectMemory('${memory.memory_id}')">
                <div class="memory-card-header">
                    <span class="badge ${impCat.class}">⭐ ${memory.importance_score.toFixed(2)} (${impCat.label})</span>
                    ${compLevel > 0 ? `<span class="badge badge-compression">📦 L${compLevel}</span>` : ''}
                </div>
                <div class="memory-card-content">${escapeHtml(memory.content)}</div>
                <div class="memory-card-meta">
                    <div class="importance-meter" title="Importance: ${memory.importance_score.toFixed(2)}">
                        <div class="importance-meter-track">
                            <div class="importance-meter-fill ${impCat.fillClass}" style="width: ${Math.round(memory.importance_score * 100)}%"></div>
                        </div>
                    </div>
                    <span>Accesses: <strong>${memory.access_count}</strong></span>
                    <span>${formatTime(memory.last_accessed || memory.created_at)}</span>
                </div>
                <div class="card-actions-row" onclick="event.stopPropagation()">
                    <button class="btn-card-action" onclick="window.ADAM.inspectMemory('${memory.memory_id}')">📜 History</button>
                    <button class="btn-card-action" onclick="window.ADAM.openTierTransitionModal('${memory.memory_id}', '${escapeHtml(memory.content)}', '${memory.tier}')">🔄 Move</button>
                    <button class="btn-card-action" style="color:#f43f5e;" onclick="window.ADAM.deleteMemory('${memory.memory_id}')">🗑️</button>
                </div>
            </div>
        `;
    }

    // =========================================================================
    // View 3: Memory Lifecycle & State Machine Explorer
    // =========================================================================

    async function loadLifecycleTracer() {
        try {
            const memories = await apiRequest(`/memories?user_id=${encodeURIComponent(state.userId)}`);
            state.memories = memories;

            elements.tracerMemorySelect.innerHTML = `
                <option value="">-- Select a Memory (${memories.length} Available) --</option>
                ${memories.map(m => `
                    <option value="${m.memory_id}" ${state.selectedMemoryId === m.memory_id ? 'selected' : ''}>
                        [${m.tier}] ${escapeHtml(m.content.substring(0, 60))}... (Imp: ${m.importance_score.toFixed(2)})
                    </option>
                `).join('')}
            `;

            if (state.selectedMemoryId) {
                renderMemoryHistoryTimeline(state.selectedMemoryId);
            }
        } catch (error) {
            console.error('Error loading lifecycle memories:', error);
        }
    }

    async function renderMemoryHistoryTimeline(memoryId) {
        if (!memoryId) {
            elements.tracerTimelineContainer.innerHTML = `
                <div class="empty-state-card">
                    <span class="empty-icon">📜</span>
                    <p>Select a memory above to view its chronological event history.</p>
                </div>
            `;
            return;
        }

        try {
            const historyData = await apiRequest(`/memory/${memoryId}/history`);
            const memory = await apiRequest(`/memory/${memoryId}`);
            const historyList = historyData.history || [];

            elements.tracerTimelineContainer.innerHTML = `
                <div style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: var(--radius-md); border: 1px solid var(--border-subtle); margin-bottom: 1rem;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
                        <span class="badge ${getTierBadge(memory.tier).class}">Current Tier: ${memory.tier}</span>
                        <span class="badge ${getImportanceCategory(memory.importance_score).class}">Importance: ${memory.importance_score.toFixed(2)}</span>
                    </div>
                    <div style="font-size:0.9rem; font-weight:600; color:#f8fafc; margin-bottom:0.35rem;">
                        ${escapeHtml(memory.content)}
                    </div>
                    <div style="font-size:0.75rem; color:var(--text-muted);">
                        ID: <code style="color:#a5b4fc;">${memory.memory_id}</code> • Created: ${formatDate(memory.created_at)} • Access Count: <strong>${memory.access_count}</strong>
                    </div>
                </div>

                <div class="details-section-title" style="margin-bottom:0.85rem;">Audit & Transition History (${historyList.length} Events)</div>
                ${historyList.length > 0 ? historyList.map(h => `
                    <div class="timeline-event-card">
                        <div class="timeline-event-header">
                            <span class="badge ${getActionBadge(h.operation).class}">${h.operation}</span>
                            <span class="timeline-event-time">${formatDate(h.created_at)}</span>
                        </div>
                        <div style="font-size:0.8rem; color:#cbd5e1; margin-top:0.25rem;">
                            <strong>Reason:</strong> ${escapeHtml(h.reason || 'Lifecycle operation')}
                        </div>
                        ${h.old_content && h.old_content !== h.new_content ? `
                            <div class="timeline-event-diff">
                                <div class="diff-old">- Previous: ${escapeHtml(h.old_content)}</div>
                                <div class="diff-new">+ Updated: ${escapeHtml(h.new_content)}</div>
                            </div>
                        ` : ''}
                    </div>
                `).join('') : '<div class="empty-state-card"><p>No consolidation or compression events logged yet.</p></div>'}
            `;
        } catch (error) {
            console.error('Failed to load memory history:', error);
            elements.tracerTimelineContainer.innerHTML = `<div class="alert-box alert-danger">Error loading history: ${error.message}</div>`;
        }
    }

    // =========================================================================
    // View 4: Consolidation Visibility & Feed
    // =========================================================================

    async function loadConsolidationFeed() {
        try {
            const historyItems = await apiRequest(`/history?user_id=${encodeURIComponent(state.userId)}&limit=40`);
            if (historyItems.length === 0) {
                elements.auditFeedList.innerHTML = `
                    <div class="empty-state-card">
                        <span class="empty-icon">⚖️</span>
                        <p>No consolidation events recorded yet. Start a chat or add memories to observe consolidation decisions.</p>
                    </div>
                `;
                return;
            }

            elements.auditFeedList.innerHTML = historyItems.map(item => `
                <div class="audit-item-card">
                    <div class="audit-item-header">
                        <span class="badge ${getActionBadge(item.operation).class}">${item.operation}</span>
                        <span style="font-size:0.75rem; color:var(--text-muted); font-family:var(--font-mono);">${formatDate(item.created_at)}</span>
                    </div>
                    <div style="font-size:0.84rem; color:var(--text-primary);">
                        <strong>Reason:</strong> ${escapeHtml(item.reason || 'Memory consolidation rule')}
                    </div>
                    ${item.old_content ? `
                        <div class="timeline-event-diff">
                            <div class="diff-old">- Prior: ${escapeHtml(item.old_content)}</div>
                            <div class="diff-new">+ New: ${escapeHtml(item.new_content)}</div>
                        </div>
                    ` : `
                        <div style="font-size:0.78rem; font-family:var(--font-mono); color:#94a3b8; background:rgba(0,0,0,0.25); padding:0.4rem 0.6rem; border-radius:var(--radius-sm);">
                            Memory: ${escapeHtml(item.new_content || '')}
                        </div>
                    `}
                </div>
            `).join('');
        } catch (error) {
            console.error('Failed to load history feed:', error);
            showToast(`Could not load audit feed: ${error.message}`, 'danger');
        }
    }

    // =========================================================================
    // View 5: System & Research Metrics Panel
    // =========================================================================

    async function loadMetrics() {
        try {
            const metrics = await apiRequest(`/metrics?user_id=${encodeURIComponent(state.userId)}`);
            state.metrics = metrics;

            elements.metricTotalMemories.textContent = metrics.total_memories;
            elements.metricAvgImportance.textContent = metrics.avg_importance.toFixed(2);
            elements.metricTotalConsolidations.textContent = metrics.total_consolidations;
            elements.metricCompressedCount.textContent = metrics.compressed_memories;

            // Tier distribution bar chart
            const total = metrics.total_memories || 1;
            const tc = metrics.tier_counts || {};

            const workingPct = Math.round(((tc.WORKING || 0) / total) * 100);
            const shortPct = Math.round(((tc.SHORT_TERM || 0) / total) * 100);
            const longPct = Math.round(((tc.LONG_TERM || 0) / total) * 100);
            const archPct = Math.round(((tc.ARCHIVE || 0) / total) * 100);

            elements.barWorking.style.width = `${workingPct}%`;
            elements.valWorking.textContent = tc.WORKING || 0;

            elements.barShortTerm.style.width = `${shortPct}%`;
            elements.valShortTerm.textContent = tc.SHORT_TERM || 0;

            elements.barLongTerm.style.width = `${longPct}%`;
            elements.valLongTerm.textContent = tc.LONG_TERM || 0;

            elements.barArchive.style.width = `${archPct}%`;
            elements.valArchive.textContent = tc.ARCHIVE || 0;

            // Consolidation breakdown
            const cc = metrics.consolidation_counts || {};
            elements.countDuplicate.textContent = cc.DUPLICATE || 0;
            elements.countRelated.textContent = cc.RELATED || 0;
            elements.countContradictory.textContent = cc.CONTRADICTORY || 0;
            elements.countCompressedOp.textContent = cc.COMPRESSED || 0;

            checkSystemStatus();
        } catch (error) {
            console.error('Failed to load metrics:', error);
            showToast(`Metrics error: ${error.message}`, 'danger');
        }
    }

    // =========================================================================
    // Modals & User Actions
    // =========================================================================

    function inspectMemory(memoryId) {
        state.selectedMemoryId = memoryId;
        switchView('lifecycle');
        elements.tracerMemorySelect.value = memoryId;
        renderMemoryHistoryTimeline(memoryId);
    }

    function openTierTransitionModal(memoryId, content, currentTier) {
        elements.transitionMemoryId.value = memoryId;
        elements.transitionMemorySummary.textContent = `Memory: "${content.substring(0, 75)}..." (Current Tier: ${currentTier})`;
        elements.targetTierSelect.value = currentTier === 'WORKING' ? 'LONG_TERM' : (currentTier === 'SHORT_TERM' ? 'ARCHIVE' : 'WORKING');
        elements.modalTierTransition.style.display = 'flex';
    }

    async function handleTierTransitionSubmit() {
        const memoryId = elements.transitionMemoryId.value;
        const targetTier = elements.targetTierSelect.value;
        if (!memoryId || !targetTier) return;

        try {
            await apiRequest(`/memory/${memoryId}/transition`, {
                method: 'POST',
                body: JSON.stringify({ target_tier: targetTier }),
            });
            showToast(`Memory successfully transitioned to ${targetTier}!`, 'success');
            closeModals();
            loadMemories();
            checkSystemStatus();
        } catch (error) {
            showToast(`Transition failed: ${error.message}`, 'danger');
        }
    }

    async function handleManualMemorySubmit() {
        const content = elements.manualMemoryContent.value.trim();
        const userId = elements.manualMemoryUser.value.trim() || state.userId;
        if (!content) return;

        try {
            const memory = await apiRequest('/memory', {
                method: 'POST',
                body: JSON.stringify({ user_id: userId, content: content }),
            });
            showToast(`Memory stored in ${memory.tier} tier (Score: ${memory.importance_score.toFixed(2)})!`, 'success');
            closeModals();
            elements.manualMemoryContent.value = '';
            loadMemories();
            checkSystemStatus();
        } catch (error) {
            showToast(`Storage failed: ${error.message}`, 'danger');
        }
    }

    async function deleteMemory(memoryId) {
        if (!confirm('Are you sure you want to delete this memory and its audit history?')) return;
        try {
            await apiRequest(`/memory/${memoryId}`, { method: 'DELETE' });
            showToast('Memory deleted', 'info');
            loadMemories();
            checkSystemStatus();
        } catch (error) {
            showToast(`Delete failed: ${error.message}`, 'danger');
        }
    }

    async function handleDatabaseReset() {
        try {
            await apiRequest('/reset', {
                method: 'POST',
                body: JSON.stringify({ confirm: true }),
            });
            showToast('Research database emptied and refreshed!', 'success');
            closeModals();
            loadMemories();
            loadMetrics();
            checkSystemStatus();
            elements.chatMessages.innerHTML = '';
        } catch (error) {
            showToast(`Reset failed: ${error.message}`, 'danger');
        }
    }

    function closeModals() {
        elements.modalAddMemory.style.display = 'none';
        elements.modalMemoryDetails.style.display = 'none';
        elements.modalTierTransition.style.display = 'none';
        elements.modalResetConfirm.style.display = 'none';
    }

    function toggleDetails(drawerId) {
        const drawer = document.getElementById(drawerId);
        if (drawer) {
            drawer.classList.toggle('open');
            scrollToBottom();
        }
    }

    function sendSample(text) {
        elements.chatTextarea.value = text;
        handleChatSubmit();
    }

    function escapeHtml(str) {
        if (!str) return '';
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    // =========================================================================
    // Event Listeners Initialization
    // =========================================================================

    function initEvents() {
        // Navigation Tabs
        elements.navTabs.forEach(tab => {
            tab.addEventListener('click', () => switchView(tab.dataset.view));
        });

        // User Change
        elements.userSelect.addEventListener('change', (e) => {
            state.userId = e.target.value.trim() || 'user-1';
            showToast(`Switched active user to: ${state.userId}`, 'info');
            loadMemories();
            loadMetrics();
        });

        // Chat Form
        elements.chatForm.addEventListener('submit', (e) => {
            e.preventDefault();
            handleChatSubmit();
        });

        elements.chatTextarea.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleChatSubmit();
            }
        });

        // Dashboard Controls
        if (elements.memorySearchInput) {
            elements.memorySearchInput.addEventListener('input', () => {
                elements.btnClearSearch.style.display = elements.memorySearchInput.value ? 'inline-block' : 'none';
                loadMemories();
            });
        }
        if (elements.btnClearSearch) {
            elements.btnClearSearch.addEventListener('click', () => {
                elements.memorySearchInput.value = '';
                elements.btnClearSearch.style.display = 'none';
                loadMemories();
            });
        }
        if (elements.filterTier) elements.filterTier.addEventListener('change', loadMemories);
        if (elements.filterImportance) elements.filterImportance.addEventListener('change', loadMemories);
        if (elements.btnRefreshMemories) elements.btnRefreshMemories.addEventListener('click', () => {
            loadMemories();
            showToast('Memory dashboard refreshed', 'info');
        });

        if (elements.btnAddMemoryManual) {
            elements.btnAddMemoryManual.addEventListener('click', () => {
                elements.manualMemoryUser.value = state.userId;
                elements.modalAddMemory.style.display = 'flex';
            });
        }
        if (elements.btnSubmitManualMemory) elements.btnSubmitManualMemory.addEventListener('click', handleManualMemorySubmit);

        // Empty Database button on Dashboard
        if (elements.btnEmptyDbDashboard) {
            elements.btnEmptyDbDashboard.addEventListener('click', () => {
                elements.modalResetConfirm.style.display = 'flex';
            });
        }

        // Lifecycle Explorer
        if (elements.tracerMemorySelect) {
            elements.tracerMemorySelect.addEventListener('change', (e) => {
                state.selectedMemoryId = e.target.value;
                renderMemoryHistoryTimeline(e.target.value);
            });
        }

        // Consolidation Feed
        if (elements.btnRefreshHistory) elements.btnRefreshHistory.addEventListener('click', loadConsolidationFeed);

        // Tier Transition Modal
        if (elements.btnSubmitTransition) elements.btnSubmitTransition.addEventListener('click', handleTierTransitionSubmit);

        // Reset Database Modal from Header
        if (elements.btnResetDb) {
            elements.btnResetDb.addEventListener('click', () => {
                elements.modalResetConfirm.style.display = 'flex';
            });
        }
        if (elements.btnConfirmReset) elements.btnConfirmReset.addEventListener('click', handleDatabaseReset);

        // Escape key closes modals
        window.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') closeModals();
        });
    }

    // Expose global methods for inline HTML onclick handlers
    window.ADAM = {
        closeModals,
        toggleDetails,
        sendSample,
        inspectMemory,
        openTierTransitionModal,
        deleteMemory,
    };

    // Initialize application on DOM load
    document.addEventListener('DOMContentLoaded', () => {
        initEvents();
        checkSystemStatus();
        loadMemories();
        loadMetrics();
        setInterval(checkSystemStatus, 15000); // Heartbeat check
    });

})();
