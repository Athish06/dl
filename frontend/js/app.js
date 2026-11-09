/**
 * dl — Multi-Model Comparative Application Logic
 * Coordinates Model V1 (Baseline), Model V2 (Multi-Task), and Model V3 (Gated Polarity Inversion)
 */

// ─── Configuration ──────────────────────────────────────────
const API_BASE = ENV.API_BASE;
console.log("API_BASE loaded:", API_BASE);
console.log("ENV object:", ENV);

// ─── Benchmark Samples ──────────────────────────────────────
const SAMPLES = {
    positive:
        "This film was absolutely incredible! The cinematography was breathtaking, the performances were deeply moving, and the storyline kept me inspired from start to finish. One of the best movies I have seen in years. Truly a masterpiece.",
    negative:
        "What a terrible waste of time. The plot made no sense, the acting was wooden, and the dialogue felt completely unnatural. The special effects were laughably bad. Save your money and avoid this entirely.",
    mixed:
        "The film had some genuinely beautiful moments and the lead actor gave a compelling performance, but sadly it was let down by a weak third act and painfully slow pacing in the middle. Not a terrible film, but could have been much tighter.",
    mock_praise:
        "What an absolute masterpiece of cinematic failure. Truly a remarkable achievement in terrible filmmaking. The director worked tirelessly to make every single second as agonizing as humanly possible. Bravo!",
    subtle_sarcasm:
        "I just love when the main character makes the exact same stupid mistake for the fifth time in a row. Because clearly what this psychological thriller needed was an unprompted 20-minute dance sequence. Brilliant writing!",
};

// ─── State ──────────────────────────────────────────────────
let comparativeHistory = JSON.parse(localStorage.getItem("sentimentHistory_multi") || "[]");
let isAnalyzing = false;
let modelsInfoCache = null;
let currentSelectedModel = "v3";

// ─── DOM Elements ───────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const textInput = $("#text-input");
const charCount = $("#char-count");
const analyzeBtn = $("#analyze-btn");
const emptyState = $("#empty-state");
const loadingState = $("#loading-state");
const resultsState = $("#results-state");
const errorState = $("#error-state");
const errorText = $("#error-text");
const retryBtn = $("#retry-btn");
const serverStatus = $("#server-status");
const historyList = $("#history-list");
const clearHistoryBtn = $("#clear-history-btn");

// ─── Initialize ─────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initMobileDrawer();
    initMobileModelSwitcher();
    initInput();
    initSampleButtons();
    initAnalyze();
    initHistory();
    initModelSelector();
    initArchModal();
    checkServerHealth(true);
    loadModelsInfo();

    // Periodic health check — strictly once every 2 minutes (120,000 ms)
    setInterval(() => {
        if (!document.hidden) {
            checkServerHealth();
        }
    }, ENV.HEALTH_CHECK_INTERVAL || 120000);

    // Refresh health when user re-focuses tab if server was offline
    document.addEventListener("visibilitychange", () => {
        if (!document.hidden && !isServerOnline) {
            checkServerHealth();
        }
    });
});

// ─── Centralized Tab Switcher ────────────────────────────────
function switchTab(tabId) {
    if (!tabId) return;

    // 1. Update topbar tabs (desktop)
    $$(".tab-btn").forEach((b) => {
        b.classList.toggle("active", b.dataset.tab === tabId);
    });

    // 2. Update mobile drawer nav items
    $$(".mobile-nav-item").forEach((b) => {
        const isActive = b.dataset.tab === tabId;
        b.classList.toggle("active", isActive);
        const tag = b.querySelector(".nav-item-tag");
        if (tag && b.dataset.tab === "analyze") {
            tag.classList.toggle("active", isActive);
            tag.textContent = isActive ? "Active" : "";
        }
    });

    // 3. Update tab panels
    $$(".tab-panel").forEach((p) => p.classList.remove("active"));
    const targetPanel = $(`#panel-${tabId}`);
    if (targetPanel) {
        targetPanel.classList.add("active");
    }

    // 5. Special tab initializations
    if (tabId === "model") {
        const overviewView = $("#arch-overview-view");
        const detailView = $("#arch-detail-view");
        if (overviewView && detailView && !detailView.classList.contains("hidden")) {
            closeComponentModal();
        }
        if (modelsInfoCache) {
            renderSelectedModelDetails(currentSelectedModel);
        }
    }
    if (tabId === "history") {
        renderHistory();
    }

    // 6. Smoothly return scroll to top on tab change
    window.scrollTo({ top: 0, behavior: "smooth" });
}

// ─── Tab Navigation (Desktop Header) ─────────────────────────
function initTabs() {
    $$(".tab-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            switchTab(btn.dataset.tab);
        });
    });
}

// ─── Mobile Navigation Drawer ───────────────────────────────
function initMobileDrawer() {
    const menuBtn = $("#mobile-menu-btn");
    const drawer = $("#mobile-drawer");
    const overlay = $("#mobile-drawer-overlay");
    const closeBtn = $("#mobile-drawer-close");

    if (!menuBtn || !drawer || !overlay) return;

    function openDrawer() {
        drawer.classList.add("open");
        overlay.classList.add("open");
        document.body.style.overflow = "hidden";
    }

    function closeDrawer() {
        drawer.classList.remove("open");
        overlay.classList.remove("open");
        document.body.style.overflow = "";
    }

    menuBtn.addEventListener("click", openDrawer);
    if (closeBtn) closeBtn.addEventListener("click", closeDrawer);
    overlay.addEventListener("click", closeDrawer);

    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && drawer.classList.contains("open")) {
            closeDrawer();
        }
    });

    $$(".mobile-nav-item").forEach((item) => {
        item.addEventListener("click", () => {
            switchTab(item.dataset.tab);
            closeDrawer();
        });
    });
}

// ─── Mobile Model Focus Switcher ────────────────────────────
function initMobileModelSwitcher() {
    const switcher = $("#mobile-model-switcher");
    if (!switcher) return;

    const cards = {
        v1: $("#card-v1"),
        v2: $("#card-v2"),
        v3: $("#card-v3")
    };

    $$(".switcher-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            const filter = btn.dataset.filter;
            $$(".switcher-btn").forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");

            if (filter === "all") {
                Object.values(cards).forEach((c) => {
                    if (c) c.style.display = "";
                });
            } else {
                Object.entries(cards).forEach(([key, card]) => {
                    if (card) {
                        card.style.display = key === filter ? "" : "none";
                    }
                });
                if (cards[filter]) {
                    cards[filter].scrollIntoView({ behavior: "smooth", block: "nearest" });
                }
            }
        });
    });

    window.addEventListener("resize", () => {
        if (window.innerWidth > 768) {
            Object.values(cards).forEach((c) => {
                if (c) c.style.display = "";
            });
        }
    });
}

// ─── Text Input ─────────────────────────────────────────────
function initInput() {
    textInput.addEventListener("input", () => {
        const len = textInput.value.length;
        charCount.textContent = `${len.toLocaleString()} chars`;
    });
}

// ─── Sample Buttons ─────────────────────────────────────────
function initSampleButtons() {
    $$(".sample-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            const key = btn.dataset.sample;
            textInput.value = SAMPLES[key] || "";
            textInput.dispatchEvent(new Event("input"));
            textInput.focus();
        });
    });
}

// ─── Analyze (Multi-Model Gateway Call) ─────────────────────
function initAnalyze() {
    analyzeBtn.addEventListener("click", analyzeAll);
    retryBtn.addEventListener("click", analyzeAll);

    document.addEventListener("keydown", (e) => {
        if (e.ctrlKey && e.key === "Enter") {
            e.preventDefault();
            analyzeAll();
        }
    });
}

async function analyzeAll() {
    const text = textInput.value.trim();
    if (!text || isAnalyzing) return;

    isAnalyzing = true;
    analyzeBtn.disabled = true;
    showState("loading");

    const startTime = performance.now();

    try {
        console.log("Invoking Multi-Model Gateway at:", `${API_BASE}/predict-all`);

        const res = await fetch(`${API_BASE}/predict-all`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text }),
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `Server error (${res.status})`);
        }

        const data = await res.json();
        const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);

        renderComparativeResults(data, elapsed);
        addToComparativeHistory(text, data, elapsed);
        showState("results");
    } catch (err) {
        console.error("Multi-model prediction failed:", err);
        errorText.textContent = err.message || "Failed to communicate with Multi-Model AI Gateway";
        showState("error");
    } finally {
        isAnalyzing = false;
        analyzeBtn.disabled = false;
    }
}

function showState(state) {
    emptyState.classList.add("hidden");
    loadingState.classList.add("hidden");
    resultsState.classList.add("hidden");
    errorState.classList.add("hidden");

    const el = {
        empty: emptyState,
        loading: loadingState,
        results: resultsState,
        error: errorState,
    }[state];

    if (el) el.classList.remove("hidden");
}

// ─── Render Comparative Results Across V1, V2, and V3 ───────
function renderComparativeResults(data, elapsed) {
    const v1 = data.v1;
    const v2 = data.v2;
    const v3 = data.v3;
    const summary = data.summary;

    // ── 1. Evolution Consensus Banner ──
    const consBadge = $("#consensus-badge");
    consBadge.textContent = `Consensus: ${summary.consensus_sentiment}`;
    consBadge.className = `consensus-badge ${summary.consensus_sentiment === "Positive" ? "pos" : "neg"}`;

    const sarcAlert = $("#sarcasm-alert-badge");
    if (summary.sarcasm_detected) {
        sarcAlert.textContent = "Sarcasm Identified (V3)";
        sarcAlert.style.display = "inline-block";
    } else {
        sarcAlert.textContent = "Literal Statement";
        sarcAlert.style.display = "inline-block";
    }

    $("#processing-time").textContent = `${elapsed}s`;
    $("#evolution-insight").textContent = summary.evolution_insight;

    // ── 2. Render Model V1 (Baseline) ──
    if (v1 && v1.sentiment) {
        const isPos1 = v1.sentiment.label === "Positive";
        const badge1 = $("#v1-sentiment-badge");
        badge1.className = `sentiment-badge ${isPos1 ? "positive" : "negative"}`;
        $("#v1-sentiment-label").textContent = v1.sentiment.label;
        $("#v1-conf-val").textContent = `${Math.round(v1.sentiment.confidence * 100)}%`;

        const posPct1 = Math.round(v1.sentiment.probability_positive * 100);
        const negPct1 = Math.round(v1.sentiment.probability_negative * 100);
        $("#v1-val-pos").textContent = `${posPct1}%`;
        $("#v1-val-neg").textContent = `${negPct1}%`;
        requestAnimationFrame(() => {
            $("#v1-bar-pos").style.width = `${posPct1}%`;
            $("#v1-bar-neg").style.width = `${negPct1}%`;
        });
    }

    // ── 3. Render Model V2 (Multi-Task Phase 1) ──
    if (v2 && v2.sentiment && v2.sarcasm) {
        const isPos2 = v2.sentiment.label === "Positive";
        const isSarc2 = v2.sarcasm.label === "Sarcastic";

        const badgeSent2 = $("#v2-sentiment-badge");
        badgeSent2.className = `sentiment-badge-sm ${isPos2 ? "pos" : "neg"}`;
        badgeSent2.textContent = v2.sentiment.label;
        $("#v2-sent-conf").textContent = `${Math.round(v2.sentiment.confidence * 100)}%`;

        const badgeSarc2 = $("#v2-sarcasm-badge");
        badgeSarc2.className = `sarcasm-badge-sm ${isSarc2 ? "sarc" : "lit"}`;
        badgeSarc2.textContent = v2.sarcasm.label;
        $("#v2-sarc-conf").textContent = `${Math.round(v2.sarcasm.confidence * 100)}%`;

        $("#v2-interpretation-pill").textContent = v2.interpretation;

        const inconf2 = v2.incongruity_score || 0;
        $("#v2-incongruity-val").textContent = inconf2.toFixed(2);
        const inconfPct2 = Math.min(100, Math.max(5, Math.round((inconf2 / 2.0) * 100)));
        requestAnimationFrame(() => {
            $("#v2-incongruity-bar").style.width = `${inconfPct2}%`;
        });

        const subwordsV2 = $("#v2-subwords-list");
        if (v2.top_attended_subwords && v2.top_attended_subwords.length > 0) {
            subwordsV2.innerHTML = v2.top_attended_subwords
                .slice(0, 5)
                .map((t) => {
                    const cleanWord = (t.subword || "").trim() || t.subword || "";
                    const wt = Math.round((t.attention_weight || 0) * 100);
                    const isHigh = wt >= 20;
                    const wtTag = wt > 0 
                        ? `<span class="subword-weight ${isHigh ? "high" : ""}">${wt}%</span>`
                        : "";
                    return `<div class="subword-chip ${isHigh ? "high-attn" : ""}" title="${wt}% attention weight"><span class="subword-text">${escapeHtml(cleanWord)}</span>${wtTag}</div>`;
                })
                .join("");
        } else {
            subwordsV2.innerHTML = '<span class="subword-empty">No attended tokens available</span>';
        }
    }

    // ── 4. Render Model V3 (Flagship Gated Polarity Inversion) ──
    if (v3 && v3.sentiment && v3.sarcasm) {
        const isPos3 = v3.sentiment.label === "Positive";
        const isSarc3 = v3.sarcasm.label === "Sarcastic";

        const badgeSent3 = $("#v3-sentiment-badge");
        badgeSent3.className = `sentiment-badge-sm ${isPos3 ? "pos" : "neg"}`;
        badgeSent3.textContent = v3.sentiment.label;
        $("#v3-sent-conf").textContent = `${Math.round(v3.sentiment.confidence * 100)}%`;

        const badgeSarc3 = $("#v3-sarcasm-badge");
        badgeSarc3.className = `sarcasm-badge-sm ${isSarc3 ? "sarc" : "lit"}`;
        badgeSarc3.textContent = v3.sarcasm.label;
        $("#v3-sarc-conf").textContent = `${Math.round(v3.sarcasm.confidence * 100)}%`;

        $("#v3-interpretation-pill").textContent = v3.interpretation;

        const invStatus = $("#v3-inversion-status");
        if (isSarc3) {
            invStatus.textContent = "Polarity Inversion Active";
            invStatus.className = "inversion-status-pill active";
            invStatus.style.display = "inline-block";
        } else {
            invStatus.textContent = "Direct Polarity";
            invStatus.className = "inversion-status-pill inactive";
            invStatus.style.display = "inline-block";
        }

        const inconf3 = v3.incongruity_score || 0;
        $("#v3-incongruity-val").textContent = inconf3.toFixed(2);
        const inconfPct3 = Math.min(100, Math.max(5, Math.round((inconf3 / 2.0) * 100)));
        requestAnimationFrame(() => {
            $("#v3-incongruity-bar").style.width = `${inconfPct3}%`;
        });

        const subwordsV3 = $("#v3-subwords-list");
        if (v3.top_attended_subwords && v3.top_attended_subwords.length > 0) {
            subwordsV3.innerHTML = v3.top_attended_subwords
                .slice(0, 5)
                .map((t) => {
                    const cleanWord = (t.subword || "").trim() || t.subword || "";
                    const wt = Math.round((t.attention_weight || 0) * 100);
                    const isHigh = wt >= 20;
                    const wtTag = wt > 0 
                        ? `<span class="subword-weight ${isHigh ? "high" : ""}">${wt}%</span>`
                        : "";
                    return `<div class="subword-chip ${isHigh ? "high-attn" : ""}" title="${wt}% attention weight"><span class="subword-text">${escapeHtml(cleanWord)}</span>${wtTag}</div>`;
                })
                .join("");
        } else {
            subwordsV3.innerHTML = '<span class="subword-empty">No attended tokens available</span>';
        }
    }
}

// ─── Comparative History ────────────────────────────────────
function initHistory() {
    const clearDesktop = $("#clear-history-btn");
    const clearView = $("#history-view-clear-btn");

    function clearAllHistory() {
        comparativeHistory = [];
        try {
            localStorage.setItem("sentimentHistory_multi", "[]");
        } catch {
            // ignore
        }
        renderHistory();
    }

    if (clearDesktop) clearDesktop.addEventListener("click", clearAllHistory);
    if (clearView) clearView.addEventListener("click", clearAllHistory);

    renderHistory();
}

function addToComparativeHistory(text, data, elapsed = "0.00") {
    const v1Sent = data.v1?.sentiment?.label || "N/A";
    const v2Sent = data.v2?.sentiment?.label || "N/A";
    const v2Sarc = data.v2?.sarcasm?.label || "N/A";
    const v3Sent = data.v3?.sentiment?.label || "N/A";
    const v3Sarc = data.v3?.sarcasm?.label || "N/A";
    const consensus = data.consensus?.consensus || data.summary?.consensus || v3Sent;
    const interpretation = data.v3?.interpretation || data.summary?.flagship_interpretation || "";

    comparativeHistory.unshift({
        id: "hist_" + Date.now(),
        text: text,
        data: data,
        elapsed: elapsed,
        v1Sent,
        v2Sent,
        v2Sarc,
        v3Sent,
        v3Sarc,
        consensus,
        interpretation,
        timestamp: Date.now(),
    });

    if (comparativeHistory.length > 25) comparativeHistory = comparativeHistory.slice(0, 25);
    try {
        localStorage.setItem("sentimentHistory_multi", JSON.stringify(comparativeHistory));
    } catch (e) {
        console.warn("Storage quota reached, trimming history", e);
        comparativeHistory = comparativeHistory.slice(0, 10);
        localStorage.setItem("sentimentHistory_multi", JSON.stringify(comparativeHistory));
    }
    renderHistory();
}

function loadHistoryItem(item) {
    if (!item) return;

    // 1. Populate the query input
    textInput.value = item.text || "";
    textInput.dispatchEvent(new Event("input"));

    // 2. Switch to the Compare Models tab
    switchTab("analyze");

    // 3. Close mobile drawer if open
    const drawer = $("#mobile-drawer");
    const overlay = $("#mobile-drawer-overlay");
    if (drawer && drawer.classList.contains("open")) {
        drawer.classList.remove("open");
        if (overlay) overlay.classList.remove("open");
        document.body.style.overflow = "";
    }

    // 4. Immediately render the exact saved evaluation results
    if (item.data) {
        renderComparativeResults(item.data, item.elapsed || "0.00");
        showState("results");

        // 5. Scroll smoothly to results
        setTimeout(() => {
            const resultsEl = $("#results-state");
            if (resultsEl) {
                resultsEl.scrollIntoView({ behavior: "smooth", block: "start" });
            }
        }, 120);
    } else {
        analyzeAll();
    }
}

function formatTimeAgo(timestamp) {
    if (!timestamp) return "Recently";
    const diff = Math.floor((Date.now() - timestamp) / 1000);
    if (diff < 60) return "Just now";
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return new Date(timestamp).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function renderHistory() {
    const desktopList = $("#history-list");
    const viewList = $("#history-view-list");
    const drawerBadge = $("#drawer-history-count-badge");

    const count = comparativeHistory.length;

    // Update count badge in mobile drawer
    if (drawerBadge) {
        drawerBadge.textContent = count > 0 ? `${count} saved` : "0 saved";
    }

    if (count === 0) {
        if (desktopList) {
            desktopList.innerHTML =
                '<p class="history-empty">No comparative analyses yet. Submissions will appear here.</p>';
        }
        if (viewList) {
            viewList.innerHTML = `
                <div class="history-empty-state">
                    <div class="history-empty-icon-wrap">
                        <svg viewBox="0 0 24 24" width="36" height="36" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
                            <circle cx="12" cy="12" r="10"></circle>
                            <polyline points="12 6 12 12 16 14"></polyline>
                        </svg>
                    </div>
                    <h3 class="history-empty-title">No Saved Evaluations</h3>
                    <p class="history-empty-text">
                        Every comparative query evaluated across V1, V2, and V3 is recorded in your local browser storage.
                    </p>
                    <button class="history-empty-btn" id="btn-goto-evaluate">
                        <span>Run an Evaluation</span>
                        <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                            <line x1="5" y1="12" x2="19" y2="12"></line>
                            <polyline points="12 5 19 12 12 19"></polyline>
                        </svg>
                    </button>
                </div>
            `;
            const gotoBtn = $("#btn-goto-evaluate");
            if (gotoBtn) {
                gotoBtn.addEventListener("click", () => {
                    switchTab("analyze");
                    textInput.focus();
                });
            }
        }
        return;
    }

    // 1. Render Desktop Bottom Card List
    if (desktopList) {
        desktopList.innerHTML = comparativeHistory
            .map((item, idx) => {
                const labelClass = item.v3Sent === "Positive" ? "pos" : "neg";
                const displayQuery = item.text.length > 95 ? item.text.slice(0, 92) + "..." : item.text;
                return `
            <div class="history-item" data-index="${idx}" title="Click to view query and full results">
                <div class="history-item-content">
                    <span class="history-text">${escapeHtml(displayQuery)}</span>
                    <span class="history-sarcasm-tag">V3: ${escapeHtml(item.v3Sarc || "N/A")} • ${escapeHtml(item.v3Sent || "N/A")}</span>
                </div>
                <div class="history-item-right">
                    <span class="history-label ${labelClass}">${escapeHtml(item.v3Sent || "—")}</span>
                    <span class="history-view-tag">View Result</span>
                </div>
            </div>
        `;
            })
            .join("");

        desktopList.querySelectorAll(".history-item").forEach((el) => {
            el.addEventListener("click", () => {
                const idx = parseInt(el.dataset.index, 10);
                loadHistoryItem(comparativeHistory[idx]);
            });
        });
    }

    // 2. Render Dedicated Full History View (#panel-history)
    if (viewList) {
        viewList.innerHTML = comparativeHistory
            .map((item, idx) => {
                const consensus = item.consensus || item.v3Sent || "Neutral";
                const consensusClass = consensus === "Positive" ? "pos" : "neg";
                const hasSarcasm = item.v3Sarc === "Sarcastic" || item.v2Sarc === "Sarcastic";
                const sarcasmBadgeHtml = hasSarcasm
                    ? `<span class="history-badge sarcasm">Sarcasm Detected</span>`
                    : `<span class="history-badge literal">Literal</span>`;

                const timeAgo = formatTimeAgo(item.timestamp);
                const interp = item.interpretation;
                const interpHtml = interp
                    ? `<div class="history-card-interpretation">V3 Interpretation: ${escapeHtml(interp)}</div>`
                    : "";

                return `
            <div class="history-card-item" data-index="${idx}" role="button" tabindex="0" title="Click to inspect full multi-model results">
                <div class="history-card-header-row">
                    <div class="history-card-badges">
                        <span class="history-badge ${consensusClass}">${escapeHtml(consensus)}</span>
                        ${sarcasmBadgeHtml}
                    </div>
                    <span class="history-card-time">${escapeHtml(timeAgo)}</span>
                </div>
                <p class="history-card-query">${escapeHtml(item.text)}</p>
                ${interpHtml}
                <div class="history-card-breakdown-row">
                    <div class="history-mini-models-group">
                        <div class="history-mini-model">
                            <span class="mini-label">V1</span>
                            <span class="mini-val ${item.v1Sent === "Positive" ? "pos" : "neg"}">${escapeHtml(item.v1Sent || "—")}</span>
                        </div>
                        <div class="history-mini-model">
                            <span class="mini-label">V2</span>
                            <span class="mini-val ${item.v2Sent === "Positive" ? "pos" : "neg"}">${escapeHtml(item.v2Sent || "—")}</span>
                            <span class="mini-sub">${escapeHtml(item.v2Sarc || "")}</span>
                        </div>
                        <div class="history-mini-model highlight">
                            <span class="mini-label">V3 Flagship</span>
                            <span class="mini-val ${item.v3Sent === "Positive" ? "pos" : "neg"}">${escapeHtml(item.v3Sent || "—")}</span>
                            <span class="mini-sub">${escapeHtml(item.v3Sarc || "")}</span>
                        </div>
                    </div>
                    <div class="history-inspect-action">
                        <span>Inspect Results</span>
                        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                            <polyline points="9 18 15 12 9 6"></polyline>
                        </svg>
                    </div>
                </div>
            </div>
        `;
            })
            .join("");

        viewList.querySelectorAll(".history-card-item").forEach((el) => {
            el.addEventListener("click", () => {
                const idx = parseInt(el.dataset.index, 10);
                loadHistoryItem(comparativeHistory[idx]);
            });
            el.addEventListener("keydown", (e) => {
                if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    const idx = parseInt(el.dataset.index, 10);
                    loadHistoryItem(comparativeHistory[idx]);
                }
            });
        });
    }
}

function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}

// ─── Gateway Server Health ──────────────────────────────────
let isCheckingHealth = false;
let isServerOnline = false;

async function checkServerHealth(forceShowChecking = false) {
    if (isCheckingHealth) return; // Prevent duplicate overlapping health requests
    isCheckingHealth = true;

    const dot = serverStatus.querySelector(".status-dot");
    const text = serverStatus.querySelector(".status-text");

    // Only visually show "Checking..." on boot or when previously offline
    if (forceShowChecking || !isServerOnline) {
        dot.className = "status-dot checking";
        text.textContent = "Checking...";
    }

    try {
        const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(ENV.HEALTH_CHECK_TIMEOUT) });
        const data = await res.json();

        if (data.all_ready) {
            dot.className = "status-dot online";
            text.textContent = "All 3 Models Ready";
            isServerOnline = true;
        } else if (data.status === "healthy" || data.status === "ok") {
            dot.className = "status-dot online";
            text.textContent = "Gateway Ready";
            isServerOnline = true;
        } else {
            dot.className = "status-dot offline";
            text.textContent = "Models Not Ready";
            isServerOnline = false;
        }
    } catch {
        dot.className = "status-dot offline";
        text.textContent = "Gateway Offline";
        isServerOnline = false;
    } finally {
        const drawerStatus = $("#drawer-gateway-status");
        if (drawerStatus) {
            drawerStatus.textContent = isServerOnline ? "Active" : "Offline";
            drawerStatus.className = `drawer-status-badge ${isServerOnline ? "online" : "offline"}`;
        }
        isCheckingHealth = false;
    }
}

// ─── Architecture Tab: Multi-Model Inspector ────────────────
function initModelSelector() {
    $$(".model-nav-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            $$(".model-nav-btn").forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");
            currentSelectedModel = btn.dataset.model;
            renderSelectedModelDetails(currentSelectedModel);
        });
    });
}

// ─── Offline Fallback Architecture Data ──────────────────────
const FALLBACK_MODELS_INFO = {
    v1: {
        model_name: "SingleTask_CNN_BiLSTM_Attention_V1",
        version: "Model V1 (Baseline Single-Task)",
        total_parameters: 4749089,
        dataset: "IMDB + Yelp (250K Reviews)",
        vocab_size: 35000,
        layers: [
            { name: "embedding", type: "Word-level Trainable Embedding", output_shape: "(batch, 200, 128)", params: 4480000, component: "trainable_embeddings" },
            { name: "conv_k3", type: "Conv1D (k=3, 32 filters)", output_shape: "(batch, 32, 200)", params: 12320, component: "conv1d_v1" },
            { name: "conv_k5", type: "Conv1D (k=5, 32 filters)", output_shape: "(batch, 32, 200)", params: 20512, component: "conv1d_v1" },
            { name: "conv_k7", type: "Conv1D (k=7, 32 filters)", output_shape: "(batch, 32, 200)", params: 28704, component: "conv1d_v1" },
            { name: "bilstm", type: "BiLSTM (64 units per direction)", output_shape: "(batch, 200, 128)", params: 82944, component: "bilstm" },
            { name: "attention", type: "Bahdanau Additive Attention", output_shape: "(batch, 128)", params: 16512, component: "bahdanau_attention" },
            { name: "dense", type: "Dense Linear + Dropout", output_shape: "(batch, 128)", params: 16512, component: "single_sentiment_head" },
            { name: "output", type: "Linear + Sigmoid (Sentiment)", output_shape: "(batch, 1)", params: 129, component: "single_sentiment_head" }
        ],
        training: { test_accuracy: 0.8895, test_loss: 0.2501, epochs_trained: 10, dataset: "IMDB + Yelp (250K)" }
    },
    v2: {
        model_name: "MultiTask_CNN_BiLSTM_Attention_Incongruity_V2",
        version: "Model V2 (Multi-Task + Incongruity)",
        total_parameters: 10611586,
        dataset: "IMDB + SST-2 Rotten Tomatoes + Headlines (150K)",
        vocab_size: 30000,
        layers: [
            { name: "embedding", type: "Stanford GloVe 300d Embeddings", output_shape: "(batch, 200, 300)", params: 9000000, component: "glove_embeddings" },
            { name: "convs", type: "4-Scale Parallel Conv1D (k=2,3,5,7)", output_shape: "(batch, 256, 200)", params: 326656, component: "multiscale_cnn" },
            { name: "bilstm", type: "Deep BiLSTM (2 Layers, 128 units)", output_shape: "(batch, 200, 256)", params: 788480, component: "bilstm" },
            { name: "incongruity", type: "Sequence Split Contrast Operator", output_shape: "(batch, 256)", params: 0, component: "incongruity_module" },
            { name: "attention", type: "4-Head Scaled Dot-Product Attention", output_shape: "(batch, 256)", params: 263168, component: "multihead_attention" },
            { name: "sarcasm_head", type: "Linear + Sigmoid (Sarcasm)", output_shape: "(batch, 1)", params: 33025, component: "multitask_heads" },
            { name: "sentiment_head", type: "Linear + Sigmoid (Conditioned)", output_shape: "(batch, 1)", params: 33281, component: "multitask_heads" }
        ],
        training: { test_accuracy: 0.9102, sarcasm_accuracy: 0.8515, test_loss: 0.3679, epochs_trained: 8, dataset: "IMDB + SST-2 + Headlines (150K)" }
    },
    v3: {
        model_name: "MultiTask_CNN_BiLSTM_Attention_GatedInversion_V3",
        version: "Model V3 (Flagship Gated Inversion)",
        total_parameters: 10664194,
        dataset: "Tier 2 Hybrid: IMDB + SARC Reddit + Headlines + iSarcasm (230K)",
        vocab_size: 30000,
        layers: [
            { name: "embedding", type: "Stanford GloVe 300d Embeddings", output_shape: "(batch, 200, 300)", params: 9000000, component: "glove_embeddings" },
            { name: "convs", type: "4-Scale Conv1D with Mish + BatchNorm", output_shape: "(batch, 256, 200)", params: 326656, component: "multiscale_cnn" },
            { name: "bilstm", type: "Deep BiLSTM (2 Layers, 128 units)", output_shape: "(batch, 200, 256)", params: 788480, component: "bilstm" },
            { name: "incongruity", type: "Sequence Split & Contrast Operator", output_shape: "(batch, 256)", params: 0, component: "incongruity_module" },
            { name: "attention", type: "4-Head Scaled Dot-Product Attention", output_shape: "(batch, 256)", params: 263168, component: "multihead_attention" },
            { name: "sarcasm_head", type: "Linear + Sigmoid (Auxiliary Task)", output_shape: "(batch, 1)", params: 33025, component: "multitask_heads" },
            { name: "gated_inversion", type: "Multiplicative Inversion Gate (1 - 2*p_sarc)", output_shape: "(batch, 256)", params: 0, component: "gated_polarity_inversion" },
            { name: "sentiment_head", type: "Linear + Sigmoid (Inverted Conditioning)", output_shape: "(batch, 1)", params: 52609, component: "gated_polarity_inversion" }
        ],
        training: { test_accuracy: 0.9028, sarcasm_accuracy: 0.8301, test_loss: 0.3198, epochs_trained: 8, dataset: "Tier 2 Hybrid (230K+ samples)" }
    }
};

async function loadModelsInfo() {
    try {
        const res = await fetch(`${API_BASE}/models-info`);
        if (!res.ok) throw new Error("Models info not available");
        modelsInfoCache = await res.json();
    } catch (err) {
        console.log("Using built-in models specifications cache:", err);
        modelsInfoCache = FALLBACK_MODELS_INFO;
    }
    renderSelectedModelDetails(currentSelectedModel);
}

function renderSelectedModelDetails(modelKey) {
    const info = (modelsInfoCache && modelsInfoCache[modelKey]) ? modelsInfoCache[modelKey] : FALLBACK_MODELS_INFO[modelKey];
    if (!info) return;

    // Title
    $("#arch-card-title").innerHTML = `
        <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="2" y="3" width="20" height="14" rx="2" />
            <line x1="8" y1="21" x2="16" y2="21" />
            <line x1="12" y1="17" x2="12" y2="21" />
        </svg>
        Neural Architecture: ${escapeHtml(info.version || info.model_name)}
    `;

    // Pipeline visualization
    renderPipelineForModel(modelKey);

    // Layers Table
    const tbody = $("#layers-tbody");
    if (info.layers && info.layers.length > 0) {
        tbody.innerHTML = info.layers
            .map(
                (l) => {
                    const compAttr = l.component ? `data-component="${escapeHtml(l.component)}"` : "";
                    const inspectPill = l.component ? `<button class="table-inspect-pill" ${compAttr}>Inspect</button>` : "";
                    return `
            <tr class="layer-table-row ${l.component ? 'clickable-row' : ''}" ${compAttr} title="Click to inspect this layer">
                <td>
                    <div class="layer-cell-name">
                        <span class="layer-name-text">${escapeHtml(l.name)}</span>
                        ${inspectPill}
                    </div>
                </td>
                <td>${escapeHtml(l.type)}</td>
                <td><code class="tensor-shape-tag">${escapeHtml(l.output_shape)}</code></td>
                <td>${l.params.toLocaleString()}</td>
            </tr>
        `;
                }
            )
            .join("");

        tbody.querySelectorAll(".clickable-row").forEach((row) => {
            row.addEventListener("click", () => {
                const comp = row.dataset.component;
                if (comp) openComponentModal(comp);
            });
        });
    }

    // Metrics Tiles
    if (info.training) {
        const t = info.training;
        const sentAcc = t.test_accuracy;
        const sarcAcc = t.sarcasm_accuracy;

        $("#metric-accuracy").textContent = sentAcc !== undefined ? `${(sentAcc * 100).toFixed(1)}%` : "—";
        $("#metric-sarcasm").textContent = sarcAcc !== undefined ? `${(sarcAcc * 100).toFixed(1)}%` : "N/A (Single-Task)";
        $("#metric-loss").textContent = t.test_loss !== undefined ? t.test_loss.toFixed(4) : "—";
        $("#metric-epochs").textContent = t.epochs_trained || "—";
        $("#metric-params").textContent = formatNumber(info.total_parameters || info.total_params);
        $("#metric-dataset").textContent = t.dataset || "—";
        $("#metric-vocab").textContent = t.vocab_size ? t.vocab_size.toLocaleString() : "—";

        // Chart
        if (t.training_history) {
            drawTrainingChart(t.training_history, modelKey);
        }
    }
}

function renderPipelineForModel(modelKey) {
    const pipeline = $("#arch-pipeline");
    if (!pipeline) return;

    function createAtomicCardHtml(cat, name, spec, desc, compId, isFlagship = false) {
        const clickAttr = compId 
            ? `class="arch-atomic-card clickable-node ${isFlagship ? 'flagship-node' : ''}" data-component="${escapeHtml(compId)}" role="button" tabindex="0" title="Click to inspect ${escapeHtml(name)}"` 
            : `class="arch-atomic-card input-card"`;
        const actionHtml = compId 
            ? `<span class="atomic-inspect-action">Inspect →</span>` 
            : `<span class="atomic-input-pill">Input</span>`;
        return `
            <div ${clickAttr}>
                <div class="atomic-card-top">
                    <span class="atomic-cat-pill">${escapeHtml(cat)}</span>
                    ${actionHtml}
                </div>
                <h4 class="atomic-name">${escapeHtml(name)}</h4>
                <span class="atomic-spec">${escapeHtml(spec)}</span>
                <p class="atomic-desc">${escapeHtml(desc)}</p>
            </div>
        `;
    }

    function intraArrowHtml() {
        return `<div class="arch-intra-arrow">→</div>`;
    }

    function phaseConnectorHtml(tensorLabel) {
        return `
            <div class="arch-phase-connector">
                <span class="arch-phase-connector-line"></span>
                <span class="arch-phase-connector-badge">
                    <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2.2">
                        <line x1="12" y1="4" x2="12" y2="20"></line>
                        <polyline points="18 14 12 20 6 14"></polyline>
                    </svg>
                    ${escapeHtml(tensorLabel)}
                </span>
                <span class="arch-phase-connector-line"></span>
            </div>
        `;
    }

    function createPhaseGroupHtml(phaseTag, phaseTitle, phaseMeta, innerNodesHtml) {
        return `
            <div class="arch-phase-group">
                <div class="arch-phase-header">
                    <div class="arch-phase-title-wrap">
                        <span class="arch-phase-tag">${escapeHtml(phaseTag)}</span>
                        <span class="arch-phase-title">${escapeHtml(phaseTitle)}</span>
                    </div>
                    <span class="arch-phase-meta">${escapeHtml(phaseMeta)}</span>
                </div>
                <div class="arch-phase-nodes">
                    ${innerNodesHtml}
                </div>
            </div>
        `;
    }

    if (modelKey === "v1") {
        pipeline.innerHTML = `
            <div class="arch-flowchart">
                ${createPhaseGroupHtml(
                    "Phase 1",
                    "Ingestion & Word Representation",
                    "Raw Corpus → Trainable Embedding Space",
                    createAtomicCardHtml("TXT", "Input Review", "Raw String (max_len=200)", "User review or social text submitted for direct single-task classification.", null) +
                    intraArrowHtml() +
                    createAtomicCardHtml("TOK", "Whitespace Tokenizer", "35,000 Vocab Dict", "Splits text on spaces. Out-of-vocabulary words map to [OOV] ID with zero representation.", "whitespace_tokenizer") +
                    intraArrowHtml() +
                    createAtomicCardHtml("EMB", "Trainable Embeddings", "(batch, 200, 128) Float32", "128-dimensional embedding matrix trained from scratch without pretrained semantic transfer.", "trainable_embeddings")
                )}

                ${phaseConnectorHtml("Trainable Word Embedding Tensor: (batch_size, 200, 128) Float32")}

                ${createPhaseGroupHtml(
                    "Phase 2",
                    "Sequential Feature Extraction",
                    "Local Filter Banks → Recurrent Temporal State",
                    createAtomicCardHtml("CNN", "Parallel Conv1D", "96 Filters (k=3, 5, 7)", "Extracts shallow fixed-width local phrase patterns without subword awareness.", "conv1d_v1") +
                    intraArrowHtml() +
                    createAtomicCardHtml("LSTM", "Single-Layer BiLSTM", "64 Units (batch, 200, 128)", "Processes sequential token dependencies bidirectionally across the review.", "bilstm")
                )}

                ${phaseConnectorHtml("Recurrent Hidden States: (batch_size, 200, 128) Float32")}

                ${createPhaseGroupHtml(
                    "Phase 3",
                    "Classical Additive Alignment",
                    "Single Scalar Spotlight Mechanism",
                    createAtomicCardHtml("ATTN", "Bahdanau Attention", "Single Head Context (B, 128)", "Computes a single scalar attention weight per step via additive tanh projection.", "bahdanau_attention")
                )}

                ${phaseConnectorHtml("Attended Context Summary: (batch_size, 128) Float32")}

                ${createPhaseGroupHtml(
                    "Phase 4",
                    "Single-Task Sentiment Prediction",
                    "Direct Linear Projection",
                    createAtomicCardHtml("OUT", "Single Sentiment Head", "Linear + Sigmoid (B, 1)", "Projects context vector directly into binary sentiment probability with no sarcasm modeling.", "single_sentiment_head")
                )}
            </div>
        `;
    } else if (modelKey === "v2") {
        pipeline.innerHTML = `
            <div class="arch-flowchart">
                ${createPhaseGroupHtml(
                    "Phase 1",
                    "Ingestion & Subword Representation",
                    "Raw Corpus → Pretrained Distributional Semantics",
                    createAtomicCardHtml("TXT", "Input Review", "Raw String (max_len=200)", "User review or social text submitted for multi-task sentiment and sarcasm analysis.", null) +
                    intraArrowHtml() +
                    createAtomicCardHtml("BPE", "Rust BPE Tokenizer", "30,000 Morphemes (Zero OOV)", "Decomposes words into frequent subwords, eliminating out-of-vocabulary failures.", "bpe_tokenizer") +
                    intraArrowHtml() +
                    createAtomicCardHtml("EMB", "Stanford GloVe 300d", "(batch, 200, 300) Pretrained", "Initializes subword tokens with dense semantic vectors trained on 6 billion tokens.", "glove_embeddings")
                )}

                ${phaseConnectorHtml("Subword Semantic Embeddings: (batch_size, 200, 300) Float32")}

                ${createPhaseGroupHtml(
                    "Phase 2",
                    "Hierarchical Feature Encoders",
                    "Multi-Scale Convolution → Deep Bidirectional Memory",
                    createAtomicCardHtml("CNN", "Multi-Scale Conv1D", "256 Filters (k=2, 3, 5, 7)", "Parallel n-gram feature detectors with Mish activations and Batch Normalization.", "multiscale_cnn") +
                    intraArrowHtml() +
                    createAtomicCardHtml("LSTM", "Deep 2-Layer BiLSTM", "256 Units (batch, 200, 256)", "Maintains bidirectional contextual history across long-distance syntactic dependencies.", "bilstm")
                )}

                ${phaseConnectorHtml("Deep Recurrent Memory Sequence: (batch_size, 200, 256) Float32")}

                ${createPhaseGroupHtml(
                    "Phase 3",
                    "Cognitive Reasoning & Incongruity",
                    "Structural Contrast → Multi-Head Attention",
                    createAtomicCardHtml("INC", "Incongruity Module", "Sequence Split & Contrast", "Computes absolute difference and Hadamard product between review halves to catch tone shifts.", "incongruity_module") +
                    intraArrowHtml() +
                    createAtomicCardHtml("ATTN", "Multi-Head Attention", "4 Heads (256-dim Scaled Dot)", "Jointly attends to contradictory affective cues across distinct representation subspaces.", "multihead_attention")
                )}

                ${phaseConnectorHtml("Shared Multi-Task Latent Vector: (batch_size, 256) Float32")}

                ${createPhaseGroupHtml(
                    "Phase 4",
                    "Dual Multi-Task Output Heads",
                    "Simultaneous Sarcasm & Conditioned Sentiment",
                    createAtomicCardHtml("HEAD", "Sarcasm Output Head", "Linear + Sigmoid (B, 1)", "Auxiliary task predicting probability of sarcastic, ironic, or mock-praise tone.", "multitask_heads") +
                    intraArrowHtml() +
                    createAtomicCardHtml("OUT", "Conditioned Sentiment Head", "Linear + Sigmoid (B, 1)", "Primary task predicting positive vs negative sentiment with conditioned shared features.", "multitask_heads")
                )}
            </div>
        `;
    } else {
        // v3 Flagship Gated Inversion
        pipeline.innerHTML = `
            <div class="arch-flowchart">
                ${createPhaseGroupHtml(
                    "Phase 1",
                    "Ingestion & Subword Representation",
                    "Raw Corpus → Pretrained Distributional Semantics",
                    createAtomicCardHtml("TXT", "Input Review", "Raw String (max_len=200)", "User review or social text submitted for multi-task sentiment and sarcasm analysis.", null) +
                    intraArrowHtml() +
                    createAtomicCardHtml("BPE", "Rust BPE Tokenizer", "30,000 Morphemes (Zero OOV)", "Decomposes words into frequent subword morphemes, completely eliminating OOV failures.", "bpe_tokenizer") +
                    intraArrowHtml() +
                    createAtomicCardHtml("EMB", "Stanford GloVe 300d", "(batch, 200, 300) Pretrained", "Dense distributional semantic vectors pretrained on 6 billion tokens of Wikipedia and Gigaword.", "glove_embeddings")
                )}

                ${phaseConnectorHtml("Subword Semantic Embeddings: (batch_size, 200, 300) Float32")}

                ${createPhaseGroupHtml(
                    "Phase 2",
                    "Hierarchical Feature Encoders",
                    "Multi-Scale Convolution → Deep Bidirectional Memory",
                    createAtomicCardHtml("CNN", "Multi-Scale Conv1D", "256 Filters (k=2, 3, 5, 7)", "Parallel n-gram filters with Mish non-linearities and BatchNorm to extract rich motifs.", "multiscale_cnn") +
                    intraArrowHtml() +
                    createAtomicCardHtml("LSTM", "Deep 2-Layer BiLSTM", "256 Units (batch, 200, 256)", "Bidirectional recurrence maintaining forward and reverse sequential dependencies.", "bilstm")
                )}

                ${phaseConnectorHtml("Deep Recurrent Memory Sequence: (batch_size, 200, 256) Float32")}

                ${createPhaseGroupHtml(
                    "Phase 3",
                    "Cognitive Reasoning & Incongruity",
                    "Structural Contrast → Multi-Head Attention",
                    createAtomicCardHtml("INC", "Incongruity Module", "Sequence Split & Contrast", "Computes absolute difference and Hadamard product between review halves to catch tonal reversals.", "incongruity_module") +
                    intraArrowHtml() +
                    createAtomicCardHtml("ATTN", "Multi-Head Attention", "4 Heads (256-dim Scaled Dot)", "Jointly attends to conflicting positive and negative emotional markers simultaneously.", "multihead_attention")
                )}

                ${phaseConnectorHtml("Shared Multi-Task Latent Vector: (batch_size, 256) Float32")}

                ${createPhaseGroupHtml(
                    "Phase 4",
                    "Gated Polarity Inversion & Calibrated Heads",
                    "Auxiliary Sarcasm Detection → Active Polarity Flip",
                    createAtomicCardHtml("HEAD", "Sarcasm Output Head", "Linear + Sigmoid (B, 1)", "Auxiliary task predicting continuous probability of sarcastic, mock-praise tone.", "multitask_heads") +
                    intraArrowHtml() +
                    createAtomicCardHtml("GATE", "Gated Polarity Inversion", "Multiplicative (1 - 2·p_sarc)", "Dynamically inverts positive mock-praise features into negative polarity when sarcasm is detected.", "gated_polarity_inversion", true) +
                    intraArrowHtml() +
                    createAtomicCardHtml("OUT", "Dual Sentiment Head", "Linear + Sigmoid (B, 1)", "Final context-aware positive or negative classification calibrated for sarcastic irony.", "multitask_heads")
                )}
            </div>
        `;
    }

    // Attach event listeners to all clickable atomic cards
    pipeline.querySelectorAll(".clickable-node").forEach((node) => {
        node.addEventListener("click", () => {
            const comp = node.dataset.component;
            if (comp) openComponentModal(comp);
        });
        node.addEventListener("keydown", (e) => {
            if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                const comp = node.dataset.component;
                if (comp) openComponentModal(comp);
            }
        });
    });
}

// ─── Architecture Deep Dive Modal Controller ─────────────────
let activeModalComponentId = "bpe_tokenizer";
let activeModalSection = "overview";

const PIPELINE_ORDER = {
    v1: ["whitespace_tokenizer", "trainable_embeddings", "conv1d_v1", "bilstm", "bahdanau_attention", "single_sentiment_head"],
    v2: ["bpe_tokenizer", "glove_embeddings", "multiscale_cnn", "bilstm", "incongruity_module", "multihead_attention", "multitask_heads"],
    v3: ["bpe_tokenizer", "glove_embeddings", "multiscale_cnn", "bilstm", "incongruity_module", "multihead_attention", "gated_polarity_inversion", "multitask_heads"]
};

function initArchModal() {
    const backBtnTop = $("#arch-back-to-pipeline");
    const backBtnBottom = $("#arch-back-bottom-btn");
    const topPrevBtn = $("#detail-top-btn-prev");
    const topNextBtn = $("#detail-top-btn-next");
    const prevBtn = $("#modal-btn-prev");
    const nextBtn = $("#modal-btn-next");

    if (backBtnTop) backBtnTop.addEventListener("click", closeComponentModal);
    if (backBtnBottom) backBtnBottom.addEventListener("click", closeComponentModal);

    if (topPrevBtn) {
        topPrevBtn.addEventListener("click", () => {
            navigateModalComponent(-1);
        });
    }
    if (topNextBtn) {
        topNextBtn.addEventListener("click", () => {
            navigateModalComponent(1);
        });
    }

    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") {
            const detailView = $("#arch-detail-view");
            if (detailView && !detailView.classList.contains("hidden")) {
                closeComponentModal();
            }
        }
    });

    // Quick chips in directory
    $$(".quick-chip-btn").forEach((chip) => {
        chip.addEventListener("click", () => {
            const comp = chip.dataset.component;
            if (comp) openComponentModal(comp);
        });
    });

    // Modal / Detail internal section tabs
    $$(".modal-tab-btn").forEach((tab) => {
        tab.addEventListener("click", () => {
            const sec = tab.dataset.section;
            if (sec) switchModalSection(sec);
        });
    });

    // Previous / Next pipeline navigation
    if (prevBtn) {
        prevBtn.addEventListener("click", () => {
            navigateModalComponent(-1);
        });
    }
    if (nextBtn) {
        nextBtn.addEventListener("click", () => {
            navigateModalComponent(1);
        });
    }
}

function openComponentModal(componentId, section = "overview") {
    if (!window.ARCH_COMPONENTS || !window.ARCH_COMPONENTS[componentId]) {
        console.warn("Component not found in ARCH_COMPONENTS:", componentId);
        return;
    }

    activeModalComponentId = componentId;
    activeModalSection = section;

    const comp = window.ARCH_COMPONENTS[componentId];
    const overviewView = $("#arch-overview-view");
    const detailView = $("#arch-detail-view");

    // Header info
    const monoBadge = $("#modal-badge-mono");
    const catBadge = $("#modal-badge-category");
    const modelsBadge = $("#modal-badge-models");
    const compTitle = $("#modal-component-title");
    const compSub = $("#modal-component-sub");

    if (monoBadge) monoBadge.textContent = comp.shortName;
    if (catBadge) catBadge.textContent = comp.category;
    if (modelsBadge) modelsBadge.textContent = Array.isArray(comp.models) ? comp.models.join(" • ") : comp.models;
    if (compTitle) compTitle.textContent = comp.title;
    if (compSub) compSub.textContent = `${comp.parameters} | Output: ${comp.outputShape}`;

    // Update section tabs
    $$(".modal-tab-btn").forEach((b) => {
        b.classList.toggle("active", b.dataset.section === activeModalSection);
    });

    // Render section content
    renderModalSectionContent(comp, activeModalSection);

    // Update previous / next button state
    updateModalNavButtons();

    // Switch view from overview to full-page detail view
    if (overviewView) overviewView.classList.add("hidden");
    if (detailView) detailView.classList.remove("hidden");

    // Scroll smoothly to top of window
    window.scrollTo({ top: 0, behavior: "smooth" });
}

function switchModalSection(section) {
    activeModalSection = section;
    const comp = window.ARCH_COMPONENTS[activeModalComponentId];
    if (!comp) return;

    $$(".modal-tab-btn").forEach((b) => {
        b.classList.toggle("active", b.dataset.section === section);
    });

    renderModalSectionContent(comp, section);
}

function renderModalSectionContent(comp, section) {
    const body = $("#arch-modal-body");
    if (!body) return;

    // Reset scroll position on tab switch
    body.scrollTop = 0;

    if (section === "overview") {
        body.innerHTML = `
            <div class="modal-section-block">
                ${comp.beginnerIntuition ? `
                    <div class="modal-intuition-card">
                        <div class="intuition-header">
                            <span class="intuition-label">Intuition for Beginners (Layman Analogy)</span>
                        </div>
                        <p class="intuition-text">${escapeHtml(comp.beginnerIntuition)}</p>
                    </div>
                ` : ""}

                <div class="modal-summary-card">
                    <span class="modal-sec-sub">EXECUTIVE SUMMARY</span>
                    <p class="modal-summary-text">${escapeHtml(comp.summary)}</p>
                </div>

                <div class="modal-text-group">
                    <h3 class="modal-group-title">Conceptual Definition & Purpose</h3>
                    <p class="modal-prose">${escapeHtml(comp.whatIsIt)}</p>
                </div>

                <div class="modal-text-group">
                    <h3 class="modal-group-title">Role in the dl Architecture</h3>
                    <p class="modal-prose">${escapeHtml(comp.architectureRole)}</p>
                </div>

                <div class="modal-spec-grid">
                    <div class="modal-spec-item">
                        <span class="spec-label">Layer Category</span>
                        <span class="spec-val">${escapeHtml(comp.category)}</span>
                    </div>
                    <div class="modal-spec-item">
                        <span class="spec-label">Parameters</span>
                        <span class="spec-val">${escapeHtml(comp.parameters)}</span>
                    </div>
                    <div class="modal-spec-item">
                        <span class="spec-label">Input Specification</span>
                        <span class="spec-val"><code>${escapeHtml(comp.inputShape)}</code></span>
                    </div>
                    <div class="modal-spec-item">
                        <span class="spec-label">Output Tensor</span>
                        <span class="spec-val"><code>${escapeHtml(comp.outputShape)}</code></span>
                    </div>
                </div>
            </div>
        `;
    } else if (section === "training") {
        body.innerHTML = `
            <div class="modal-section-block">
                <div class="modal-text-group">
                    <h3 class="modal-group-title">How It Is Trained / Initialized</h3>
                    <div class="modal-code-box">
                        <pre><code>${escapeHtml(comp.howIsItTrained)}</code></pre>
                    </div>
                </div>

                <div class="modal-text-group">
                    <h3 class="modal-group-title">Training Data Fed into this Layer</h3>
                    <p class="modal-prose">${escapeHtml(comp.dataFed)}</p>
                </div>
            </div>
        `;
    } else if (section === "diagram" || section === "tensors") {
        const spec = comp.inputOutputSpec || {};
        const specRows = Object.entries(spec)
            .map(([k, v]) => `<tr><td><strong>${escapeHtml(k)}</strong></td><td><code>${escapeHtml(v)}</code></td></tr>`)
            .join("");

        body.innerHTML = `
            <div class="modal-section-block">
                <div class="modal-text-group">
                    <h3 class="modal-group-title">Tensor Data Flow Diagram</h3>
                    <div class="modal-diagram-container">
                        ${comp.diagram || '<p>Diagram not available</p>'}
                    </div>
                    <div class="diagram-scroll-hint">
                        <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2">
                            <polyline points="15 18 9 12 15 6"></polyline>
                        </svg>
                        <span>Pan or swipe horizontally to view complete tensor graph</span>
                        <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2">
                            <polyline points="9 18 15 12 9 6"></polyline>
                        </svg>
                    </div>
                </div>

                <div class="modal-text-group">
                    <h3 class="modal-group-title">Input & Output Specifications</h3>
                    <div class="modal-table-scroll-wrap">
                        <table class="modal-spec-table">
                            <tbody>
                                ${specRows}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        `;
    } else if (section === "traversal") {
        const trav = comp.traversalExample;
        if (!trav) {
            body.innerHTML = `<p class="modal-prose">Traversal walkthrough not available for this component.</p>`;
            return;
        }

        const stepsHtml = (trav.steps || [])
            .map((s, idx) => {
                const cleanStage = (s.stage || "").replace(/^\d+\.\s*/, "");
                const conceptText = s.concept || s.conceptSummary || "";
                const impactText = s.impact || "";
                const simpleIn = s.simpleInput || s.input || "";
                const simpleOp = s.simpleProcessing || s.operation || "";
                const simpleOut = s.simpleOutput || s.output || "";
                const mathObj = s.math || {
                    formula: s.operation || "",
                    inputTensor: s.input || "",
                    operation: s.operation || "",
                    outputTensor: s.output || ""
                };

                return `
                <div class="traversal-step-card" data-step-index="${idx}">
                    <div class="traversal-step-header">
                        <div class="step-title-wrap">
                            <span class="step-badge">Step ${idx + 1}</span>
                            <span class="step-title">${escapeHtml(cleanStage)}</span>
                        </div>
                        <button class="trav-math-btn" data-step-idx="${idx}" type="button" title="Toggle detailed tensor mathematics for this step">
                            <span class="math-btn-icon">fx</span>
                            <span class="math-btn-text">Show Math</span>
                        </button>
                    </div>

                    ${conceptText ? `
                        <div class="trav-concept-card">
                            <span class="trav-concept-badge">What is this & What does it do?</span>
                            <p class="trav-concept-text">${escapeHtml(conceptText)}</p>
                        </div>
                    ` : ""}

                    <!-- Simple Theoretical Walkthrough -->
                    <div class="traversal-step-grid simple-grid">
                        <div class="trav-sub-box">
                            <span class="trav-label">Input (What Goes In)</span>
                            <p class="trav-prose-text">${escapeHtml(simpleIn)}</p>
                        </div>
                        <div class="trav-sub-box">
                            <span class="trav-label">Processing (How It Transforms)</span>
                            <p class="trav-prose-text">${escapeHtml(simpleOp)}</p>
                        </div>
                        <div class="trav-sub-box highlight">
                            <span class="trav-label">Output (What Comes Out)</span>
                            <p class="trav-prose-text">${escapeHtml(simpleOut)}</p>
                            ${impactText ? `
                                <div class="trav-impact-badge">
                                    <span class="impact-label">Architectural Impact:</span>
                                    <span class="impact-val">${escapeHtml(impactText)}</span>
                                </div>
                            ` : ""}
                        </div>
                    </div>

                    <!-- Deep Mathematical Traversal (Collapsible) -->
                    <div class="trav-math-panel hidden" id="trav-math-panel-${idx}">
                        <div class="math-panel-header">
                            <span class="math-panel-label">Mathematical & Tensor Formulation</span>
                            ${mathObj.formula ? `<code class="math-formula-tag">${escapeHtml(mathObj.formula)}</code>` : ""}
                        </div>
                        <div class="traversal-step-grid math-grid">
                            <div class="trav-sub-box math-box">
                                <span class="trav-label">Tensor Input</span>
                                <div class="trav-val"><code>${escapeHtml(mathObj.inputTensor || s.input)}</code></div>
                            </div>
                            <div class="trav-sub-box math-box">
                                <span class="trav-label">Matrix Operation</span>
                                <div class="trav-val"><code>${escapeHtml(mathObj.operation || s.operation)}</code></div>
                            </div>
                            <div class="trav-sub-box math-box highlight">
                                <span class="trav-label">Tensor Output Coordinates & Norms</span>
                                <div class="trav-val"><code>${escapeHtml(mathObj.outputTensor || s.output)}</code></div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            })
            .join("");

        body.innerHTML = `
            <div class="modal-section-block">
                <div class="traversal-intro-card">
                    <div class="traversal-intro-top">
                        <div>
                            <span class="modal-sec-sub">SAMPLE TRACE QUERY</span>
                            <p class="traversal-sample-text">"${escapeHtml(trav.sampleText)}"</p>
                        </div>
                        <button class="trav-global-math-btn" id="trav-global-math-toggle" type="button">
                            Show Math on All Steps
                        </button>
                    </div>
                    <p class="traversal-sample-note">Follow how this review is processed step-by-step. Each step explains what happens in plain English, with a "Show Math" button for deep tensor equations and coordinates.</p>
                </div>
                <div class="traversal-steps-list">
                    ${stepsHtml}
                </div>
            </div>
        `;

        // Attach event listeners for local and global math toggle
        let allMathExpanded = false;
        const globalBtn = body.querySelector("#trav-global-math-toggle");

        body.querySelectorAll(".trav-math-btn").forEach((btn) => {
            btn.addEventListener("click", () => {
                const sIdx = btn.dataset.stepIdx;
                const panel = body.querySelector(`#trav-math-panel-${sIdx}`);
                if (!panel) return;

                const isHidden = panel.classList.contains("hidden");
                panel.classList.toggle("hidden", !isHidden);
                btn.classList.toggle("active", isHidden);
                const textSpan = btn.querySelector(".math-btn-text");
                if (textSpan) {
                    textSpan.textContent = isHidden ? "Hide Math" : "Show Math";
                }
            });
        });

        if (globalBtn) {
            globalBtn.addEventListener("click", () => {
                allMathExpanded = !allMathExpanded;
                globalBtn.textContent = allMathExpanded ? "Hide Math on All Steps" : "Show Math on All Steps";
                body.querySelectorAll(".trav-math-panel").forEach((p) => {
                    p.classList.toggle("hidden", !allMathExpanded);
                });
                body.querySelectorAll(".trav-math-btn").forEach((b) => {
                    b.classList.toggle("active", allMathExpanded);
                    const textSpan = b.querySelector(".math-btn-text");
                    if (textSpan) {
                        textSpan.textContent = allMathExpanded ? "Hide Math" : "Show Math";
                    }
                });
            });
        }
    } else if (section === "ablation") {
        body.innerHTML = `
            <div class="modal-section-block">
                <div class="modal-ablation-card">
                    <div class="ablation-alert-header">
                        <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="#ef4444" stroke-width="2">
                            <circle cx="12" cy="12" r="10" />
                            <line x1="12" y1="8" x2="12" y2="12" />
                            <line x1="12" y1="16" x2="12.01" y2="16" />
                        </svg>
                        <span class="ablation-title">Ablation Study: What Happens If Absent?</span>
                    </div>
                    <div class="ablation-body">
                        <p class="modal-prose">${escapeHtml(comp.ablationStudy).replace(/\n/g, '<br>')}</p>
                    </div>
                </div>
            </div>
        `;
    }
}

function updateModalNavButtons() {
    const list = PIPELINE_ORDER[currentSelectedModel] || PIPELINE_ORDER.v3;
    const idx = list.indexOf(activeModalComponentId);

    const prevBtn = $("#modal-btn-prev");
    const nextBtn = $("#modal-btn-next");
    const topPrevBtn = $("#detail-top-btn-prev");
    const topNextBtn = $("#detail-top-btn-next");
    const topPrevLabel = $("#detail-top-prev-label");
    const topNextLabel = $("#detail-top-next-label");

    const hasPrev = idx > 0;
    const hasNext = idx >= 0 && idx < list.length - 1;

    const prevComp = hasPrev ? window.ARCH_COMPONENTS[list[idx - 1]] : null;
    const nextComp = hasNext ? window.ARCH_COMPONENTS[list[idx + 1]] : null;

    if (prevBtn) {
        prevBtn.disabled = !hasPrev;
        prevBtn.textContent = hasPrev ? `← ${prevComp ? prevComp.shortName : 'Previous'}` : "Start of Pipeline";
    }

    if (nextBtn) {
        nextBtn.disabled = !hasNext;
        nextBtn.textContent = hasNext ? `${nextComp ? nextComp.shortName : 'Next'} →` : "End of Pipeline";
    }

    if (topPrevBtn) {
        topPrevBtn.disabled = !hasPrev;
        if (topPrevLabel) {
            topPrevLabel.textContent = hasPrev ? (prevComp ? prevComp.shortName : 'Prev') : 'Start';
        }
    }

    if (topNextBtn) {
        topNextBtn.disabled = !hasNext;
        if (topNextLabel) {
            topNextLabel.textContent = hasNext ? (nextComp ? nextComp.shortName : 'Next') : 'End';
        }
    }
}

function navigateModalComponent(delta) {
    const list = PIPELINE_ORDER[currentSelectedModel] || PIPELINE_ORDER.v3;
    const idx = list.indexOf(activeModalComponentId);
    const targetIdx = idx + delta;

    if (targetIdx >= 0 && targetIdx < list.length) {
        openComponentModal(list[targetIdx], activeModalSection);
    }
}

function closeComponentModal() {
    const overviewView = $("#arch-overview-view");
    const detailView = $("#arch-detail-view");

    if (detailView) detailView.classList.add("hidden");
    if (overviewView) overviewView.classList.remove("hidden");

    // Smoothly scroll back to pipeline position
    const pipelineCard = $(".arch-pipeline-card");
    if (pipelineCard) {
        pipelineCard.scrollIntoView({ behavior: "smooth", block: "start" });
    }
}

function formatNumber(n) {
    if (!n) return "—";
    if (n >= 1e6) return (n / 1e6).toFixed(2) + "M";
    if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
    return n.toString();
}

// ─── Training Curves Chart (Canvas) ─────────────────────────
function drawTrainingChart(history, modelKey) {
    const canvas = $("#history-chart");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const W = rect.width;
    const H = rect.height;
    const pad = { top: 30, right: 20, bottom: 40, left: 50 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    const acc1 = history.accuracy || [];
    const acc2 = history.val_accuracy || [];
    const epochs = Math.max(acc1.length, acc2.length);
    if (epochs === 0) return;

    ctx.clearRect(0, 0, W, H);

    // Grid
    ctx.strokeStyle = "#e5e7eb";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
        const y = pad.top + (plotH / 5) * i;
        ctx.beginPath();
        ctx.moveTo(pad.left, y);
        ctx.lineTo(W - pad.right, y);
        ctx.stroke();

        const val = (1.0 - (i / 5) * 0.5).toFixed(2);
        ctx.fillStyle = "#9ca3af";
        ctx.font = '11px "JetBrains Mono", monospace';
        ctx.textAlign = "right";
        ctx.fillText(val, pad.left - 8, y + 4);
    }

    // X labels
    ctx.textAlign = "center";
    ctx.fillStyle = "#9ca3af";
    for (let i = 0; i < epochs; i++) {
        const x = pad.left + (plotW / (epochs - 1 || 1)) * i;
        ctx.fillText(`Ep ${i + 1}`, x, H - pad.bottom + 18);
    }

    ctx.fillStyle = "#4b5563";
    ctx.font = '12px "Inter", sans-serif';
    ctx.textAlign = "center";
    ctx.fillText("Epoch", W / 2, H - 4);

    ctx.save();
    ctx.translate(14, H / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Validation Accuracy", 0, 0);
    ctx.restore();

    function drawLine(data, color) {
        if (data.length === 0) return;
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.5;

        data.forEach((v, i) => {
            const x = pad.left + (plotW / (epochs - 1 || 1)) * i;
            const normalizedY = (v - 0.5) / 0.5;
            const y = pad.top + plotH * (1 - Math.max(0, Math.min(1, normalizedY)));
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();

        data.forEach((v, i) => {
            const x = pad.left + (plotW / (epochs - 1 || 1)) * i;
            const normalizedY = (v - 0.5) / 0.5;
            const y = pad.top + plotH * (1 - Math.max(0, Math.min(1, normalizedY)));
            ctx.beginPath();
            ctx.arc(x, y, 4, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.fill();
        });
    }

    if (modelKey === "v1") {
        drawLine(acc1, "#58a6ff");
        drawLine(acc2, "#3fb950");
        // Legend
        drawLegend(ctx, pad.left, pad.top - 12, "#58a6ff", "Train Acc", "#3fb950", "Val Acc");
    } else {
        drawLine(acc1, "#3fb950");
        drawLine(acc2, "#f0883e");
        // Legend
        drawLegend(ctx, pad.left, pad.top - 12, "#3fb950", "Sentiment Val Acc", "#f0883e", "Sarcasm Val Acc");
    }
}

function drawLegend(ctx, x, y, c1, label1, c2, label2) {
    ctx.beginPath();
    ctx.arc(x + 12, y, 4, 0, Math.PI * 2);
    ctx.fillStyle = c1;
    ctx.fill();
    ctx.fillStyle = "#8b949e";
    ctx.font = '11px "Inter", sans-serif';
    ctx.textAlign = "left";
    ctx.fillText(label1, x + 22, y + 3);

    ctx.beginPath();
    ctx.arc(x + 150, y, 4, 0, Math.PI * 2);
    ctx.fillStyle = c2;
    ctx.fill();
    ctx.fillStyle = "#8b949e";
    ctx.fillText(label2, x + 160, y + 3);
}

window.addEventListener("resize", () => {
    if (modelsInfoCache) renderSelectedModelDetails(currentSelectedModel);
});
