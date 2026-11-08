/**
 * SentimentLens — Multi-Model Comparative Application Logic
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
    initInput();
    initSampleButtons();
    initAnalyze();
    initHistory();
    initModelSelector();
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

// ─── Tab Navigation ─────────────────────────────────────────
function initTabs() {
    $$(".tab-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            const tabId = btn.dataset.tab;
            $$(".tab-btn").forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");

            $$(".tab-panel").forEach((p) => p.classList.remove("active"));
            $(`#panel-${tabId}`).classList.add("active");

            if (tabId === "model" && modelsInfoCache) {
                renderSelectedModelDetails(currentSelectedModel);
            }
        });
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
        addToComparativeHistory(text, data);
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
                .slice(0, 3)
                .map((t) => `<span class="subword-chip">${escapeHtml(t.subword.trim() || t.subword)}</span>`)
                .join("");
        } else {
            subwordsV2.innerHTML = '<span class="subword-chip">—</span>';
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
            invStatus.style.display = "inline-block";
        } else {
            invStatus.textContent = "Direct Polarity";
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
                .map((t) => {
                    const cleanWord = t.subword.trim() || t.subword;
                    const wt = Math.round(t.attention_weight * 100);
                    return `<span class="subword-chip" title="${wt}% attention">${escapeHtml(cleanWord)} <span class="subword-weight">${wt}%</span></span>`;
                })
                .join("");
        } else {
            subwordsV3.innerHTML = '<span class="subword-chip">—</span>';
        }
    }
}

// ─── Comparative History ────────────────────────────────────
function initHistory() {
    clearHistoryBtn.addEventListener("click", () => {
        comparativeHistory = [];
        localStorage.setItem("sentimentHistory_multi", "[]");
        renderHistory();
    });
    renderHistory();
}

function addToComparativeHistory(text, data) {
    const v1Sent = data.v1?.sentiment?.label || "N/A";
    const v2Sent = data.v2?.sentiment?.label || "N/A";
    const v2Sarc = data.v2?.sarcasm?.label || "N/A";
    const v3Sent = data.v3?.sentiment?.label || "N/A";
    const v3Sarc = data.v3?.sarcasm?.label || "N/A";

    comparativeHistory.unshift({
        text: text.slice(0, 100),
        v1Sent,
        v2Sent,
        v2Sarc,
        v3Sent,
        v3Sarc,
        interpretation: data.v3?.interpretation || data.summary?.flagship_interpretation || "",
        timestamp: Date.now(),
    });

    if (comparativeHistory.length > 20) comparativeHistory = comparativeHistory.slice(0, 20);
    localStorage.setItem("sentimentHistory_multi", JSON.stringify(comparativeHistory));
    renderHistory();
}

function renderHistory() {
    if (comparativeHistory.length === 0) {
        historyList.innerHTML =
            '<p class="history-empty">No comparative analyses yet. Submissions will appear here.</p>';
        return;
    }

    historyList.innerHTML = comparativeHistory
        .map(
            (item) => `
        <div class="history-item" title="${escapeHtml(item.text)}">
            <span class="history-text">${escapeHtml(item.text)}</span>
            <span class="history-sarcasm-tag">V3: ${item.v3Sarc} / ${item.v3Sent}</span>
            <span class="history-label ${item.v3Sent === "Positive" ? "pos" : "neg"}">${item.v3Sent}</span>
        </div>
    `
        )
        .join("");

    $$(".history-item").forEach((el, i) => {
        el.addEventListener("click", () => {
            textInput.value = comparativeHistory[i].text;
            textInput.dispatchEvent(new Event("input"));
            $('[data-tab="analyze"]').click();
        });
    });
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

async function loadModelsInfo() {
    try {
        const res = await fetch(`${API_BASE}/models-info`);
        if (!res.ok) throw new Error("Models info not available");
        modelsInfoCache = await res.json();
        renderSelectedModelDetails(currentSelectedModel);
    } catch (err) {
        console.log("Failed to load models-info:", err);
    }
}

function renderSelectedModelDetails(modelKey) {
    if (!modelsInfoCache || !modelsInfoCache[modelKey]) return;
    const info = modelsInfoCache[modelKey];

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
                (l) => `
            <tr>
                <td>${escapeHtml(l.name)}</td>
                <td>${escapeHtml(l.type)}</td>
                <td>${escapeHtml(l.output_shape)}</td>
                <td>${l.params.toLocaleString()}</td>
            </tr>
        `
            )
            .join("");
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
    if (modelKey === "v1") {
        pipeline.innerHTML = `
            <div class="arch-stage"><div class="arch-node input-node"><span class="arch-icon">TXT</span><span class="arch-name">Input</span><span class="arch-detail">Raw Review</span></div><div class="arch-arrow">→</div></div>
            <div class="arch-stage"><div class="arch-node preprocess-node"><span class="arch-icon">TOK</span><span class="arch-name">Tokenizer</span><span class="arch-detail">Whitespace (35K)</span></div><div class="arch-arrow">→</div></div>
            <div class="arch-stage"><div class="arch-node embed-node"><span class="arch-icon">EMB</span><span class="arch-name">Embedding</span><span class="arch-detail">128-dim trainable</span></div><div class="arch-arrow">→</div></div>
            <div class="arch-stage"><div class="arch-node cnn-node"><span class="arch-icon">CNN</span><span class="arch-name">Conv1D</span><span class="arch-detail">k=3,5,7 (96 filters)</span></div><div class="arch-arrow">→</div></div>
            <div class="arch-stage"><div class="arch-node lstm-node"><span class="arch-icon">LSTM</span><span class="arch-name">BiLSTM</span><span class="arch-detail">64 units</span></div><div class="arch-arrow">→</div></div>
            <div class="arch-stage"><div class="arch-node embed-node"><span class="arch-icon">ATTN</span><span class="arch-name">Attention</span><span class="arch-detail">Bahdanau</span></div><div class="arch-arrow">→</div></div>
            <div class="arch-stage"><div class="arch-node output-node"><span class="arch-icon">OUT</span><span class="arch-name">Output</span><span class="arch-detail">Single Sentiment Head</span></div></div>
        `;
    } else if (modelKey === "v2") {
        pipeline.innerHTML = `
            <div class="arch-stage"><div class="arch-node input-node"><span class="arch-icon">TXT</span><span class="arch-name">Input</span><span class="arch-detail">Raw Review</span></div><div class="arch-arrow">→</div></div>
            <div class="arch-stage"><div class="arch-node preprocess-node"><span class="arch-icon">BPE</span><span class="arch-name">BPE Tokenizer</span><span class="arch-detail">Rust BPE (30K)</span></div><div class="arch-arrow">→</div></div>
            <div class="arch-stage"><div class="arch-node embed-node"><span class="arch-icon">EMB</span><span class="arch-name">Embedding</span><span class="arch-detail">300-dim GloVe</span></div><div class="arch-arrow">→</div></div>
            <div class="arch-stage"><div class="arch-node cnn-node"><span class="arch-icon">CNN</span><span class="arch-name">Multi-Scale CNN</span><span class="arch-detail">k=2,3,5,7 (256 filters)</span></div><div class="arch-arrow">→</div></div>
            <div class="arch-stage"><div class="arch-node lstm-node"><span class="arch-icon">LSTM</span><span class="arch-name">BiLSTM</span><span class="arch-detail">2 Layers (256-dim)</span></div><div class="arch-arrow">→</div></div>
            <div class="arch-stage"><div class="arch-node incongruity-node"><span class="arch-icon">INC</span><span class="arch-name">Incongruity</span><span class="arch-detail">Sequence Split Diff</span></div><div class="arch-arrow">→</div></div>
            <div class="arch-stage"><div class="arch-node embed-node"><span class="arch-icon">ATTN</span><span class="arch-name">Self-Attention</span><span class="arch-detail">4-Head Multi-Head</span></div><div class="arch-arrow">→</div></div>
            <div class="arch-stage"><div class="arch-node output-node"><span class="arch-icon">OUT</span><span class="arch-name">Dual Heads</span><span class="arch-detail">Sarcasm + Sentiment</span></div></div>
        `;
    } else {
        pipeline.innerHTML = `
            <div class="arch-stage"><div class="arch-node input-node"><span class="arch-icon">TXT</span><span class="arch-name">Input</span><span class="arch-detail">Raw Text</span></div><div class="arch-arrow">→</div></div>
            <div class="arch-stage"><div class="arch-node preprocess-node"><span class="arch-icon">BPE</span><span class="arch-name">BPE Tokenizer</span><span class="arch-detail">Rust BPE (30K)</span></div><div class="arch-arrow">→</div></div>
            <div class="arch-stage"><div class="arch-node embed-node"><span class="arch-icon">EMB</span><span class="arch-name">Embedding</span><span class="arch-detail">300-dim GloVe</span></div><div class="arch-arrow">→</div></div>
            <div class="arch-stage"><div class="arch-node cnn-node"><span class="arch-icon">CNN</span><span class="arch-name">Multi-Scale CNN</span><span class="arch-detail">k=2,3,5,7 (256 filters)</span></div><div class="arch-arrow">→</div></div>
            <div class="arch-stage"><div class="arch-node lstm-node"><span class="arch-icon">LSTM</span><span class="arch-name">Deep BiLSTM</span><span class="arch-detail">2 Layers (256-dim)</span></div><div class="arch-arrow">→</div></div>
            <div class="arch-stage"><div class="arch-node incongruity-node"><span class="arch-icon">INC</span><span class="arch-name">Incongruity</span><span class="arch-detail">Split & Hadamard Prod</span></div><div class="arch-arrow">→</div></div>
            <div class="arch-stage"><div class="arch-node embed-node"><span class="arch-icon">ATTN</span><span class="arch-name">Multi-Head Attn</span><span class="arch-detail">4 Heads Scaled Dot</span></div><div class="arch-arrow">→</div></div>
            <div class="arch-stage"><div class="arch-node output-node"><span class="arch-icon">GATE</span><span class="arch-name">Gated Inversion</span><span class="arch-detail">Inverts Mock Praise</span></div></div>
        `;
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
    ctx.strokeStyle = "#21262d";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
        const y = pad.top + (plotH / 5) * i;
        ctx.beginPath();
        ctx.moveTo(pad.left, y);
        ctx.lineTo(W - pad.right, y);
        ctx.stroke();

        const val = (1.0 - (i / 5) * 0.5).toFixed(2);
        ctx.fillStyle = "#484f58";
        ctx.font = '11px "JetBrains Mono", monospace';
        ctx.textAlign = "right";
        ctx.fillText(val, pad.left - 8, y + 4);
    }

    // X labels
    ctx.textAlign = "center";
    ctx.fillStyle = "#484f58";
    for (let i = 0; i < epochs; i++) {
        const x = pad.left + (plotW / (epochs - 1 || 1)) * i;
        ctx.fillText(`Ep ${i + 1}`, x, H - pad.bottom + 18);
    }

    ctx.fillStyle = "#8b949e";
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
