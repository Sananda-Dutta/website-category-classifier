// ═══════════════════════════════════════════════════════════════
// popup.js  —  Website Category Classifier Extension  v1.0.0
// Calls /safe-check (gives category + confidence + safety in one
// request) then renders Top 3 from a second /classify/url call.
// ═══════════════════════════════════════════════════════════════

const API_BASE = "https://website-category-classifier.onrender.com";

const CATEGORY_EMOJI = {
  Adult: "🔞", Arts: "🎨", Business: "💼", Education: "🎓",
  Gaming: "🎮", Health: "🏥", Kids: "👧", Lifestyle: "🏠",
  News: "📰", Recreation: "🏕️", Technology: "💻",
  Unknown: "🌐", Error: "❌",
};

// ── DOM refs ──────────────────────────────────────────────────
const $ = id => document.getElementById(id);

const states = {
  loading:     $("state-loading"),
  error:       $("state-error"),
  unsupported: $("state-unsupported"),
  result:      $("state-result"),
};

// ── Show exactly one state panel ─────────────────────────────
function showState(name) {
  Object.entries(states).forEach(([k, el]) => {
    el.classList.toggle("hidden", k !== name);
  });
}

// ── API status dot ────────────────────────────────────────────
async function checkApiStatus() {
  const dot = $("api-status");
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(6000) });
    dot.className = res.ok
      ? "status-dot status-online"
      : "status-dot status-offline";
  } catch {
    dot.className = "status-dot status-offline";
  }
}

// ── Render Top 3 bars ─────────────────────────────────────────
function renderTop3(top3) {
  const container = $("top3-list");
  container.innerHTML = "";

  top3.forEach((item, i) => {
    const emoji = CATEGORY_EMOJI[item.category] || "🌐";
    const pct   = item.confidence;

    const row = document.createElement("div");
    row.className = "top3-item";
    row.innerHTML = `
      <span class="top3-label">${emoji} ${item.category}</span>
      <div class="top3-bar-track">
        <div class="top3-bar-fill ${i === 0 ? "top1" : ""}"
             style="width:${pct}%"></div>
      </div>
      <span class="top3-pct">${pct}%</span>
    `;
    container.appendChild(row);
  });
}

// ── Render safety verdict bar ─────────────────────────────────
function renderVerdict(data) {
  const bar = $("verdict-bar");
  bar.className = "verdict-bar";   // reset classes

  if (data.adult_flag) {
    bar.classList.add("verdict-unsafe");
    bar.textContent = "🔴 ADULT — block recommended";
  } else if (data.kids_safe) {
    bar.classList.add("verdict-kids");
    bar.textContent = "🟢 KIDS SAFE";
  } else if (data.safe_for_kids) {
    bar.classList.add("verdict-kids");
    bar.textContent = "🟡 SAFE FOR KIDS";
  } else {
    bar.classList.add("verdict-safe");
    bar.textContent = "🟢 SAFE";
  }

  $("met-safe").textContent  = data.safe          ? "✅ Yes" : "🚫 No";
  $("met-kids").textContent  = data.kids_safe      ? "✅ Yes" : "🚫 No";
  $("met-sfk").textContent   = data.safe_for_kids  ? "✅ Yes" : "🚫 No";
}

// ── Main classify + render flow ───────────────────────────────
async function classifyCurrentTab(url) {
  showState("loading");

  try {
    // ── Call /safe-check → gives category + safety in one shot ──
    const safeRes = await fetch(`${API_BASE}/safe-check`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ url }),
      signal:  AbortSignal.timeout(40000),   // 40s — Render cold start
    });

    if (!safeRes.ok) {
      const err = await safeRes.json().catch(() => ({}));
      throw new Error(err.detail || `API error ${safeRes.status}`);
    }

    const safeData = await safeRes.json();

    // ── Call /classify/url to get Top 3 ──────────────────────
    const classRes = await fetch(`${API_BASE}/classify/url`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ url }),
      signal:  AbortSignal.timeout(40000),
    });

    const classData = classRes.ok ? await classRes.json() : null;

    // ── Render result card ────────────────────────────────────
    const category   = safeData.category   || "Unknown";
    const confidence = safeData.confidence || 0;
    const method     = safeData.method     || "ml_model";
    const timeMs     = safeData.time_ms    || "—";
    const emoji      = CATEGORY_EMOJI[category] || "🌐";

    $("result-emoji").textContent    = emoji;
    $("result-category").textContent = category;
    $("badge-conf").textContent      = `${confidence}% confidence`;
    $("badge-time").textContent      = `${timeMs} ms`;
    $("method-badge").textContent    = method === "domain_shortcut"
      ? "Domain shortcut"
      : "DistilBERT";

    renderVerdict(safeData);

    // Top 3 — use classify/url response if available, else fake it
    const top3 = classData?.top3 || [
      { category, confidence }
    ];
    renderTop3(top3);

    // ── Cache result for badge (background.js reads this) ────
    chrome.storage.local.set({
      [`cache:${url}`]: {
        category,
        confidence,
        timestamp: Date.now(),
      }
    });

    showState("result");

  } catch (err) {
    $("error-text").textContent = err.message.includes("timeout")
      ? "Request timed out — API may be waking up. Click retry in 30s."
      : err.message || "Could not reach API";
    showState("error");
  }
}

// ── Retry button ──────────────────────────────────────────────
$("btn-retry").addEventListener("click", () => {
  chrome.tabs.query({ active: true, currentWindow: true }, tabs => {
    const url = tabs[0]?.url;
    if (url) classifyCurrentTab(url);
  });
});

// ── Reclassify button ─────────────────────────────────────────
$("btn-reclassify").addEventListener("click", () => {
  chrome.tabs.query({ active: true, currentWindow: true }, tabs => {
    const url = tabs[0]?.url;
    if (url) classifyCurrentTab(url);
  });
});

// ── Entry point ───────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {

  checkApiStatus();   // ping /health for the status dot

  chrome.tabs.query({ active: true, currentWindow: true }, tabs => {
    const tab = tabs[0];
    const url = tab?.url || "";

    // Show URL in the bar (truncated)
    const display = url.replace(/^https?:\/\/(www\.)?/, "").slice(0, 55);
    $("current-url").textContent = display || "—";

    // Only classify real http/https pages
    if (!url.startsWith("http://") && !url.startsWith("https://")) {
      showState("unsupported");
      return;
    }

    // Check cache first (5 min TTL) — avoids re-calling API on re-open
    chrome.storage.local.get([`cache:${url}`], result => {
      const cached = result[`cache:${url}`];
      const TTL    = 5 * 60 * 1000;   // 5 minutes

      if (cached && (Date.now() - cached.timestamp) < TTL) {
        // Serve from cache — still show full result by re-classifying silently
        // but show cached badge immediately
        $("result-emoji").textContent    = CATEGORY_EMOJI[cached.category] || "🌐";
        $("result-category").textContent = cached.category;
        $("badge-conf").textContent      = `${cached.confidence}% confidence`;
        $("badge-time").textContent      = "cached";
        $("method-badge").textContent    = "cached";
      }

      // Always fetch fresh data (cache just prevents blank flash)
      classifyCurrentTab(url);
    });
  });
});