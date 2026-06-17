// ═══════════════════════════════════════════════════════════════
// popup.js  —  v1.1.0
//
// v1.1.0 fixes:
//   - normalizeUrl() applied before API calls — fixes YouTube
//     /watch?v= and any other query-param-heavy URLs
//   - 404 / 422 responses handled with friendly messages
//     instead of raw FastAPI error strings
//   - Local shortcuts checked first — popup is instant for
//     YouTube, Instagram, etc. without waiting for API
//   - Error messages are human-readable in all cases
// ═══════════════════════════════════════════════════════════════

const API_BASE = "https://website-category-classifier.onrender.com";

const CATEGORY_EMOJI = {
  Adult:"🔞", Arts:"🎨", Business:"💼", Education:"🎓",
  Gaming:"🎮", Health:"🏥", Kids:"👧", Lifestyle:"🏠",
  News:"📰", Recreation:"🏕️", Technology:"💻",
  Unknown:"🌐", Error:"❌",
};

// Mirror of background.js LOCAL_SHORTCUTS
const LOCAL_SHORTCUTS = {
  "youtube.com":"Arts",   "youtu.be":"Arts",
  "netflix.com":"Arts",   "primevideo.com":"Arts",
  "hotstar.com":"Arts",   "disneyplus.com":"Arts",
  "zee5.com":"Arts",      "sonyliv.com":"Arts",
  "twitch.tv":"Gaming",   "instagram.com":"Lifestyle",
  "twitter.com":"News",   "x.com":"News",
  "facebook.com":"News",  "linkedin.com":"Business",
  "reddit.com":"News",    "pinterest.com":"Lifestyle",
  "amazon.com":"Business","amazon.in":"Business",
  "flipkart.com":"Business","meesho.com":"Business",
  "ndtv.com":"News",      "bbc.com":"News","bbc.co.uk":"News",
  "cricbuzz.com":"Recreation","github.com":"Technology",
  "byjus.com":"Education","wikipedia.org":"Education",
  "practo.com":"Health",  "gaana.com":"Arts",
  "spotify.com":"Arts",   "dream11.com":"Gaming",
  "zomato.com":"Lifestyle","swiggy.com":"Lifestyle",
};

// ── Helpers ───────────────────────────────────────────────────
const $ = id => document.getElementById(id);

function normalizeUrl(url) {
  try {
    const u = new URL(url);
    return `${u.protocol}//${u.hostname}${u.pathname}`.replace(/\/$/, "");
  } catch { return url; }
}

function getDomain(url) {
  try { return new URL(url).hostname.replace(/^www\./, ""); }
  catch { return ""; }
}

function showState(name) {
  ["loading","error","unsupported","result"].forEach(s => {
    $(s === name ? `state-${s}` : `state-${s}`)
      .classList.toggle("hidden", s !== name);
  });
}

// ── API status dot ────────────────────────────────────────────
async function checkApiStatus() {
  const dot = $("api-status");
  try {
    const res = await fetch(`${API_BASE}/health`,
      { signal: AbortSignal.timeout(6000) });
    dot.className = `status-dot ${res.ok ? "status-online" : "status-offline"}`;
  } catch {
    dot.className = "status-dot status-offline";
  }
}

// ── Render Top 3 bars ─────────────────────────────────────────
function renderTop3(top3) {
  const container = $("top3-list");
  container.innerHTML = "";
  top3.forEach((item, i) => {
    const row = document.createElement("div");
    row.className = "top3-item";
    row.innerHTML = `
      <span class="top3-label">
        ${CATEGORY_EMOJI[item.category]||"🌐"} ${item.category}
      </span>
      <div class="top3-bar-track">
        <div class="top3-bar-fill ${i===0?"top1":""}"
             style="width:${item.confidence}%"></div>
      </div>
      <span class="top3-pct">${item.confidence}%</span>`;
    container.appendChild(row);
  });
}

// ── Render safety verdict ─────────────────────────────────────
function renderVerdict(data) {
  const bar = $("verdict-bar");
  bar.className = "verdict-bar";
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
  $("met-safe").textContent = data.safe         ? "✅ Yes" : "🚫 No";
  $("met-kids").textContent = data.kids_safe    ? "✅ Yes" : "🚫 No";
  $("met-sfk").textContent  = data.safe_for_kids? "✅ Yes" : "🚫 No";
}

// ── Show result from local shortcut (instant, no API) ─────────
function showLocalShortcut(url, category) {
  const safeForKids = ["Education","Kids","Arts","Recreation"].includes(category);
  const isAdult     = category === "Adult";

  $("result-emoji").textContent    = CATEGORY_EMOJI[category] || "🌐";
  $("result-category").textContent = category;
  $("badge-conf").textContent      = "99% confidence";
  $("badge-time").textContent      = "instant";
  $("method-badge").textContent    = "Domain shortcut";

  renderVerdict({
    adult_flag:    isAdult,
    kids_safe:     category === "Kids",
    safe_for_kids: safeForKids,
    safe:          !isAdult,
  });

  renderTop3([{ category, confidence: 99 }]);
  showState("result");
}

// ── Human-readable error messages ────────────────────────────
function friendlyError(status, message) {
  if (!navigator.onLine)
    return "No internet connection.";
  if (status === 404 || message?.includes("not found") || message?.includes("Not Found"))
    return "This page could not be classified — it may require login or block scrapers.";
  if (status === 422)
    return "Could not extract content from this URL. Try a simpler page URL.";
  if (status === 429)
    return "Too many requests — wait 1 minute and retry.";
  if (status >= 500)
    return "API server error. Check render.com status or retry in 30 seconds.";
  if (message?.includes("timeout") || message?.includes("TimeoutError"))
    return "Request timed out — API may be waking up (cold start). Retry in 30 seconds.";
  return message || "Could not reach API.";
}

// ── Main classify flow ────────────────────────────────────────
async function classifyUrl(originalUrl) {
  showState("loading");

  const domain      = getDomain(originalUrl);
  const normalUrl   = normalizeUrl(originalUrl);

  // ── 1. Check local shortcuts first — instant result ───────
  if (LOCAL_SHORTCUTS[domain]) {
    showLocalShortcut(originalUrl, LOCAL_SHORTCUTS[domain]);
    return;
  }

  try {
    // ── 2. /safe-check with normalized URL ────────────────
    const safeRes = await fetch(`${API_BASE}/safe-check`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ url: normalUrl }),
      signal:  AbortSignal.timeout(40000),
    });

    if (!safeRes.ok) {
      let detail = "";
      try { detail = (await safeRes.json()).detail || ""; } catch {}
      throw { status: safeRes.status, message: detail };
    }

    const safeData = await safeRes.json();

    // ── 3. /classify/url for Top 3 ────────────────────────
    const classRes = await fetch(`${API_BASE}/classify/url`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ url: normalUrl }),
      signal:  AbortSignal.timeout(40000),
    });
    const classData = classRes.ok ? await classRes.json() : null;

    // ── 4. Render ──────────────────────────────────────────
    const category   = safeData.category   || "Unknown";
    const confidence = safeData.confidence || 0;
    const timeMs     = safeData.time_ms    || "—";
    const method     = safeData.method     || "ml_model";

    $("result-emoji").textContent    = CATEGORY_EMOJI[category] || "🌐";
    $("result-category").textContent = category;
    $("badge-time").textContent      = `${timeMs} ms`;

    if (method === "combined_features" || method === "domain_shortcut" || method === "path_shortcut") {
        $("badge-conf").textContent   = `${confidence}% confidence`;
        $("method-badge").textContent = method === "domain_shortcut" ? "Domain shortcut" : "DistilBERT";
        $("method-badge").style.backgroundColor = "#28a745"; 
        $("method-badge").style.color = "#fff";
    } else {
        $("badge-conf").textContent   = "Low-Confidence Guess";
        $("method-badge").textContent = "Limited Content Fallback";
        $("method-badge").style.backgroundColor = "#ffc107"; 
        $("method-badge").style.color = "#000";             
    }
    // ===========================================================================
    renderVerdict(safeData);
    renderTop3(classData?.top3 || [{ category, confidence }]);

    // Cache for background.js badge
    chrome.storage.local.set({
      [`cache:${normalUrl}`]: { category, confidence, timestamp: Date.now() }
    });

    showState("result");

  } catch (err) {
    $("error-text").textContent = friendlyError(err.status, err.message);
    showState("error");
  }
}

// ── Retry / Reclassify buttons ────────────────────────────────
function getCurrentUrl(cb) {
  chrome.tabs.query({ active: true, currentWindow: true }, tabs => {
    cb(tabs[0]?.url || "");
  });
}

$("btn-retry").addEventListener("click", () =>
  getCurrentUrl(url => { if (url) classifyUrl(url); }));

$("btn-reclassify").addEventListener("click", () =>
  getCurrentUrl(url => { if (url) classifyUrl(url); }));

// ── Entry point ───────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  checkApiStatus();

  getCurrentUrl(url => {
    const display = url.replace(/^https?:\/\/(www\.)?/, "").slice(0, 55);
    $("current-url").textContent = display || "—";

    if (!url.startsWith("http://") && !url.startsWith("https://")) {
      showState("unsupported");
      return;
    }

    classifyUrl(url);
  });
});