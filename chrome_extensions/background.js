// ═══════════════════════════════════════════════════════════════
// background.js  —  Service Worker  v1.0.0
//
// Responsibilities:
//   1. Listen for tab navigation events
//   2. Call /classify/url for every http/https tab
//   3. Update the extension badge with category initial + color
//   4. Cache results in chrome.storage.local (5 min TTL)
//   5. Skip internal pages, PDFs, already-cached URLs
// ═══════════════════════════════════════════════════════════════

const API_BASE  = "https://website-category-classifier.onrender.com";
const CACHE_TTL = 5 * 60 * 1000;   // 5 minutes

// Badge color per category — matches popup.css palette
const BADGE_COLORS = {
  Adult:      "#cc3333",
  Arts:       "#9b6b9b",
  Business:   "#2d7a4a",
  Education:  "#2d6a9f",
  Gaming:     "#7a3d9b",
  Health:     "#2d8a6a",
  Kids:       "#4a7abf",
  Lifestyle:  "#8a6a2d",
  News:       "#5a5a8a",
  Recreation: "#2d7a5a",
  Technology: "#2d5a9f",
  Unknown:    "#555555",
  Error:      "#993333",
};

// Short badge text per category (max 4 chars for MV3 badge)
const BADGE_TEXT = {
  Adult:      "18+",
  Arts:       "ART",
  Business:   "BIZ",
  Education:  "EDU",
  Gaming:     "GAM",
  Health:     "MED",
  Kids:       "KID",
  Lifestyle:  "LIF",
  News:       "NEWS",
  Recreation: "REC",
  Technology: "TECH",
  Unknown:    "?",
  Error:      "ERR",
};

// ── Set badge on a specific tab ───────────────────────────────
function setBadge(tabId, category, confidence) {
  const text  = BADGE_TEXT[category]  || "?";
  const color = BADGE_COLORS[category] || "#555";

  chrome.action.setBadgeText({       text,   tabId });
  chrome.action.setBadgeBackgroundColor({ color, tabId });
  chrome.action.setTitle({
    title: `${category} — ${confidence}% confidence\nWebsite Category Classifier`,
    tabId,
  });
}

function clearBadge(tabId) {
  chrome.action.setBadgeText({ text: "", tabId });
  chrome.action.setTitle({ title: "Website Category Classifier", tabId });
}

// ── Check cache ───────────────────────────────────────────────
async function getCached(url) {
  return new Promise(resolve => {
    chrome.storage.local.get([`cache:${url}`], result => {
      const cached = result[`cache:${url}`];
      if (cached && (Date.now() - cached.timestamp) < CACHE_TTL) {
        resolve(cached);
      } else {
        resolve(null);
      }
    });
  });
}

function setCache(url, category, confidence) {
  chrome.storage.local.set({
    [`cache:${url}`]: { category, confidence, timestamp: Date.now() }
  });
}

// ── Classify a URL and update the tab badge ───────────────────
async function classifyTab(tabId, url) {
  // Only classify real web pages
  if (!url || !url.startsWith("http")) {
    clearBadge(tabId);
    return;
  }

  // Skip PDFs, images, and other non-HTML resources
  const skipExtensions = [".pdf", ".png", ".jpg", ".jpeg", ".gif", ".svg",
                          ".mp4", ".mp3", ".zip", ".exe"];
  if (skipExtensions.some(ext => url.toLowerCase().includes(ext))) {
    clearBadge(tabId);
    return;
  }

  // Check cache first — avoids hitting API on every page reload
  const cached = await getCached(url);
  if (cached) {
    setBadge(tabId, cached.category, cached.confidence);
    return;
  }

  // Show loading badge while API call is in flight
  chrome.action.setBadgeText({ text: "...", tabId });
  chrome.action.setBadgeBackgroundColor({ color: "#2d6a9f", tabId });

  try {
    const res = await fetch(`${API_BASE}/classify/url`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ url }),
      signal:  AbortSignal.timeout(45000),   // 45s — generous for cold starts
    });

    if (!res.ok) {
      setBadge(tabId, "Error", 0);
      return;
    }

    const data       = await res.json();
    const category   = data.category   || "Unknown";
    const confidence = data.confidence || 0;

    setCache(url, category, confidence);
    setBadge(tabId, category, confidence);

  } catch (err) {
    // Network error or timeout — don't show error badge, just clear it
    // so it doesn't alarm the user for every Render cold start
    clearBadge(tabId);
    console.warn("WCC extension: classify failed for", url, err.message);
  }
}

// ── Listen for tab events ─────────────────────────────────────

// Fires when a tab finishes loading a new URL
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  // Only trigger when the page has fully loaded, not on every redirect
  if (changeInfo.status === "complete" && tab.url) {
    classifyTab(tabId, tab.url);
  }
});

// Fires when the user switches to a different tab —
// re-set the badge in case it was cleared by another extension
chrome.tabs.onActivated.addListener(async ({ tabId }) => {
  const tab = await chrome.tabs.get(tabId).catch(() => null);
  if (!tab?.url) return;

  const cached = await getCached(tab.url);
  if (cached) {
    setBadge(tabId, cached.category, cached.confidence);
  }
});

// Fires when the extension is first installed or updated
chrome.runtime.onInstalled.addListener(() => {
  console.log("Website Category Classifier extension installed.");

  // Classify whatever tab is currently open
  chrome.tabs.query({ active: true, currentWindow: true }, tabs => {
    if (tabs[0]) classifyTab(tabs[0].id, tabs[0].url);
  });
});