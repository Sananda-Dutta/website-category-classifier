// ═══════════════════════════════════════════════════════════════
// background.js  —  Service Worker  v1.1.0
//
// v1.1.0 fixes:
//   - normalizeUrl() strips query params + fragments before
//     cache lookup and API call — fixes YouTube /watch?v= problem
//   - classify() sends normalized URL to API (domain shortcuts
//     match correctly) but keeps original URL for display
//   - Better error handling — timeout vs network vs API errors
//     shown differently in badge
//   - Extended shortcut awareness: badge auto-shows for YouTube,
//     Instagram, Twitter etc. without hitting the API at all
// ═══════════════════════════════════════════════════════════════

const API_BASE  = "https://website-category-classifier.onrender.com";
const CACHE_TTL = 5 * 60 * 1000;   // 5 minutes

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

const BADGE_TEXT = {
  Adult: "18+", Arts: "ART", Business: "BIZ", Education: "EDU",
  Gaming: "GAM", Health: "MED", Kids: "KID", Lifestyle: "LIF",
  News: "NEWS", Recreation: "REC", Technology: "TECH",
  Unknown: "?", Error: "ERR",
};

// ── Client-side shortcuts — mirrors api.py DOMAIN_SHORTCUTS ──
// Keeps the badge instant for well-known sites without any API call
const LOCAL_SHORTCUTS = {
  "youtube.com":       "Arts",    "youtu.be":          "Arts",
  "netflix.com":       "Arts",    "primevideo.com":    "Arts",
  "hotstar.com":       "Arts",    "disneyplus.com":    "Arts",
  "zee5.com":          "Arts",    "sonyliv.com":       "Arts",
  "voot.com":          "Arts",    "twitch.tv":         "Gaming",
  "dailymotion.com":   "Arts",    "instagram.com":     "Lifestyle",
  "twitter.com":       "News",    "x.com":             "News",
  "facebook.com":      "News",    "linkedin.com":      "Business",
  "reddit.com":        "News",    "pinterest.com":     "Lifestyle",
  "snapchat.com":      "Lifestyle","whatsapp.com":     "Technology",
  "amazon.com":        "Business","amazon.in":         "Business",
  "flipkart.com":      "Business","meesho.com":        "Business",
  "myntra.com":        "Business","ajio.com":          "Business",
  "ndtv.com":          "News",    "thehindu.com":      "News",
  "bbc.com":           "News",    "bbc.co.uk":         "News",
  "cricbuzz.com":      "Recreation","espncricinfo.com":"Recreation",
  "github.com":        "Technology","stackoverflow.com":"Technology",
  "byjus.com":         "Education","khanacademy.org":  "Education",
  "wikipedia.org":     "Education","practo.com":       "Health",
  "gaana.com":         "Arts",    "spotify.com":       "Arts",
  "dream11.com":       "Gaming",  "zomato.com":        "Lifestyle",
  "swiggy.com":        "Lifestyle",
};

// ── Normalize URL for consistent caching and API calls ────────
// Strips query string and fragment — ?v=xyz, #section etc.
// youtube.com/watch?v=ABC and youtube.com/watch?v=XYZ
// both become youtube.com/watch → same cache key, same API call
function normalizeUrl(url) {
  try {
    const u = new URL(url);
    // Keep only scheme + hostname + pathname (no query, no hash)
    // Also strip trailing slash for consistency
    const normalized = `${u.protocol}//${u.hostname}${u.pathname}`.replace(/\/$/, "");
    return normalized;
  } catch {
    return url;
  }
}

// ── Extract bare domain from URL ──────────────────────────────
function getDomain(url) {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return "";
  }
}

// ── Badge helpers ─────────────────────────────────────────────
function setBadge(tabId, category, confidence) {
  const text  = BADGE_TEXT[category]  || "?";
  const color = BADGE_COLORS[category] || "#555";
  chrome.action.setBadgeText({ text, tabId });
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

// ── Cache helpers ─────────────────────────────────────────────
function cacheKey(url) {
  return `cache:${normalizeUrl(url)}`;
}

async function getCached(url) {
  return new Promise(resolve => {
    chrome.storage.local.get([cacheKey(url)], result => {
      const cached = result[cacheKey(url)];
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
    [cacheKey(url)]: { category, confidence, timestamp: Date.now() }
  });
}

// ── Main classify function ────────────────────────────────────
async function classifyTab(tabId, url) {
  if (!url || !url.startsWith("http")) {
    clearBadge(tabId);
    return;
  }

  // Skip non-HTML resources
  const skipExts = [".pdf",".png",".jpg",".jpeg",".gif",".svg",
                    ".mp4",".mp3",".zip",".exe",".woff",".ttf"];
  if (skipExts.some(ext => url.toLowerCase().split("?")[0].endsWith(ext))) {
    clearBadge(tabId);
    return;
  }

  const domain = getDomain(url);

  // ── 1. Client-side shortcut — instant, no API call ───────
  if (LOCAL_SHORTCUTS[domain]) {
    const category = LOCAL_SHORTCUTS[domain];
    setCache(url, category, 99);
    setBadge(tabId, category, 99);
    return;
  }

  // ── 2. Check local cache ──────────────────────────────────
  const cached = await getCached(url);
  if (cached) {
    setBadge(tabId, cached.category, cached.confidence);
    return;
  }

  // ── 3. Call API with normalized URL ──────────────────────
  chrome.action.setBadgeText({ text: "...", tabId });
  chrome.action.setBadgeBackgroundColor({ color: "#2d6a9f", tabId });

  // Send normalized URL — strips ?v=xyz so YouTube shortcuts
  // in api.py match correctly and scraper isn't called for
  // URLs it can't scrape (JS-heavy, auth-gated pages)
  const normalizedUrl = normalizeUrl(url);

  try {
    const res = await fetch(`${API_BASE}/classify/url`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ url: normalizedUrl }),
      signal:  AbortSignal.timeout(45000),
    });

    if (res.status === 404 || res.status === 422) {
      // API couldn't classify — show unknown rather than error
      setBadge(tabId, "Unknown", 0);
      return;
    }

    if (!res.ok) {
      clearBadge(tabId);
      return;
    }

    const data       = await res.json();
    const category   = data.category   || "Unknown";
    const confidence = data.confidence || 0;

    setCache(url, category, confidence);
    setBadge(tabId, category, confidence);

  } catch (err) {
    // Timeout or network error — clear badge silently
    // Don't show ERROR for every Render cold start
    clearBadge(tabId);
    console.warn("WCC: classify failed", normalizedUrl, err.message);
  }
}

// ── Tab event listeners ───────────────────────────────────────
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === "complete" && tab.url) {
    classifyTab(tabId, tab.url);
  }
});

chrome.tabs.onActivated.addListener(async ({ tabId }) => {
  const tab = await chrome.tabs.get(tabId).catch(() => null);
  if (!tab?.url) return;
  const cached = await getCached(tab.url);
  if (cached) setBadge(tabId, cached.category, cached.confidence);
  else classifyTab(tabId, tab.url);
});

chrome.runtime.onInstalled.addListener(() => {
  console.log("Website Category Classifier v1.1.0 installed.");
  chrome.tabs.query({ active: true, currentWindow: true }, tabs => {
    if (tabs[0]) classifyTab(tabs[0].id, tabs[0].url);
  });
});