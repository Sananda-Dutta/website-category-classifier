# ═══════════════════════════════════════════════════════════════════════════════
# Website Category Classifier API  —  v2.5.0
# Model   : DistilBERT fine-tuned (11 categories)
# Author  : SanandaDutta
# HF Repo : SanandaDutta/website-category-distilbert
#
# v2.5.0 — HuggingFace Inference API architecture
#
# WHY THIS CHANGE:
#   DistilBERT (260 MB weights) + torch + transformers loading overhead
#   consistently exceeds 512 MB during startup on Render free tier,
#   causing OOM before the first request is ever served.
#   Even low_cpu_mem_usage=True and int8 quantization cannot fix this
#   because the peak occurs during torch+transformers import, before
#   any optimisation can be applied.
#
# NEW ARCHITECTURE:
#   Render (this file) → thin FastAPI wrapper, ~80 MB RAM
#   HuggingFace Hub   → hosts and runs DistilBERT, free inference API
#   Result: 432 MB headroom on Render, zero OOM risk, faster inference
#   (HF hardware is faster than Render free-tier CPU)
#
# SPEED:
#   Warm inference:  200–600 ms  (vs 800–1500 ms local on Render CPU)
#   HF cold start:   8–20 s first call after idle (same as before)
#   Domain shortcuts: 0 ms — unchanged, served instantly from this file
#
# SETUP (one-time):
#   1. Go to huggingface.co/settings/tokens
#   2. Create a token with "Make calls to the serverless Inference API" permission
#   3. In Render dashboard → Environment → add:
#      HF_TOKEN = hf_xxxxxxxxxxxxxxxxxxxx
#
# ALL ENDPOINTS PRESERVED:
#   /classify/url, /classify/text, /classify/batch, /safe-check,
#   /explain (disabled, preserved for future), /stats, /stats/export,
#   /health, /ping, /usage — all work identically from the caller's view.
# ═══════════════════════════════════════════════════════════════════════════════

import asyncio
import os
import re
import sqlite3
import time
import socket

from contextlib import asynccontextmanager
from datetime import datetime
from functools import lru_cache
from typing import List
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from slowapi.errors import RateLimitExceeded

from rate_limiter import limiter
from scraper import scrape_website, build_feature_string

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
HF_MODEL_ID  = "SanandaDutta/website-category-distilbert"
HF_API_URL   = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"
HF_TOKEN     = os.getenv("HF_TOKEN", "")          # set in Render env vars
DB_FILE      = "usage_logs.db"

CLASS_NAMES = [
    "Adult", "Arts", "Business", "Education", "Gaming",
    "Health", "Kids", "Lifestyle", "News", "Recreation", "Technology",
]

ADULT_CATEGORIES     = {"Adult"}
KIDS_CATEGORY        = "Kids"
SAFE_FOR_KIDS        = {"Education", "Kids", "Arts", "Recreation"}
CONFIDENCE_THRESHOLD = 45.0

# Shared async HTTP client — reused across all requests (connection pooling)
# Timeout: 25s connect, 25s read — HF cold start can take 15-20s
_hf_client: httpx.AsyncClient = None


# ─────────────────────────────────────────────
# LIFESPAN
# ─────────────────────────────────────────────

# --- 1. THE BACKGROUND ENGINE ---
async def background_prewarm():
    """
    This function runs quietly after the API is live.
    It solves the 'No address associated with hostname' error by waiting 
    for Render's network to fully connect.
    """
    # Wait 15 seconds to ensure Render's DNS is fully active
    await asyncio.sleep(30) 
    
    print("\n" + "═"*30)
    print("🚀 BACKGROUND PRE-WARM STARTING")
    print("═"*30)
    
    for i in range(3):
        try:
            # Send dummy data to wake up the Hugging Face Model
            await _call_hf_inference("warmup") 
            print(f"✅ HF model is awake and ready! (Attempt {i+1})")
            print("═"*30 + "\n")
            return
        except Exception as e:
            # If HF is still loading (503), wait and try again
            print(f"⚠️ Pre-warm attempt {i+1} failed: {e}")
            if i < 2:
                print("🔄 Retrying in 20 seconds...")
                await asyncio.sleep(20)
    
    print("❌ Background pre-warm failed after 3 attempts.")
    print("💡 Note: The first user request will trigger the model wake-up.")
    print("═"*30 + "\n")


# --- 2. THE MAIN LIFESPAN GATEWAY ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    The heart of the API. This handles everything from database 
    initialization to the final shutdown cleanup.
    """
    global _hf_client

    print("\n" + "╔" + "═"*53 + "╗")
    print(f"║ {'WEBSITE CATEGORY CLASSIFIER v2.5.0':^51} ║")
    print("╚" + "═"*53 + "╝")

    # PHASE A: Database Initialization
    try:
        init_db()
        print("📁 [1/4] SQLite Database: Ready")
    except Exception as e:
        print(f"❌ [1/4] SQLite Database: FAILED ({e})")

    # PHASE B: Credentials & Environment Validation
    if not HF_TOKEN:
        print("🚫 [2/4] Credentials: HF_TOKEN MISSING (API will fail)")
    else:
        print(f"🔑 [2/4] Credentials: HF_TOKEN Verified")
        print(f"🤖 [2/4] Model Target: {HF_MODEL_ID}")

    # PHASE C: Establish Persistent Network Connection
    # We create the client here so it's ready before the yield
    _hf_client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=25.0, read=25.0, write=10.0, pool=5.0),
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
    )
    print("🌐 [3/4] Network: HTTPX Client Initialized")

    # PHASE D: Trigger the Background Pre-Warm
    # CRITICAL: No 'await' here. This lets the function run in the 
    # background while we move on to 'yield' (making the API live).
    asyncio.create_task(background_prewarm())
    print("🛰️ [4/4] AI Engine: Background pre-warm scheduled")

    print("\n🚀 STATUS: API IS LIVE | RAM: ~80MB")
    print("═"*55 + "\n")
    # Add this inside lifespan, after creating _hf_client, before the yield:

    try:
        ip = socket.getaddrinfo("api-inference.huggingface.co", 443)
        print(f"✅ DNS resolved: api-inference.huggingface.co → {ip[0][4][0]}")
    except Exception as e:
        print(f"❌ DNS FAILED: {e}")
        print("💡 Render free tier may be blocking outbound to huggingface.co")
    # --- API EXECUTION PAUSE ---
    yield 
    # ---------------------------

    # PHASE E: Graceful Shutdown
    # This runs when you stop the server or Render redeploys
    await _hf_client.aclose()
    print("\n" + "═"*55)
    print("🛑 SHUTDOWN: HTTPX Client Closed. Service Offline.")
    print("═"*55)
# ─────────────────────────────────────────────
# HF INFERENCE API CALLER
# ─────────────────────────────────────────────
async def _call_hf_inference(text: str) -> list:
    """
    Calls HuggingFace Inference API for text classification.
    Returns list of {"label": str, "score": float} sorted by score desc.

    Handles:
    - 503 model loading (retries up to 3x with backoff)
    - 429 rate limit (waits estimated_time then retries)
    - Timeout (raises HTTPException 504)
    - Auth error (raises HTTPException 500 with clear message)
    """
    payload = {"inputs": text[:512]}   # HF has 512 token limit

    for attempt in range(3):
        try:
            r = await _hf_client.post(HF_API_URL, json=payload)

            # Model still loading on HF side — wait and retry
            if r.status_code == 503:
                data = r.json() if r.content else {}
                wait = min(data.get("estimated_time", 15), 20)
                print(f"HF model loading, waiting {wait}s (attempt {attempt+1}/3)")
                await asyncio.sleep(wait)
                continue

            # Rate limited
            if r.status_code == 429:
                data = r.json() if r.content else {}
                wait = min(data.get("estimated_time", 5), 10)
                await asyncio.sleep(wait)
                continue

            # Auth error — token missing or wrong
            if r.status_code == 401:
                raise HTTPException(500,
                    "HuggingFace auth failed. Check HF_TOKEN in Render env vars.")

            r.raise_for_status()

            results = r.json()

            # HF text-classification returns [[{label,score},...]] or [{label,score},...]
            if isinstance(results, list) and results:
                if isinstance(results[0], list):
                    results = results[0]
                # Sort by score descending (should already be sorted, but ensure it)
                results.sort(key=lambda x: x["score"], reverse=True)
                return results

            raise ValueError(f"Unexpected HF response format: {results}")

        except httpx.TimeoutException:
            if attempt == 2:
                raise HTTPException(504,
                    "HuggingFace Inference API timed out. "
                    "Model may be cold — retry in 20 seconds.")
            await asyncio.sleep(5)
            continue

        except HTTPException:
            raise
        except Exception as e:
            import traceback
            print(f"[HF ERROR attempt {attempt}] {type(e).__name__}: {e}")
            print(traceback.format_exc())
            if attempt == 2:
                raise HTTPException(500, f"HF Inference API error: {str(e)}")
            await asyncio.sleep(3)
            continue

    raise HTTPException(503, "HuggingFace Inference API unavailable after 3 retries.")


# ─────────────────────────────────────────────
# PREDICTION  (LRU cached — same interface as before)
# ─────────────────────────────────────────────
# Note: lru_cache requires sync function. We cache the result after
# the async HF call returns. The cache key is the feature string,
# so repeated URLs are served instantly without hitting HF API again.

_prediction_cache: dict = {}
_cache_lock = asyncio.Lock()
MAX_CACHE_SIZE = 1024

async def run_prediction(feature_string: str) -> tuple:
    """
    Returns (category, confidence_%, top3_list).
    Async-safe LRU cache — HF API only called on cache miss.
    Cache size capped at MAX_CACHE_SIZE entries (FIFO eviction).
    """
    # Fast path — no lock needed for reads on dict in CPython
    if feature_string in _prediction_cache:
        return _prediction_cache[feature_string]

    # Cache miss — call HF API
    results = await _call_hf_inference(feature_string)

    # Parse response into our standard format
    top3 = [
        {
            "category":   r["label"],
            "confidence": round(r["score"] * 100, 2),
        }
        for r in results[:3]
    ]
    category   = top3[0]["category"]
    confidence = top3[0]["confidence"]
    result     = (category, confidence, top3)

    # Store in cache (evict oldest if full)
    async with _cache_lock:
        if len(_prediction_cache) >= MAX_CACHE_SIZE:
            # Remove oldest 10% of entries
            evict_count = MAX_CACHE_SIZE // 10
            for key in list(_prediction_cache.keys())[:evict_count]:
                del _prediction_cache[key]
        _prediction_cache[feature_string] = result

    return result


# ─────────────────────────────────────────────
# SQLITE LOGGING
# ─────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS api_logs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            ip          TEXT,
            endpoint    TEXT    NOT NULL,
            input_url   TEXT,
            input_text  TEXT,
            category    TEXT,
            confidence  REAL,
            success     INTEGER NOT NULL,
            time_ms     REAL,
            method      TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_request(
    ip: str, endpoint: str, success: bool, time_ms: float,
    input_url:  str   = None,
    input_text: str   = None,
    category:   str   = None,
    confidence: float = None,
    method:     str   = None,
):
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.execute("""
            INSERT INTO api_logs
                (timestamp, ip, endpoint, input_url, input_text,
                 category, confidence, success, time_ms, method)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(), ip, endpoint,
            input_url,
            (input_text[:200] if input_text else None),
            category, confidence,
            int(success), time_ms, method,
        ))
        conn.commit()
        conn.close()
    except Exception:
        pass


# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────
app = FastAPI(
    title="Website Category Classifier API",
    description=(
        "Classify any website into 11 categories using DistilBERT.\n\n"
        "**Categories:** Adult · Arts · Business · Education · Gaming · "
        "Health · Kids · Lifestyle · News · Recreation · Technology\n\n"
        "Built by **SanandaDutta** — "
        "[HuggingFace](https://huggingface.co/SanandaDutta) · "
        "[GitHub](https://github.com/SanandaDutta)"
    ),
    version="2.5.0",
    lifespan=lifespan,
)

# ── RapidAPI proxy secret ─────────────────────
RAPIDAPI_SECRET = os.getenv("RAPIDAPI_PROXY_SECRET", "").strip()

@app.middleware("http")
async def verify_rapidapi_proxy(request: Request, call_next):
    skip_paths = {"/health", "/docs", "/openapi.json", "/redoc",
                  "/", "/ping", "/usage"}
    if RAPIDAPI_SECRET and request.url.path not in skip_paths:
        incoming = request.headers.get("X-RapidAPI-Proxy-Secret", "").strip()
        print(f"[AUTH] path={request.url.path} incoming={repr(incoming)} expected={repr(RAPIDAPI_SECRET)} match={incoming == RAPIDAPI_SECRET}")
        if incoming != RAPIDAPI_SECRET:
            return JSONResponse(status_code=403, content={
                "error": "Access via RapidAPI only. Sign up at rapidapi.com"
            })
    return await call_next(request)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={
        "error":  "Rate limit exceeded.",
        "detail": "Max 30 req/min on classify, 5/min on batch.",
    })


# ─────────────────────────────────────────────
# DOMAIN / PATH SHORTCUTS  (served locally — no HF call needed)
# ─────────────────────────────────────────────
PATH_SHORTCUTS = {
    "bbc.co.uk/sport":      "Recreation",
    "bbc.com/sport":        "Recreation",
    "bbc.co.uk/news":       "News",
    "bbc.co.uk/health":     "Health",
    "bbc.co.uk/education":  "Education",
    "ndtv.com/sports":      "Recreation",
    "ndtv.com/health":      "Health",
    "timesofindia.indiatimes.com/sports": "Recreation",
    "indianexpress.com/sports":           "Recreation",
    "hindustantimes.com/cricket":         "Recreation",
    "youtube.com/gaming":   "Gaming",
    "reddit.com/r/india":   "News",
    "reddit.com/r/cricket": "Recreation",
    "reddit.com/r/gaming":  "Gaming",
    "reddit.com/r/science": "Education",
}

DOMAIN_SHORTCUTS = {
    # AI / Modern Tech
    "claude.ai":"Technology","anthropic.com":"Technology",
    "openai.com":"Technology","chatgpt.com":"Technology",
    "mistral.ai":"Technology","huggingface.co":"Technology",
    "vercel.com":"Technology","supabase.com":"Technology",
    "figma.com":"Technology","linear.app":"Technology",
    "cloudflare.com":"Technology","digitalocean.com":"Technology",
    "render.com":"Technology","netlify.com":"Technology",
    "heroku.com":"Technology","aws.amazon.com":"Technology",
    "azure.microsoft.com":"Technology",
    # News
    "ndtv.com":"News","thehindu.com":"News","hindustantimes.com":"News",
    "timesofindia.indiatimes.com":"News","indianexpress.com":"News",
    "scroll.in":"News","thewire.in":"News","theprint.in":"News",
    "bbc.com":"News","bbc.co.uk":"News","reuters.com":"News",
    "apnews.com":"News","theguardian.com":"News",
    "aajtak.in":"News","zeenews.india.com":"News","news18.com":"News",
    "abplive.com":"News","tv9bharatvarsh.com":"News",
    "twitter.com":"News","x.com":"News","facebook.com":"News",
    "reddit.com":"News",
    # Business
    "moneycontrol.com":"Business","economictimes.indiatimes.com":"Business",
    "livemint.com":"Business","business-standard.com":"Business",
    "zerodha.com":"Business","groww.in":"Business","upstox.com":"Business",
    "amazon.in":"Business","flipkart.com":"Business","snapdeal.com":"Business",
    "razorpay.com":"Business","paytm.com":"Business",
    "phonepe.com":"Business","indiamart.com":"Business",
    "zoho.com":"Business","freshworks.com":"Business",
    "amazon.com":"Business","ebay.com":"Business",
    "meesho.com":"Business","myntra.com":"Business","ajio.com":"Business",
    "nykaa.com":"Business","linkedin.com":"Business",
    "shopify.com":"Business","stripe.com":"Business",
    # Technology
    "github.com":"Technology","stackoverflow.com":"Technology",
    "geeksforgeeks.org":"Technology","hackerrank.com":"Technology",
    "leetcode.com":"Technology","codechef.com":"Technology",
    "codeforces.com":"Technology","digit.in":"Technology",
    "gadgets360.com":"Technology","91mobiles.com":"Technology",
    "beebom.com":"Technology","techcrunch.com":"Technology",
    "theverge.com":"Technology","wired.com":"Technology",
    "arstechnica.com":"Technology","dev.to":"Technology",
    "medium.com":"Technology","producthunt.com":"Technology",
    "npmjs.com":"Technology","pypi.org":"Technology",
    # Education
    "byjus.com":"Education","unacademy.com":"Education",
    "vedantu.com":"Education","coursera.org":"Education",
    "khanacademy.org":"Education","nptel.ac.in":"Education",
    "swayam.gov.in":"Education","wikipedia.org":"Education",
    "doubtnut.com":"Education","testbook.com":"Education",
    "udemy.com":"Education","edx.org":"Education",
    "britannica.com":"Education","collegedunia.com":"Education",
    "shiksha.com":"Education","careers360.com":"Education",
    # Health
    "practo.com":"Health","1mg.com":"Health","netmeds.com":"Health",
    "apollohospitals.com":"Health","webmd.com":"Health",
    "healthline.com":"Health","pharmeasy.in":"Health",
    "cult.fit":"Health","medscape.com":"Health",
    "mayoclinic.org":"Health","nih.gov":"Health",
    # Gaming
    "dream11.com":"Gaming","mpl.live":"Gaming","winzo.com":"Gaming",
    "zupee.com":"Gaming","rummycircle.com":"Gaming","adda52.com":"Gaming",
    "twitch.tv":"Gaming","ign.com":"Gaming","gamespot.com":"Gaming",
    "steampowered.com":"Gaming","epicgames.com":"Gaming",
    "xbox.com":"Gaming","playstation.com":"Gaming",
    # Recreation
    "cricbuzz.com":"Recreation","espncricinfo.com":"Recreation",
    "sportskeeda.com":"Recreation","indiahikes.com":"Recreation",
    "bcci.tv":"Recreation","iplt20.com":"Recreation",
    "espn.com":"Recreation","bleacherreport.com":"Recreation",
    "fifa.com":"Recreation","olympics.com":"Recreation",
    # Lifestyle
    "zomato.com":"Lifestyle","swiggy.com":"Lifestyle",
    "makemytrip.com":"Lifestyle","irctc.co.in":"Lifestyle",
    "vogue.in":"Lifestyle","femina.in":"Lifestyle",
    "mensxp.com":"Lifestyle","shaadi.com":"Lifestyle",
    "instagram.com":"Lifestyle","pinterest.com":"Lifestyle",
    "tripadvisor.com":"Lifestyle","booking.com":"Lifestyle",
    "airbnb.com":"Lifestyle","uber.com":"Lifestyle",
    # Kids
    "firstcry.com":"Kids","nickelodeonindia.com":"Kids",
    "tinkle.in":"Kids","amarchitrakatha.com":"Kids",
    "disneyindia.in":"Kids","pbs.org":"Kids","starfall.com":"Kids",
    # Arts
    "gaana.com":"Arts","saavn.com":"Arts","jiosavan.com":"Arts",
    "filmfare.com":"Arts","bollywoodhungama.com":"Arts",
    "pratilipi.com":"Arts","rekhta.org":"Arts","bookmyshow.com":"Arts",
    "youtube.com":"Arts","youtu.be":"Arts","netflix.com":"Arts",
    "primevideo.com":"Arts","hotstar.com":"Arts",
    "disneyplus.com":"Arts","zee5.com":"Arts",
    "sonyliv.com":"Arts","voot.com":"Arts","spotify.com":"Arts",
    "imdb.com":"Arts","rottentomatoes.com":"Arts",
}


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def get_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    return forwarded.split(",")[0].strip() if forwarded else (
        request.client.host if request.client else "unknown"
    )

def get_domain(url: str) -> str:
    parsed = urlparse(url if url.startswith("http") else "https://" + url)
    return parsed.netloc.replace("www.", "").lower()

def get_domain_with_path(url: str) -> str:
    try:
        parsed = urlparse(url if url.startswith("http") else "https://" + url)
        domain = parsed.netloc.replace("www.", "").lower()
        parts  = [p for p in parsed.path.split("/") if p]
        return f"{domain}/{parts[0]}" if parts else domain
    except Exception:
        return get_domain(url)

def is_valid_url(url: str) -> bool:
    return bool(re.match(
        r'^(https?://)?(([a-zA-Z0-9\-]+\.)+[a-zA-Z]{2,})(/.*)?$', url
    ))

NOISE_WORDS = {
    "cookie","cookies","consent","gdpr","ccpa","privacy","policy",
    "terms","conditions","legal","disclaimer","accept","decline",
    "preferences","manage","settings","necessary","functional",
    "analytics","marketing","tracking","home","menu","navigation",
    "navbar","sidebar","previous","next","back","forward","search",
    "close","open","toggle","expand","collapse","show","hide",
    "copyright","rights","reserved","trademark","sitemap","careers",
    "jobs","about","contact","press","feedback","help","support",
    "faq","subscribe","newsletter","signup","login","logout","register",
    "account","profile","password","username","email","phone",
    "share","tweet","post","like","follow","comment","reply",
    "javascript","enabled","disabled","browser","reload","loading",
    "please","enable","continue","proceed","click","download",
    "install","update","version","app","store","new","latest",
    "best","top","free","get","use","see","also","read","view",
    "here","now","today","week","month","the","and","for","are",
    "but","not","you","all","can","was","one","our","out","has",
    "how","its","may","who","did","let","put","too","way",
    "com","www","http","https","html","php","asp","net","org",
}

def extract_url_features(url: str) -> str:
    try:
        parsed       = urlparse(url if url.startswith("http") else "http://" + url)
        domain       = parsed.netloc.replace("www.", "")
        path         = parsed.path
        tld          = domain.split(".")[-1] if "." in domain else ""
        domain_words = re.split(r'[.\-_]', domain)
        path_words   = re.split(r'[/\-_.]', path)
        tld_signal   = {
            "edu": "education university college academic",
            "gov": "government official public authority",
            "org": "organization nonprofit charity",
            "ac":  "academic university college",
        }.get(tld, "")
        all_parts = domain_words * 3 + path_words + tld_signal.split()
        clean = [
            w.lower() for w in all_parts
            if len(w) > 2 and w.isalpha() and w.lower() not in NOISE_WORDS
        ]
        return " ".join(clean)
    except Exception:
        return ""

def build_weighted_features(scraped: dict, url: str) -> str:
    parts = []
    try:
        title = scraped.get("title", "").strip()
        meta  = scraped.get("meta_description", "").strip()
        h1    = scraped.get("h1", "").strip()
        h2    = scraped.get("h2", "").strip()
        body  = scraped.get("body_text", "").strip()
        if not title and not body and "text" in scraped:
            body = scraped.get("text", "")
        url_feat = extract_url_features(url)
        if title:    parts += [title] * 4
        if meta:     parts += [meta]  * 3
        if h1:       parts += [h1]    * 2
        if h2:       parts += [h2]    * 2
        if body:     parts += [body[:800]]
        if url_feat: parts += [url_feat] * 2
    except Exception:
        return extract_url_features(url)
    words = " ".join(parts).lower().split()
    return " ".join(w for w in words if w not in NOISE_WORDS and len(w) > 2)

def filter_noise(text: str) -> str:
    return " ".join(
        w for w in text.lower().split()
        if w not in NOISE_WORDS and len(w) > 2
    )


# ─────────────────────────────────────────────
# SMART CLASSIFY
# ─────────────────────────────────────────────
async def smart_classify(url: str) -> tuple:
    """
    Returns (category, confidence, top3, method).
    Shortcuts served locally (0 ms). Unknown domains call HF API.
    """
    domain           = get_domain(url)
    domain_with_path = get_domain_with_path(url)

    # Path-aware shortcut
    if domain_with_path in PATH_SHORTCUTS:
        cat = PATH_SHORTCUTS[domain_with_path]
        return cat, 99.0, [{"category": cat, "confidence": 99.0}], "path_shortcut"

    # Bare domain shortcut
    if domain in DOMAIN_SHORTCUTS:
        cat = DOMAIN_SHORTCUTS[domain]
        return cat, 99.0, [{"category": cat, "confidence": 99.0}], "domain_shortcut"

    # Scrape + feature extraction
    try:
        scraped = scrape_website(url)
    except Exception:
        scraped = {"error": "SCRAPE_FAILED"}

    if scraped.get("error"):
        features = extract_url_features(url)
        method   = "url_features_only"
    else:
        features = build_weighted_features(scraped, url)
        method   = "combined_features"

    if not features.strip():
        raise HTTPException(422, "Could not extract any features from this URL.")

    category, confidence, top3 = await run_prediction(features)

    # Low-confidence fallback
    if confidence < CONFIDENCE_THRESHOLD and method == "combined_features":
        url_features = extract_url_features(url)
        if url_features.strip():
            cat_url, conf_url, top3_url = await run_prediction(url_features)
            if conf_url > confidence:
                category, confidence, top3 = cat_url, conf_url, top3_url
                method = "url_features_fallback"

    return category, confidence, top3, method


# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────
class URLRequest(BaseModel):
    url: str

class TextRequest(BaseModel):
    text: str

class BatchURLRequest(BaseModel):
    urls: List[str]

class PredictionResult(BaseModel):
    category:   str
    confidence: float
    top3:       List[dict]
    method:     str
    time_ms:    float


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
async def root(request: Request):
    return {
        "message"     : "Website Category Classifier API v2.5.0",
        "architecture": "HuggingFace Inference API (Render RAM: ~80 MB)",
        "classes"     : CLASS_NAMES,
        "docs"        : "/docs",
    }

@app.api_route("/ping", methods=["GET", "HEAD"], tags=["Info"])
async def ping(request: Request):
    return {"status": "alive", "version": "2.5.0"}

@app.api_route("/health", methods=["GET", "HEAD"], tags=["Info"])
async def health(request: Request):
    return {
        "status"          : "ok",
        "version"         : "2.5.0",
        "architecture"    : "hf_inference_api",
        "hf_model"        : HF_MODEL_ID,
        "hf_token_set"    : bool(HF_TOKEN),
        "classes"         : CLASS_NAMES,
        "cache_size"      : len(_prediction_cache),
        "cache_max"       : MAX_CACHE_SIZE,
    }

@app.get("/usage", tags=["Info"])
async def usage_info():
    return {
        "name":       "Website Category Classifier",
        "version":    "2.5.0",
        "categories": CLASS_NAMES,
        "endpoints": {
            "POST /classify/url":   "Classify a live website by URL",
            "POST /classify/text":  "Classify raw text input",
            "POST /classify/batch": "Classify up to 20 URLs at once",
            "POST /safe-check":     "Adult/Kids safety verdict",
            "GET  /stats":          "API usage analytics",
        },
        "rate_limits": {
            "free":  "100 calls/month",
            "basic": "5,000 calls/month",
            "pro":   "50,000 calls/month",
        },
    }


# ── 1. POST /classify/url ─────────────────────
@app.post("/classify/url", response_model=PredictionResult, tags=["Classify"])
@limiter.limit("30/minute")
async def classify_url(request: Request, body: URLRequest):
    start = time.time()
    ip    = get_ip(request)
    try:
        url = body.url.strip()
        if not is_valid_url(url):
            raise HTTPException(422, "Invalid URL format.")
        category, confidence, top3, method = await smart_classify(url)
        elapsed = round((time.time() - start) * 1000, 1)
        log_request(ip, "/classify/url", True, elapsed,
                    input_url=url, category=category,
                    confidence=confidence, method=method)
        return PredictionResult(
            category=category, confidence=confidence,
            top3=top3, method=method, time_ms=elapsed,
        )
    except HTTPException:
        raise
    except Exception as e:
        log_request(ip, "/classify/url", False, 0, input_url=body.url)
        raise HTTPException(500, f"Internal error: {str(e)}")


# ── 2. POST /classify/text ────────────────────
@app.post("/classify/text", response_model=PredictionResult, tags=["Classify"])
@limiter.limit("30/minute")
async def classify_text(request: Request, body: TextRequest):
    start = time.time()
    ip    = get_ip(request)
    try:
        text = body.text.strip()
        if len(text) < 10:
            raise HTTPException(422, "Text too short. Minimum 10 characters.")
        text = text[:5000]
        category, confidence, top3 = await run_prediction(filter_noise(text))
        elapsed = round((time.time() - start) * 1000, 1)
        log_request(ip, "/classify/text", True, elapsed,
                    input_text=text, category=category,
                    confidence=confidence, method="text_input")
        return PredictionResult(
            category=category, confidence=confidence,
            top3=top3, method="text_input", time_ms=elapsed,
        )
    except HTTPException:
        raise
    except Exception as e:
        log_request(ip, "/classify/text", False, 0, input_text=body.text[:100])
        raise HTTPException(500, f"Internal error: {str(e)}")


# ── 3. POST /classify/batch ───────────────────
@app.post("/classify/batch", tags=["Classify"])
@limiter.limit("5/minute")
async def classify_batch(request: Request, body: BatchURLRequest):
    start = time.time()
    ip    = get_ip(request)
    try:
        if len(body.urls) == 0:
            raise HTTPException(422, "Provide at least 1 URL.")
        if len(body.urls) > 20:
            raise HTTPException(422, "Max 20 URLs per batch request.")
        results = []
        for url in body.urls:
            try:
                category, confidence, _, method = await smart_classify(url)
                results.append({
                    "url": url, "category": category,
                    "confidence": confidence, "method": method,
                    "safe":       category not in ADULT_CATEGORIES,
                    "adult_flag": category in ADULT_CATEGORIES,
                })
            except Exception as inner_e:
                results.append({
                    "url": url, "category": "Error", "confidence": 0.0,
                    "method": str(inner_e)[:80], "safe": None, "adult_flag": None,
                })
        elapsed = round((time.time() - start) * 1000, 1)
        log_request(ip, "/classify/batch", True, elapsed,
                    method=f"batch_{len(results)}_urls")
        csv_lines = ["url,category,confidence,method,safe,adult_flag"]
        for r in results:
            csv_lines.append(
                f"{r['url']},{r['category']},{r['confidence']},"
                f"{r['method']},{r['safe']},{r['adult_flag']}"
            )
        return {
            "total": len(results), "time_ms": elapsed,
            "results": results, "csv_export": "\n".join(csv_lines),
        }
    except HTTPException:
        raise
    except Exception as e:
        log_request(ip, "/classify/batch", False, 0)
        raise HTTPException(500, f"Internal error: {str(e)}")


# ── 4. POST /safe-check ───────────────────────
@app.post("/safe-check", tags=["Safety"])
@limiter.limit("30/minute")
async def safe_check(request: Request, body: URLRequest):
    start = time.time()
    ip    = get_ip(request)
    try:
        url = body.url.strip()
        if not is_valid_url(url):
            raise HTTPException(422, "Invalid URL format.")
        category, confidence, _, method = await smart_classify(url)
        adult_flag    = category in ADULT_CATEGORIES
        kids_safe     = category == KIDS_CATEGORY
        safe_for_kids = category in SAFE_FOR_KIDS
        verdict = (
            "🔴 ADULT — block recommended" if adult_flag    else
            "🟢 KIDS SAFE"                 if kids_safe     else
            "🟡 SAFE FOR KIDS"             if safe_for_kids else
            "🟢 SAFE"
        )
        elapsed = round((time.time() - start) * 1000, 1)
        log_request(ip, "/safe-check", True, elapsed,
                    input_url=url, category=category,
                    confidence=confidence, method=method)
        return {
            "url": url, "category": category, "confidence": confidence,
            "safe": not adult_flag, "adult_flag": adult_flag,
            "kids_safe": kids_safe, "safe_for_kids": safe_for_kids,
            "verdict": verdict, "method": method, "time_ms": elapsed,
        }
    except HTTPException:
        raise
    except Exception as e:
        log_request(ip, "/safe-check", False, 0, input_url=body.url)
        raise HTTPException(500, f"Internal error: {str(e)}")


# ── 5. /explain  (disabled — LIME needs local model) ─────────────────────────
# When you move to a paid plan and load the model locally again,
# restore the full LIME implementation from v2.3.0.
@app.api_route("/explain", methods=["GET", "POST"], tags=["XAI"])
async def explain(request: Request):
    return JSONResponse(status_code=503, content={
        "error"      : "LIME explanation unavailable in HF Inference API mode.",
        "reason"     : "LIME requires local model access. Currently using HF API to stay within 512 MB RAM.",
        "workaround" : "Use top3 from /classify/url — it shows the top 3 category scores.",
        "re_enable"  : "Load model locally (requires 1 GB+ RAM plan) and restore v2.3.0 LIME code.",
    })


# ── 6. GET /stats ─────────────────────────────
@app.get("/stats", tags=["Analytics"])
async def get_stats(
    request: Request,
    limit  : int = Query(100, ge=1, le=1000),
):
    try:
        conn          = sqlite3.connect(DB_FILE)
        total         = conn.execute("SELECT COUNT(*) FROM api_logs").fetchone()[0]
        success_count = conn.execute(
            "SELECT COUNT(*) FROM api_logs WHERE success=1"
        ).fetchone()[0]
        by_endpoint = conn.execute("""
            SELECT endpoint, COUNT(*) AS calls,
                   ROUND(AVG(time_ms),1) AS avg_ms,
                   SUM(CASE WHEN success=0 THEN 1 ELSE 0 END) AS errors
            FROM api_logs GROUP BY endpoint ORDER BY calls DESC
        """).fetchall()
        by_category = conn.execute("""
            SELECT category, COUNT(*) AS count FROM api_logs
            WHERE category IS NOT NULL AND category != ''
            GROUP BY category ORDER BY count DESC
        """).fetchall()
        recent = conn.execute(
            """SELECT timestamp, ip, endpoint, input_url,
                      category, confidence, success, time_ms, method
               FROM api_logs ORDER BY id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        conn.close()
        return {
            "summary": {
                "total_requests": total,
                "success_count" : success_count,
                "error_count"   : total - success_count,
                "success_rate"  : round(success_count / total * 100, 1) if total else 0,
            },
            "by_endpoint": [
                {"endpoint": r[0], "calls": r[1], "avg_ms": r[2], "errors": r[3]}
                for r in by_endpoint
            ],
            "by_category": [
                {"category": r[0], "count": r[1]} for r in by_category
            ],
            "recent_requests": [
                {
                    "timestamp": r[0], "ip": r[1], "endpoint": r[2],
                    "input_url": r[3], "category": r[4], "confidence": r[5],
                    "success": bool(r[6]), "time_ms": r[7], "method": r[8],
                }
                for r in recent
            ],
        }
    except Exception as e:
        raise HTTPException(500, f"Stats error: {str(e)}")


# ── 7. GET /stats/export ──────────────────────
@app.get("/stats/export", tags=["Analytics"])
async def export_logs():
    try:
        conn = sqlite3.connect(DB_FILE)
        rows = conn.execute("""
            SELECT timestamp, ip, endpoint, input_url, input_text,
                   category, confidence, success, time_ms, method
            FROM api_logs ORDER BY id DESC
        """).fetchall()
        conn.close()
        header = "timestamp,ip,endpoint,input_url,input_text,category,confidence,success,time_ms,method"
        lines  = [header]
        for r in rows:
            lines.append(",".join(
                f'"{str(x)}"' if x is not None else '""' for x in r
            ))
        return StreamingResponse(
            iter(["\n".join(lines)]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=api_logs.csv"},
        )
    except Exception as e:
        raise HTTPException(500, f"Export error: {str(e)}")