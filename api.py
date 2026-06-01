# ═══════════════════════════════════════════════════════════════════════════════
# Website Category Classifier API  —  v2.4.0  (COMPLETE)
# Model   : DistilBERT fine-tuned (11 categories)
# Author  : SanandaDutta
# HF Repo : SanandaDutta/website-category-distilbert
# Render  : website-category-classifier.onrender.com
#
# Memory profile on Render free tier (512 MB):
#   int8 quantized DistilBERT : ~ 65 MB
#   torch base                : ~ 80 MB
#   FastAPI + dependencies    : ~ 50 MB
#   scraping peak             : ~ 30 MB
#   ─────────────────────────────────
#   Total peak                : ~225 MB  (287 MB headroom)
#
# All endpoints:
#   POST /classify/url     scrape + predict, path/domain shortcuts
#   POST /classify/text    raw text input
#   POST /classify/batch   up to 20 URLs, CSV export
#   POST /safe-check       brand safety + parental control verdict
#   GET  /stats            usage analytics
#   GET  /stats/export     download logs as CSV
#   GET  /health           model status + uptime + cache info
#   GET  /usage            API capabilities summary (RapidAPI)
#   GET  /ping             keepalive
#
# LIME /explain: removed in v2.4.0 to fit 512 MB free tier.
# To restore: upgrade Render to Starter ($7/mo, 1 GB RAM),
# add lime==0.2.0.1 to requirements.txt, uncomment stub at bottom.
#
# v2.4.0 vs v2.3.0:
#   - LIME + predict_proba_batch removed  → ~120 MB saved at peak
#   - asyncio lazy lock removed           → simpler, eager load
#   - int8 quantization at startup        → 260 MB → 65 MB model RAM
#   - low_cpu_mem_usage=True              → no double-copy OOM during load
#   - All accuracy + speed improvements from v2.3.0 preserved:
#       smart_classify(), build_weighted_features(), confidence fallback,
#       path-aware shortcuts, 150+ noise words, torch.inference_mode(),
#       max_length=192, LRU cache 256, 3-pass warmup
#   - RapidAPI proxy secret middleware added
#   - HEAD method on /, /ping, /health  (RapidAPI health probe requires it)
# ═══════════════════════════════════════════════════════════════════════════════

import csv
import os
import re
import sqlite3
import time

from contextlib import asynccontextmanager
from datetime import datetime
from functools import lru_cache
from typing import List
from urllib.parse import urlparse

import torch
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from slowapi.errors import RateLimitExceeded
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from rate_limiter import limiter
from scraper import scrape_website, build_feature_string

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
HF_MODEL_ID = "SanandaDutta/website-category-distilbert"
MODEL_DIR   = "./distilbert_final"
DB_FILE     = "usage_logs.db"

CLASS_NAMES = [
    "Adult", "Arts", "Business", "Education", "Gaming",
    "Health", "Kids", "Lifestyle", "News", "Recreation", "Technology",
]

ADULT_CATEGORIES     = {"Adult"}
KIDS_CATEGORY        = "Kids"
SAFE_FOR_KIDS        = {"Education", "Kids", "Arts", "Recreation"}
CONFIDENCE_THRESHOLD = 45.0

# Globals set in lifespan
tokenizer    = None
model        = None
startup_time = None
device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# SQLITE LOGGING
# ─────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_FILE,timeout=10)
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
    input_url: str = None, input_text: str = None,
    category: str = None, confidence: float = None, method: str = None,
):
    """Non-crashing logger — API never fails due to a log error."""
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.execute("""
            INSERT INTO api_logs
                (timestamp, ip, endpoint, input_url, input_text,
                 category, confidence, success, time_ms, method)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(), ip, endpoint,
            input_url, (input_text[:200] if input_text else None),
            category, confidence, int(success), time_ms, method,
        ))
        conn.commit()
        conn.close()
    except Exception:
        pass


# ─────────────────────────────────────────────
# LIFESPAN — eager model load
#
# WHY EAGER (not lazy like v2.3.0):
#   RapidAPI probes HEAD / before routing traffic. Lazy loading means the
#   first classify request triggers a 30-60s HF download — RapidAPI times
#   out, user sees 503. Eager load: port opens only after model is ready,
#   so every request including the first is fast.
#
# WHY low_cpu_mem_usage=True:
#   Without it, from_pretrained() holds a full second copy of weights in
#   RAM during loading, causing a ~260 MB spike → OOM before server starts.
#   This flag streams weights directly into place — no double-copy.
#
# WHY quantize_dynamic:
#   float32 DistilBERT = ~260 MB resident RAM
#   int8  DistilBERT   = ~ 65 MB resident RAM  (saves ~195 MB permanently)
#   Accuracy loss on text classification: typically < 0.5%
#   quantize_dynamic is safe here — it runs AFTER model is fully loaded,
#   replacing the float32 Linear layers one at a time, peak spike is ~30 MB.
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model, startup_time

    print("=" * 55)
    print("  Website Category Classifier API  v2.4.0")
    print("=" * 55)

    init_db()
    print("SQLite ready")

    # ── Load tokenizer + model ───────────────────────────────────────────────
    try:
        if os.path.isdir(MODEL_DIR):
            print(f"Loading from local: {MODEL_DIR}")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_DIR, low_cpu_mem_usage=True
            )
            print("Loaded from local folder")
        else:
            print(f"Loading from HuggingFace: {HF_MODEL_ID}")
            tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
            model = AutoModelForSequenceClassification.from_pretrained(
                HF_MODEL_ID, low_cpu_mem_usage=True
            )
            print("Loaded from HuggingFace Hub")
    except Exception as e:
        raise RuntimeError(
            f"Model load failed: {e}\n"
            f"HF repo: {HF_MODEL_ID}\n"
            f"Local dir: {MODEL_DIR} (exists={os.path.isdir(MODEL_DIR)})"
        ) from e

    model.to(device)
    model.eval()

    # ── int8 quantization (CPU only) ─────────────────────────────────────────
    if device.type == "cpu":
        model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8,
        )
        print("int8 quantization applied — ~195 MB saved")

    # ── 3-pass warmup ────────────────────────────────────────────────────────
    # Covers short / medium / long token lengths so JIT traces all paths.
    # First real request will then be instant.
    warmup_texts = [
        "cricket sports news scores live match updates india",
        "buy smartphones laptops online free delivery best price",
        "education university courses academic learning students online",
    ]
    for wt in warmup_texts:
        enc = tokenizer(
            wt, return_tensors="pt",
            truncation=True, max_length=192, padding=True,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        enc.pop("token_type_ids", None)
        with torch.inference_mode():
            model(**enc)
    print("Warmup complete (3 passes)")

    startup_time = time.time()
    print("=" * 55)
    print("API ready — port opening now")
    print("=" * 55)

    yield

    print("Shutting down")


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
    version="2.4.0",
    lifespan=lifespan,
)

# ── RapidAPI proxy secret middleware ─────────────────────────────────────────
# Set RAPIDAPI_PROXY_SECRET in Render → Environment.
# Leave blank during development — all requests pass through.
# Fill in after your RapidAPI listing is approved.
RAPIDAPI_SECRET = os.getenv("RAPIDAPI_PROXY_SECRET", "")

@app.middleware("http")
async def verify_rapidapi_proxy(request: Request, call_next):
    skip_paths = {
        "/", "/health", "/ping", "/docs",
        "/openapi.json", "/redoc", "/usage",
    }
    if RAPIDAPI_SECRET and request.url.path not in skip_paths:
        incoming = request.headers.get("X-RapidAPI-Proxy-Secret", "")
        if incoming != RAPIDAPI_SECRET:
            return JSONResponse(
                status_code=403,
                content={
                    "error":  "Access via RapidAPI only.",
                    "detail": "Subscribe at rapidapi.com/SanandaDutta/api/website-category-classifier",
                },
            )
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
        "detail": "Max 30 req/min on classify endpoints, 5/min on batch.",
    })


# ─────────────────────────────────────────────
# DOMAIN / PATH SHORTCUTS
# Path shortcuts checked first — more specific wins.
# ─────────────────────────────────────────────
PATH_SHORTCUTS = {
    # BBC subdomains
    "bbc.co.uk/sport":      "Recreation",
    "bbc.com/sport":        "Recreation",
    "bbc.co.uk/news":       "News",
    "bbc.co.uk/health":     "Health",
    "bbc.co.uk/education":  "Education",
    # Indian news sport paths
    "ndtv.com/sports":                        "Recreation",
    "ndtv.com/health":                        "Health",
    "timesofindia.indiatimes.com/sports":     "Recreation",
    "indianexpress.com/sports":               "Recreation",
    "hindustantimes.com/cricket":             "Recreation",
    # YouTube
    "youtube.com/gaming":   "Gaming",
    # Reddit subreddits
    "reddit.com/r/india":   "News",
    "reddit.com/r/cricket": "Recreation",
    "reddit.com/r/gaming":  "Gaming",
    "reddit.com/r/science": "Education",
    "reddit.com/r/health":  "Health",
    "reddit.com/r/news":    "News",
    # Amazon sections
    "amazon.in/health":     "Health",
    "amazon.com/health":    "Health",
}

DOMAIN_SHORTCUTS = {
    # ── AI / Modern Tech ──────────────────────────────────────────────────────
    "claude.ai":             "Technology",
    "anthropic.com":         "Technology",
    "openai.com":            "Technology",
    "chatgpt.com":           "Technology",
    "mistral.ai":            "Technology",
    "huggingface.co":        "Technology",
    "vercel.com":            "Technology",
    "supabase.com":          "Technology",
    "figma.com":             "Technology",
    "linear.app":            "Technology",
    "cloudflare.com":        "Technology",
    "digitalocean.com":      "Technology",
    "render.com":            "Technology",
    "netlify.com":           "Technology",
    "heroku.com":            "Technology",
    "aws.amazon.com":        "Technology",
    "azure.microsoft.com":   "Technology",
    "notion.so":             "Technology",
    "airtable.com":          "Technology",
    "zapier.com":            "Technology",
    # ── News ──────────────────────────────────────────────────────────────────
    "ndtv.com":              "News",
    "thehindu.com":          "News",
    "hindustantimes.com":    "News",
    "timesofindia.indiatimes.com": "News",
    "indianexpress.com":     "News",
    "scroll.in":             "News",
    "thewire.in":            "News",
    "theprint.in":           "News",
    "bbc.com":               "News",
    "bbc.co.uk":             "News",
    "reuters.com":           "News",
    "apnews.com":            "News",
    "theguardian.com":       "News",
    "aajtak.in":             "News",
    "zeenews.india.com":     "News",
    "news18.com":            "News",
    "abplive.com":           "News",
    "tv9bharatvarsh.com":    "News",
    "twitter.com":           "News",
    "x.com":                 "News",
    "facebook.com":          "News",
    "reddit.com":            "News",
    # ── Business ──────────────────────────────────────────────────────────────
    "moneycontrol.com":      "Business",
    "economictimes.indiatimes.com": "Business",
    "livemint.com":          "Business",
    "business-standard.com": "Business",
    "zerodha.com":           "Business",
    "groww.in":              "Business",
    "upstox.com":            "Business",
    "amazon.in":             "Business",
    "flipkart.com":          "Business",
    "snapdeal.com":          "Business",
    "razorpay.com":          "Business",
    "paytm.com":             "Business",
    "phonepe.com":           "Business",
    "indiamart.com":         "Business",
    "zoho.com":              "Business",
    "freshworks.com":        "Business",
    "amazon.com":            "Business",
    "ebay.com":              "Business",
    "meesho.com":            "Business",
    "myntra.com":            "Business",
    "ajio.com":              "Business",
    "nykaa.com":             "Business",
    "linkedin.com":          "Business",
    "shopify.com":           "Business",
    "stripe.com":            "Business",
    # ── Technology ────────────────────────────────────────────────────────────
    "github.com":            "Technology",
    "stackoverflow.com":     "Technology",
    "geeksforgeeks.org":     "Technology",
    "hackerrank.com":        "Technology",
    "leetcode.com":          "Technology",
    "codechef.com":          "Technology",
    "codeforces.com":        "Technology",
    "topcoder.com":          "Technology",
    "digit.in":              "Technology",
    "gadgets360.com":        "Technology",
    "91mobiles.com":         "Technology",
    "beebom.com":            "Technology",
    "techcrunch.com":        "Technology",
    "theverge.com":          "Technology",
    "wired.com":             "Technology",
    "arstechnica.com":       "Technology",
    "dev.to":                "Technology",
    "medium.com":            "Technology",
    "producthunt.com":       "Technology",
    "npmjs.com":             "Technology",
    "pypi.org":              "Technology",
    "replit.com":            "Technology",
    "codesandbox.io":        "Technology",
    # ── Education ─────────────────────────────────────────────────────────────
    "byjus.com":             "Education",
    "unacademy.com":         "Education",
    "vedantu.com":           "Education",
    "coursera.org":          "Education",
    "khanacademy.org":       "Education",
    "nptel.ac.in":           "Education",
    "swayam.gov.in":         "Education",
    "wikipedia.org":         "Education",
    "doubtnut.com":          "Education",
    "testbook.com":          "Education",
    "udemy.com":             "Education",
    "edx.org":               "Education",
    "britannica.com":        "Education",
    "scholastic.com":        "Education",
    "entrancecorner.com":    "Education",
    "collegedunia.com":      "Education",
    "shiksha.com":           "Education",
    "careers360.com":        "Education",
    # ── Health ────────────────────────────────────────────────────────────────
    "practo.com":            "Health",
    "1mg.com":               "Health",
    "netmeds.com":           "Health",
    "apollohospitals.com":   "Health",
    "webmd.com":             "Health",
    "healthline.com":        "Health",
    "pharmeasy.in":          "Health",
    "cult.fit":              "Health",
    "medscape.com":          "Health",
    "mayoclinic.org":        "Health",
    "nih.gov":               "Health",
    "nhp.gov.in":            "Health",
    "mohfw.gov.in":          "Health",
    # ── Gaming ────────────────────────────────────────────────────────────────
    "dream11.com":           "Gaming",
    "mpl.live":              "Gaming",
    "winzo.com":             "Gaming",
    "zupee.com":             "Gaming",
    "rummycircle.com":       "Gaming",
    "adda52.com":            "Gaming",
    "twitch.tv":             "Gaming",
    "ign.com":               "Gaming",
    "gamespot.com":          "Gaming",
    "steampowered.com":      "Gaming",
    "epicgames.com":         "Gaming",
    "xbox.com":              "Gaming",
    "playstation.com":       "Gaming",
    # ── Recreation ────────────────────────────────────────────────────────────
    "cricbuzz.com":          "Recreation",
    "espncricinfo.com":      "Recreation",
    "sportskeeda.com":       "Recreation",
    "indiahikes.com":        "Recreation",
    "bcci.tv":               "Recreation",
    "iplt20.com":            "Recreation",
    "espn.com":              "Recreation",
    "bleacherreport.com":    "Recreation",
    "fifa.com":              "Recreation",
    "icc-cricket.com":       "Recreation",
    "olympics.com":          "Recreation",
    "bwf.travel":            "Recreation",
    # ── Lifestyle ─────────────────────────────────────────────────────────────
    "zomato.com":            "Lifestyle",
    "swiggy.com":            "Lifestyle",
    "makemytrip.com":        "Lifestyle",
    "irctc.co.in":           "Lifestyle",
    "vogue.in":              "Lifestyle",
    "femina.in":             "Lifestyle",
    "mensxp.com":            "Lifestyle",
    "shaadi.com":            "Lifestyle",
    "instagram.com":         "Lifestyle",
    "pinterest.com":         "Lifestyle",
    "tripadvisor.com":       "Lifestyle",
    "booking.com":           "Lifestyle",
    "airbnb.com":            "Lifestyle",
    "ola.com":               "Lifestyle",
    "uber.com":              "Lifestyle",
    "rapido.bike":           "Lifestyle",
    # ── Kids ──────────────────────────────────────────────────────────────────
    "firstcry.com":          "Kids",
    "nickelodeonindia.com":  "Kids",
    "tinkle.in":             "Kids",
    "amarchitrakatha.com":   "Kids",
    "disneyindia.in":        "Kids",
    "chuchuTV.com":          "Kids",
    "pbs.org":               "Kids",
    "starfall.com":          "Kids",
    # ── Arts ──────────────────────────────────────────────────────────────────
    "gaana.com":             "Arts",
    "saavn.com":             "Arts",
    "jiosavan.com":          "Arts",
    "filmfare.com":          "Arts",
    "bollywoodhungama.com":  "Arts",
    "pratilipi.com":         "Arts",
    "rekhta.org":            "Arts",
    "bookmyshow.com":        "Arts",
    "youtube.com":           "Arts",
    "youtu.be":              "Arts",
    "netflix.com":           "Arts",
    "primevideo.com":        "Arts",
    "hotstar.com":           "Arts",
    "disneyplus.com":        "Arts",
    "zee5.com":              "Arts",
    "sonyliv.com":           "Arts",
    "voot.com":              "Arts",
    "spotify.com":           "Arts",
    "hungama.com":           "Arts",
    "wynk.in":               "Arts",
    "imdb.com":              "Arts",
    "rottentomatoes.com":    "Arts",
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
    """Returns domain/firstsegment for path-aware shortcut matching."""
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

# 150+ noise words: cookie banners, GDPR, nav, footer, social share, JS boilerplate
NOISE_WORDS = {
    # Cookie / GDPR
    "cookie", "cookies", "consent", "gdpr", "ccpa", "coppa",
    "privacy", "policy", "terms", "conditions", "legal", "disclaimer",
    "accept", "decline", "reject", "preferences", "manage", "settings",
    "necessary", "functional", "analytics", "marketing", "tracking",
    # Navigation boilerplate
    "home", "menu", "navigation", "navbar", "sidebar", "breadcrumb",
    "previous", "next", "back", "forward", "search", "close", "open",
    "toggle", "expand", "collapse", "show", "hide", "more", "less",
    # Footer boilerplate
    "copyright", "rights", "reserved", "trademark", "registered",
    "sitemap", "careers", "jobs", "about", "contact", "press",
    "investor", "relations", "feedback", "help", "support", "faq",
    "subscribe", "newsletter", "signup", "login", "logout", "register",
    "account", "profile", "password", "username", "email", "phone",
    # Social / sharing
    "share", "tweet", "post", "like", "follow", "comment", "reply",
    "facebook", "twitter", "whatsapp", "telegram", "instagram",
    # JS / browser
    "javascript", "enabled", "disabled", "browser", "reload", "refresh",
    "loading", "please", "enable", "continue", "proceed", "click",
    "download", "install", "update", "version", "app", "store",
    # Generic weak words
    "new", "latest", "best", "top", "free", "get", "use", "see",
    "also", "read", "view", "here", "now", "today", "week", "month",
    "the", "and", "for", "are", "but", "not", "you", "all", "can",
    "her", "was", "one", "our", "out", "day", "has", "him", "his",
    "how", "its", "may", "who", "did", "let", "put", "too", "way",
    # URL fragments
    "com", "www", "http", "https", "html", "php", "asp", "net", "org",
}

def extract_url_features(url: str) -> str:
    """URL → token string. Matches training feature engineering exactly."""
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
            "mil": "military government defense",
            "in":  "",
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
    """
    Weighted feature string — content-rich fields outweigh nav/footer noise.

    title            × 4   highest signal, author-curated
    meta_description × 3   author-written summary
    h1 / h2          × 2   structural keywords
    body_text[:800]  × 1   first 800 chars only (nav/footer comes later)
    url_features     × 2   always-available fallback signal
    """
    parts = []
    try:
        title = scraped.get("title", "").strip()
        meta  = scraped.get("meta_description", "").strip()
        h1    = scraped.get("h1", "").strip()
        h2    = scraped.get("h2", "").strip()
        body  = scraped.get("body_text", "").strip()
        # Fallback for older scraper that returns flat "text" key
        if not title and not body and "text" in scraped:
            body = scraped.get("text", "")
        url_feat = extract_url_features(url)

        if title:    parts += [title]      * 4
        if meta:     parts += [meta]       * 3
        if h1:       parts += [h1]         * 2
        if h2:       parts += [h2]         * 2
        if body:     parts += [body[:800]] * 1
        if url_feat: parts += [url_feat]   * 2
    except Exception:
        return extract_url_features(url)

    combined = " ".join(parts)
    clean    = [
        w for w in combined.lower().split()
        if w not in NOISE_WORDS and len(w) > 2
    ]
    return " ".join(clean)

def filter_noise(text: str) -> str:
    """Remove noise words from any text string."""
    return " ".join(
        w for w in text.lower().split()
        if w not in NOISE_WORDS and len(w) > 2
    )


# ─────────────────────────────────────────────
# PREDICTION  (LRU-cached, 256 entries)
# ─────────────────────────────────────────────
@lru_cache(maxsize=256)
def run_prediction(feature_string: str):
    """
    DistilBERT inference. Returns (category, confidence_%, top3_list).

    max_length=192: web feature strings never meaningfully exceed this.
                    Shorter = faster, same accuracy for this task.
    torch.inference_mode(): fastest CPU path, more aggressive than no_grad.
    LRU cache: repeated URLs served instantly without re-inference.
    """
    import numpy as np

    enc = tokenizer(
        feature_string,
        truncation=True, max_length=192,
        padding=True, return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    enc.pop("token_type_ids", None)   # DistilBERT has no token_type_ids

    with torch.inference_mode():
        logits = model(**enc).logits

    probs    = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    top3_idx = probs.argsort()[::-1][:3]
    top3 = [
        {
            "category":   CLASS_NAMES[i],
            "confidence": round(float(probs[i]) * 100, 2),
        }
        for i in top3_idx
    ]
    return top3[0]["category"], top3[0]["confidence"], top3


# ─────────────────────────────────────────────
# SMART CLASSIFY
# Central classification logic used by /url, /batch, /safe-check.
# Returns (category, confidence, top3, method).
# ─────────────────────────────────────────────
def smart_classify(url: str) -> tuple:
    """
    Classification pipeline:
      1. Path-aware shortcut  → instant, 99% confidence
      2. Domain shortcut      → instant, 99% confidence
      3. Scrape + weighted features → DistilBERT
      4. Confidence < 45%: cross-check URL-only features,
         pick whichever gives higher confidence.
         Handles React SPAs and auth-gated pages that scrape poorly.
    """
    domain           = get_domain(url)
    domain_with_path = get_domain_with_path(url)

    # Step 1 — path-aware shortcut (more specific)
    if domain_with_path in PATH_SHORTCUTS:
        cat = PATH_SHORTCUTS[domain_with_path]
        return cat, 99.0, [{"category": cat, "confidence": 99.0}], "path_shortcut"

    # Step 2 — bare domain shortcut
    if domain in DOMAIN_SHORTCUTS:
        cat = DOMAIN_SHORTCUTS[domain]
        return cat, 99.0, [{"category": cat, "confidence": 99.0}], "domain_shortcut"

    # Step 3 — scrape + DistilBERT
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

    category, confidence, top3 = run_prediction(features)

    # Step 4 — low-confidence fallback
    if confidence < CONFIDENCE_THRESHOLD and method == "combined_features":
        url_features = extract_url_features(url)
        if url_features.strip():
            cat_url, conf_url, top3_url = run_prediction(url_features)
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

# HEAD method on these 3 routes is required — RapidAPI probes HEAD /
# to verify health before routing any traffic. Without it: 405 → 503.
@app.api_route("/", methods=["GET", "HEAD"], tags=["Info"], include_in_schema=False)
async def home(request: Request):
    return JSONResponse({
        "message": "Website Category Classifier API v2.4.0",
        "model"  : "DistilBERT int8-quantized",
        "classes": CLASS_NAMES,
        "docs"   : "/docs",
    })

@app.api_route("/ping", methods=["GET", "HEAD"], tags=["Info"],
               include_in_schema=False)
async def ping(request: Request):
    return JSONResponse({"status": "alive"})

@app.api_route("/health", methods=["GET", "HEAD"], tags=["Info"])
async def health(request: Request):
    uptime = round(time.time() - startup_time, 1) if startup_time else None
    return JSONResponse({
        "status"        : "ok",
        "version"       : "2.4.0",
        "model_loaded"  : model is not None,
        "device"        : str(device),
        "classes"       : CLASS_NAMES,
        "classes_loaded": len(CLASS_NAMES),
        "uptime_seconds": uptime,
        "hf_repo"       : HF_MODEL_ID,
        "quantized"     : device.type == "cpu",
        "cache_info"    : run_prediction.cache_info()._asdict(),
    })

@app.get("/usage", tags=["Info"])
async def usage_info():
    """API capabilities summary — displayed in RapidAPI endpoint explorer."""
    return {
        "name"      : "Website Category Classifier",
        "version"   : "2.4.0",
        "categories": CLASS_NAMES,
        "endpoints" : {
            "POST /classify/url":   "Classify a live website by URL (scrapes + predicts)",
            "POST /classify/text":  "Classify raw text input",
            "POST /classify/batch": "Classify up to 20 URLs at once",
            "POST /safe-check":     "Brand safety / parental control verdict",
            "GET  /stats":          "API usage analytics",
            "GET  /stats/export":   "Download logs as CSV",
            "GET  /health":         "Service health check",
        },
        "rate_limits": {
            "classify": "30 req/min per IP",
            "batch":    " 5 req/min per IP",
        },
        "lime_note": (
            "LIME /explain removed in v2.4.0 to fit Render free tier (512 MB). "
            "Will be restored on upgrade to Render Starter (1 GB)."
        ),
    }


# ── 1. POST /classify/url ─────────────────────
@app.post("/classify/url", response_model=PredictionResult, tags=["Classify"])
@limiter.limit("30/minute")
async def classify_url(request: Request, body: URLRequest):
    """
    Classify a website by URL.
    Known domains → instant via shortcut. Unknown → scrape + DistilBERT.
    """
    start = time.time()
    ip    = get_ip(request)
    try:
        url = body.url.strip()
        if not is_valid_url(url):
            raise HTTPException(422, "Invalid URL format.")
        category, confidence, top3, method = smart_classify(url)
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
    """Classify raw text. Min 10 chars, trimmed to 5 000."""
    start = time.time()
    ip    = get_ip(request)
    try:
        text = body.text.strip()
        if len(text) < 10:
            raise HTTPException(422, "Text too short. Minimum 10 characters.")
        text = text[:5000]
        category, confidence, top3 = run_prediction(filter_noise(text))
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
    """Classify up to 20 URLs. Returns JSON + CSV export string."""
    start = time.time()
    ip    = get_ip(request)
    try:
        if not body.urls:
            raise HTTPException(422, "Provide at least 1 URL.")
        if len(body.urls) > 20:
            raise HTTPException(422, "Max 20 URLs per batch request.")
        results = []
        for url in body.urls:
            try:
                category, confidence, _, method = smart_classify(url)
                results.append({
                    "url"       : url,
                    "category"  : category,
                    "confidence": confidence,
                    "method"    : method,
                    "safe"      : category not in ADULT_CATEGORIES,
                    "adult_flag": category in ADULT_CATEGORIES,
                })
            except Exception as inner_e:
                results.append({
                    "url"       : url,
                    "category"  : "Error",
                    "confidence": 0.0,
                    "method"    : str(inner_e)[:80],
                    "safe"      : None,
                    "adult_flag": None,
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
            "total"     : len(results),
            "time_ms"   : elapsed,
            "results"   : results,
            "csv_export": "\n".join(csv_lines),
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
    """Brand safety / parental control verdict with adult + kids flags."""
    start = time.time()
    ip    = get_ip(request)
    try:
        url = body.url.strip()
        if not is_valid_url(url):
            raise HTTPException(422, "Invalid URL format.")
        category, confidence, _, method = smart_classify(url)
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
            "url"          : url,
            "category"     : category,
            "confidence"   : confidence,
            "safe"         : not adult_flag,
            "adult_flag"   : adult_flag,
            "kids_safe"    : kids_safe,
            "safe_for_kids": safe_for_kids,
            "verdict"      : verdict,
            "method"       : method,
            "time_ms"      : elapsed,
        }
    except HTTPException:
        raise
    except Exception as e:
        log_request(ip, "/safe-check", False, 0, input_url=body.url)
        raise HTTPException(500, f"Internal error: {str(e)}")


# ── 5. GET /stats ─────────────────────────────
@app.get("/stats", tags=["Analytics"])
async def get_stats(request: Request,
                    limit: int = Query(100, ge=1, le=1000)):
    """API usage analytics — powers the Streamlit analytics dashboard."""
    try:
        conn          = sqlite3.connect(DB_FILE,timeout=10)
        total         = conn.execute(
            "SELECT COUNT(*) FROM api_logs"
        ).fetchone()[0]
        success_count = conn.execute(
            "SELECT COUNT(*) FROM api_logs WHERE success=1"
        ).fetchone()[0]
        by_endpoint = conn.execute("""
            SELECT endpoint, COUNT(*) AS calls,
                   ROUND(AVG(time_ms),1) AS avg_ms,
                   SUM(CASE WHEN success=0 THEN 1 ELSE 0 END) AS errors
            FROM api_logs
            GROUP BY endpoint ORDER BY calls DESC
        """).fetchall()
        by_category = conn.execute("""
            SELECT category, COUNT(*) AS count
            FROM api_logs
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
                "success_rate"  : (
                    round(success_count / total * 100, 1) if total else 0
                ),
            },
            "by_endpoint": [
                {
                    "endpoint": r[0], "calls": r[1],
                    "avg_ms": r[2], "errors": r[3],
                }
                for r in by_endpoint
            ],
            "by_category": [
                {"category": r[0], "count": r[1]}
                for r in by_category
            ],
            "recent_requests": [
                {
                    "timestamp": r[0], "ip":         r[1],
                    "endpoint" : r[2], "input_url":  r[3],
                    "category" : r[4], "confidence": r[5],
                    "success"  : bool(r[6]),
                    "time_ms"  : r[7], "method":     r[8],
                }
                for r in recent
            ],
        }
    except Exception as e:
        raise HTTPException(500, f"Stats error: {str(e)}")


# ── 6. GET /stats/export ──────────────────────
@app.get("/stats/export", tags=["Analytics"])
async def export_logs():
    """Download all API logs as a CSV file."""
    try:
        conn = sqlite3.connect(DB_FILE)
        rows = conn.execute("""
            SELECT timestamp, ip, endpoint, input_url, input_text,
                   category, confidence, success, time_ms, method
            FROM api_logs ORDER BY id DESC
        """).fetchall()
        conn.close()
        header = (
            "timestamp,ip,endpoint,input_url,input_text,"
            "category,confidence,success,time_ms,method"
        )
        lines = [header]
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


# ═══════════════════════════════════════════════════════════════════════════════
# LIME /explain — DISABLED in v2.4.0 (Render free tier 512 MB)
#
# TO RE-ENABLE when upgrading to Render Starter ($7/mo, 1 GB RAM):
#   1. Add to requirements.txt:   lime==0.2.0.1
#   2. Uncomment the entire block below
#   3. Redeploy
#
# The block below is syntactically valid and self-contained.
# predict_proba_batch is defined inline — no other changes needed.
# ═══════════════════════════════════════════════════════════════════════════════

# class ExplainRequest(BaseModel):
#     url: str
#     n_words: int = 10
#
# def predict_proba_batch(texts: list):
#     """Batched inference for LIME. Not cached. Batch size 8 for RAM safety."""
#     import numpy as np
#     all_probs = []
#     for i in range(0, len(texts), 8):
#         batch = texts[i : i + 8]
#         enc   = tokenizer(
#             batch, truncation=True, max_length=128,
#             padding=True, return_tensors="pt",
#         )
#         enc = {k: v.to(device) for k, v in enc.items()}
#         enc.pop("token_type_ids", None)
#         with torch.inference_mode():
#             logits = model(**enc).logits
#         all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
#     return np.vstack(all_probs)
#
# async def _run_explain(request: Request, url: str, n_words: int):
#     from lime.lime_text import LimeTextExplainer
#     start = time.time()
#     ip    = get_ip(request)
#     try:
#         url     = url.strip()
#         n_words = min(max(n_words, 1), 20)
#         if not is_valid_url(url):
#             raise HTTPException(422, "Invalid URL format.")
#         try:
#             scraped = scrape_website(url)
#         except Exception:
#             scraped = {"error": "SCRAPE_FAILED"}
#         if scraped.get("error"):
#             features      = extract_url_features(url)
#             scrape_method = "url_features_only"
#         else:
#             features      = build_weighted_features(scraped, url)
#             scrape_method = "combined_features"
#         if not features.strip():
#             raise HTTPException(422, "Could not extract features.")
#         category, confidence, top3 = run_prediction(features)
#         pred_idx  = CLASS_NAMES.index(category)
#         explainer = LimeTextExplainer(
#             class_names=CLASS_NAMES, bow=False, random_state=42
#         )
#         exp = explainer.explain_instance(
#             features, predict_proba_batch,
#             labels=[pred_idx], num_features=n_words, num_samples=150,
#         )
#         word_weights = [
#             {
#                 "word":      word,
#                 "weight":    round(weight, 4),
#                 "direction": "supports" if weight > 0 else "opposes",
#             }
#             for word, weight in exp.as_list(label=pred_idx)
#         ]
#         elapsed = round((time.time() - start) * 1000, 1)
#         log_request(ip, "/explain", True, elapsed,
#                     input_url=url, category=category,
#                     confidence=confidence, method="lime")
#         return {
#             "url": url, "category": category, "confidence": confidence,
#             "top3": top3, "explanation": word_weights,
#             "scrape_method": scrape_method,
#             "note": (
#                 f"Words 'supports' pushed toward '{category}'. "
#                 f"'opposes' pushed against. 150 LIME samples."
#             ),
#             "time_ms": elapsed,
#         }
#     except HTTPException:
#         raise
#     except Exception as e:
#         log_request(ip, "/explain", False, 0, input_url=url)
#         raise HTTPException(500, f"Explain error: {str(e)}")
#
# @app.get("/explain", tags=["XAI"])
# @app.post("/explain", tags=["XAI"])
# @limiter.limit("5/minute")
# async def explain_get(
#     request: Request,
#     url    : str = Query(...),
#     n_words: int = Query(10, ge=1, le=20),
# ):
#     return await _run_explain(request, url, n_words)
#
# @app.post("/explain", tags=["XAI"])
# @limiter.limit("5/minute")
# async def explain_post(request: Request, body: ExplainRequest):
#     return await _run_explain(request, body.url, body.n_words)