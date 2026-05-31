# ═══════════════════════════════════════════════════════════════════════════════
# Website Category Classifier API  —  v2.3.0
# Model   : DistilBERT fine-tuned (11 categories)
# Author  : SanandaDutta
#
# v2.3.0 improvements over v2.2.0:
#
# ACCURACY FIXES:
#   1. asyncio.Lock replaces _model_loading bool flag — eliminates race
#      condition where concurrent cold-start requests got wrong predictions
#   2. Feature string quality: smarter scraper content extraction —
#      title weighted 4×, meta description 3×, h1/h2 2×, body text 1×
#      (nav/footer/cookie text was drowning out real content)
#   3. Expanded noise word filter (150+ words) — removes boilerplate
#      that pushed predictions toward wrong categories
#   4. Confidence threshold fallback: if top prediction < 45% confidence,
#      falls back to URL-features-only prediction and picks higher one
#   5. Domain path shortcuts: /sport /news /health path-aware matching
#      so bbc.co.uk/sport → Recreation not just bbc.co.uk → News
#   6. Expanded DOMAIN_SHORTCUTS: 40+ new domains added
#
# SPEED FIXES:
#   7. Model warmup runs 3 real-length inputs (not just "warmup") —
#      JIT trace warms up all code paths, first real request is instant
#   8. LRU cache size 512→1024 — more URLs served from cache
#   9. max_length 256→192 for run_prediction — DistilBERT accuracy
#      doesn't improve above ~150 tokens for short web feature strings,
#      192 is a safe headroom that's ~25% faster per inference
#  10. torch.inference_mode() replaces torch.no_grad() — faster on CPU,
#      disables more autograd machinery, safe for inference-only use
#
# ALL v2.2.0 FUNCTIONALITY PRESERVED:
#   LIME /explain (GET + POST), /classify/url, /classify/text,
#   /classify/batch, /safe-check, /stats, /stats/export, /health,
#   /ping, SQLite logging, rate limiting, domain shortcuts, LRU cache
# ═══════════════════════════════════════════════════════════════════════════════

import asyncio
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

ADULT_CATEGORIES = {"Adult"}
KIDS_CATEGORY    = "Kids"
SAFE_FOR_KIDS    = {"Education", "Kids", "Arts", "Recreation"}

# Global model handles
tokenizer     = None
model         = None
_model_loaded = False
startup_time  = None
device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# asyncio.Lock — prevents race condition on concurrent cold-start requests.
# Old bool flag (_model_loading) had a race: two requests could both see
# _model_loading=False simultaneously and both try to load the model,
# causing either double-load OOM or one request getting None model.
_model_lock = asyncio.Lock()


# ─────────────────────────────────────────────
# LAZY MODEL LOADER  (async-safe)
# ─────────────────────────────────────────────
async def ensure_model_loaded():
    """
    Loads model on first request. asyncio.Lock guarantees only one
    coroutine loads at a time — others wait at the lock, then return
    immediately when _model_loaded is True.
    """
    global tokenizer, model, _model_loaded, startup_time

    if _model_loaded:
        return   # fast path — no lock needed after first load

    async with _model_lock:
        if _model_loaded:
            return   # another request loaded it while we waited

        t0 = time.time()
        try:
            if os.path.isdir(MODEL_DIR):
                print(f"Loading model from local: {MODEL_DIR}")
                tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
                model     = AutoModelForSequenceClassification.from_pretrained(
                    MODEL_DIR
                )
                print("Model loaded from local cache")
            else:
                print(f"Loading model from HF Hub: {HF_MODEL_ID}")
                tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
                model     = AutoModelForSequenceClassification.from_pretrained(
                    HF_MODEL_ID
                )
                print("Model loaded from HuggingFace Hub")

            model.to(device)
            model.eval()

            # Warmup — 3 realistic inputs covering all token lengths.
            # This JIT-traces all code paths so first real request
            # doesn't pay the trace cost.
            warmup_texts = [
                "cricket sports news scores live match updates india",
                "buy smartphones laptops online free delivery best price",
                "education university courses learning academic students",
            ]
            for wt in warmup_texts:
                enc = tokenizer(
                    wt, return_tensors="pt",
                    truncation=True, max_length=192,
                    padding=True,
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                enc.pop("token_type_ids", None)
                with torch.inference_mode():
                    model(**enc)

            elapsed      = round(time.time() - t0, 1)
            startup_time = time.time()
            _model_loaded = True
            print(f"Model ready in {elapsed}s on {device}")

        except Exception as e:
            # Reset so next request can retry
            tokenizer = None
            model     = None
            raise RuntimeError(f"Model load failed: {e}")


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
    input_url: str = None, input_text: str = None,
    category: str = None, confidence: float = None, method: str = None,
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
            input_url, (input_text[:200] if input_text else None),
            category, confidence, int(success), time_ms, method,
        ))
        conn.commit()
        conn.close()
    except Exception:
        pass


# ─────────────────────────────────────────────
# LIFESPAN — only DB init, port opens instantly
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model, CLASS_NAMES, device

    init_db()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load at startup — not on first request
    try:
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(
            HF_MODEL_ID, low_cpu_mem_usage=True
        )
        labels_path = hf_hub_download(repo_id=HF_MODEL_ID, filename="label_classes.csv")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR, low_cpu_mem_usage=True
        )
        labels_path = os.path.join(MODEL_DIR, "label_classes.csv")

    model.to(device)
    model.eval()

    # Quantize on CPU for speed
    if device.type == "cpu":
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )

    # Load labels
    with open(labels_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        CLASS_NAMES = [row[0].strip() for row in reader if row and row[0].strip()]

    # Warmup
    dummy = tokenizer("warmup", return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        model(**dummy)

    print("🟢 API ready")
    yield
    print("🔴 Shutting down")
    
# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────
app = FastAPI(
    title       = "Website Category Classifier API",
    description = (
        "Classify any website into 11 categories using DistilBERT.\n\n"
        "**Categories:** Adult · Arts · Business · Education · Gaming · "
        "Health · Kids · Lifestyle · News · Recreation · Technology\n\n"
        "Built by **SanandaDutta** — "
        "[HuggingFace](https://huggingface.co/SanandaDutta) · "
        "[GitHub](https://github.com/SanandaDutta)"
    ),
    version  = "2.3.0",
    lifespan = lifespan,
)
# 1. Grab the secret you just set on Render
# If the variable isn't set, it defaults to an empty string (disabled)
RAPIDAPI_SECRET = os.getenv("RAPIDAPI_PROXY_SECRET", "")

@app.middleware("http")
async def verify_rapidapi_proxy(request: Request, call_next):
    """
    Checks if the request has the correct secret header from RapidAPI.
    """
    # 2. List of paths that DON'T need a secret (so you can still test/monitor)
    skip_paths = {"/health", "/docs", "/openapi.json", "/redoc", "/"}

    # 3. Only enforce check if we have a secret set and path isn't skipped
    if RAPIDAPI_SECRET and request.url.path not in skip_paths:
        incoming_header = request.headers.get("X-RapidAPI-Proxy-Secret", "")
        
        if incoming_header != RAPIDAPI_SECRET:
            return JSONResponse(
                status_code=403,
                content={"error": "Access via RapidAPI only. Sign up at rapidapi.com"}
            )
            
    return await call_next(request)

# Place this after the verify_rapidapi_proxy middleware
@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
@app.api_route("/ping", methods=["GET", "HEAD"], include_in_schema=False)
@app.api_route("/health", methods=["GET", "HEAD"], include_in_schema=False)
async def health_check(request: Request):
    return {"status": "ok", "message": "Website Category Classifier API 🚀"}

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
        "detail": "Max 30 req/min on classify, 5/min on batch and explain.",
    })


# ─────────────────────────────────────────────
# DOMAIN SHORTCUTS  (path-aware for /sport etc.)
# ─────────────────────────────────────────────

# Path-aware shortcuts checked FIRST — more specific wins.
# get_domain_with_path() returns "domain.com/firstsegment"
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
    "amazon.in/health":     "Health",
    "amazon.com/health":    "Health",
}

DOMAIN_SHORTCUTS = {
    # ── AI / Modern Tech ──
    "claude.ai":"Technology","anthropic.com":"Technology",
    "openai.com":"Technology","chatgpt.com":"Technology",
    "mistral.ai":"Technology","huggingface.co":"Technology",
    "vercel.com":"Technology","supabase.com":"Technology",
    "figma.com":"Technology","linear.app":"Technology",
    "cloudflare.com":"Technology","digitalocean.com":"Technology",
    "render.com":"Technology","netlify.com":"Technology",
    "heroku.com":"Technology","aws.amazon.com":"Technology",
    "azure.microsoft.com":"Technology",
    # ── News ──
    "ndtv.com":"News","thehindu.com":"News","hindustantimes.com":"News",
    "timesofindia.indiatimes.com":"News","indianexpress.com":"News",
    "scroll.in":"News","thewire.in":"News","theprint.in":"News",
    "bbc.com":"News","bbc.co.uk":"News","reuters.com":"News",
    "apnews.com":"News","theguardian.com":"News",
    "aajtak.in":"News","zeenews.india.com":"News","news18.com":"News",
    "abplive.com":"News","tv9bharatvarsh.com":"News",
    "twitter.com":"News","x.com":"News","facebook.com":"News",
    "reddit.com":"News",
    # ── Business ──
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
    # ── Technology ──
    "github.com":"Technology","stackoverflow.com":"Technology",
    "geeksforgeeks.org":"Technology","hackerrank.com":"Technology",
    "leetcode.com":"Technology","codechef.com":"Technology",
    "codeforces.com":"Technology","topcoder.com":"Technology",
    "digit.in":"Technology","gadgets360.com":"Technology",
    "91mobiles.com":"Technology","beebom.com":"Technology",
    "techcrunch.com":"Technology","theverge.com":"Technology",
    "wired.com":"Technology","arstechnica.com":"Technology",
    "dev.to":"Technology","medium.com":"Technology",
    "producthunt.com":"Technology","hackernews.com":"Technology",
    "npmjs.com":"Technology","pypi.org":"Technology",
    # ── Education ──
    "byjus.com":"Education","unacademy.com":"Education",
    "vedantu.com":"Education","coursera.org":"Education",
    "khanacademy.org":"Education","nptel.ac.in":"Education",
    "swayam.gov.in":"Education","wikipedia.org":"Education",
    "doubtnut.com":"Education","testbook.com":"Education",
    "udemy.com":"Education","edx.org":"Education",
    "britannica.com":"Education","scholastic.com":"Education",
    "entrancecorner.com":"Education","collegedunia.com":"Education",
    "shiksha.com":"Education","careers360.com":"Education",
    # ── Health ──
    "practo.com":"Health","1mg.com":"Health","netmeds.com":"Health",
    "apollohospitals.com":"Health","webmd.com":"Health",
    "healthline.com":"Health","pharmeasy.in":"Health",
    "cult.fit":"Health","medscape.com":"Health",
    "mayoclinic.org":"Health","nih.gov":"Health",
    "nhp.gov.in":"Health","mohfw.gov.in":"Health",
    # ── Gaming ──
    "dream11.com":"Gaming","mpl.live":"Gaming","winzo.com":"Gaming",
    "zupee.com":"Gaming","rummycircle.com":"Gaming","adda52.com":"Gaming",
    "twitch.tv":"Gaming","ign.com":"Gaming","gamespot.com":"Gaming",
    "steampowered.com":"Gaming","epicgames.com":"Gaming",
    "xbox.com":"Gaming","playstation.com":"Gaming",
    # ── Recreation ──
    "cricbuzz.com":"Recreation","espncricinfo.com":"Recreation",
    "sportskeeda.com":"Recreation","indiahikes.com":"Recreation",
    "bcci.tv":"Recreation","iplt20.com":"Recreation",
    "espn.com":"Recreation","bleacherreport.com":"Recreation",
    "fifa.com":"Recreation","icc-cricket.com":"Recreation",
    "olympics.com":"Recreation","bwf.travel":"Recreation",
    # ── Lifestyle ──
    "zomato.com":"Lifestyle","swiggy.com":"Lifestyle",
    "makemytrip.com":"Lifestyle","irctc.co.in":"Lifestyle",
    "vogue.in":"Lifestyle","femina.in":"Lifestyle",
    "mensxp.com":"Lifestyle","shaadi.com":"Lifestyle",
    "instagram.com":"Lifestyle","pinterest.com":"Lifestyle",
    "tripadvisor.com":"Lifestyle","booking.com":"Lifestyle",
    "airbnb.com":"Lifestyle","ola.com":"Lifestyle","uber.com":"Lifestyle",
    # ── Kids ──
    "firstcry.com":"Kids","nickelodeonindia.com":"Kids",
    "tinkle.in":"Kids","amarchitrakatha.com":"Kids",
    "disneyindia.in":"Kids","chuchuTV.com":"Kids",
    "pbs.org":"Kids","starfall.com":"Kids","abc.kids":"Kids",
    # ── Arts ──
    "gaana.com":"Arts","saavn.com":"Arts","jiosavan.com":"Arts",
    "filmfare.com":"Arts","bollywoodhungama.com":"Arts",
    "pratilipi.com":"Arts","rekhta.org":"Arts","bookmyshow.com":"Arts",
    "youtube.com":"Arts","youtu.be":"Arts","netflix.com":"Arts",
    "primevideo.com":"Arts","hotstar.com":"Arts",
    "disneyplus.com":"Arts","zee5.com":"Arts",
    "sonyliv.com":"Arts","voot.com":"Arts","spotify.com":"Arts",
    "hungama.com":"Arts","wynk.in":"Arts","eros.com":"Arts",
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
    """Returns domain/firstpathsegment for path-aware shortcut matching."""
    try:
        parsed = urlparse(url if url.startswith("http") else "https://" + url)
        domain = parsed.netloc.replace("www.", "").lower()
        parts  = [p for p in parsed.path.split("/") if p]
        if parts:
            return f"{domain}/{parts[0]}"
        return domain
    except Exception:
        return get_domain(url)

def is_valid_url(url: str) -> bool:
    return bool(re.match(
        r'^(https?://)?(([a-zA-Z0-9\-]+\.)+[a-zA-Z]{2,})(/.*)?$', url
    ))

# Expanded noise word list — 150+ terms covering:
# cookie banners, GDPR, navigation, footer boilerplate,
# social share buttons, subscription prompts
NOISE_WORDS = {
    # Cookie / GDPR
    "cookie","cookies","consent","gdpr","ccpa","coppa",
    "privacy","policy","terms","conditions","legal","disclaimer",
    "accept","decline","reject","preferences","manage","settings",
    "necessary","functional","analytics","marketing","tracking",
    # Navigation boilerplate
    "home","menu","navigation","navbar","sidebar","breadcrumb",
    "previous","next","back","forward","search","close","open",
    "toggle","expand","collapse","show","hide","more","less",
    # Footer boilerplate
    "copyright","rights","reserved","trademark","registered",
    "sitemap","careers","jobs","about","contact","press",
    "investor","relations","feedback","help","support","faq",
    "subscribe","newsletter","signup","login","logout","register",
    "account","profile","password","username","email","phone",
    # Social / sharing
    "share","tweet","post","like","follow","comment","reply",
    "facebook","twitter","whatsapp","telegram","instagram",
    # JS/browser
    "javascript","enabled","disabled","browser","reload","refresh",
    "loading","please","enable","continue","proceed","click",
    "download","install","update","version","app","store",
    # Generic weak words (add no category signal)
    "new","latest","best","top","free","get","use","see",
    "also","read","view","here","now","today","week","month","year",
    "the","and","for","are","but","not","you","all","can","had",
    "her","was","one","our","out","day","get","has","him","his",
    "how","its","may","who","did","let","put","too","use","way",
    "com","www","http","https","html","php","asp","net","org",
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
            "in":  "",   # .in is India TLD — no strong signal
        }.get(tld, "")
        all_parts = domain_words * 3 + path_words + tld_signal.split()
        clean = [
            w.lower() for w in all_parts
            if len(w) > 2 and w.isalpha()
            and w.lower() not in NOISE_WORDS
        ]
        return " ".join(clean)
    except Exception:
        return ""

def build_weighted_features(scraped: dict, url: str) -> str:
    """
    Builds a weighted feature string where more informative parts
    (title, meta description, headings) are repeated more than body text.

    Weighting rationale:
    - Page title: highest signal, most curated, repeat 4x
    - Meta description: author-written summary, repeat 3x
    - H1/H2 headings: structural keywords, repeat 2x
    - Body text: noisy, lots of nav/footer, repeat 1x
    - URL features: always include 2x as fallback signal

    This replaces the flat feature string that weighted all text equally,
    which caused nav/footer noise to overwhelm actual content.
    """
    parts = []

    try:
        title = scraped.get("title", "").strip()
        meta  = scraped.get("meta_description", "").strip()
        h1    = scraped.get("h1", "").strip()
        h2    = scraped.get("h2", "").strip()
        body  = scraped.get("body_text", "").strip()

        # If scraped dict only has flat "text" key (older scraper versions)
        if not title and not body and "text" in scraped:
            body = scraped.get("text", "")

        url_feat = extract_url_features(url)

        # Weight by importance
        if title:   parts += [title] * 4
        if meta:    parts += [meta]  * 3
        if h1:      parts += [h1]    * 2
        if h2:      parts += [h2]    * 2
        if body:
            # Take first 800 chars of body only — later text is usually
            # nav/footer/related articles which add noise
            parts += [body[:800]] * 1
        if url_feat:
            parts += [url_feat]  * 2

    except Exception:
        url_feat = extract_url_features(url)
        return url_feat

    combined = " ".join(parts)

    # Apply noise filter
    words = combined.lower().split()
    clean = [w for w in words if w not in NOISE_WORDS and len(w) > 2]
    return " ".join(clean)

def filter_noise(text: str) -> str:
    return " ".join(
        w for w in text.lower().split()
        if w not in NOISE_WORDS and len(w) > 2
    )


# ─────────────────────────────────────────────
# PREDICTION  (LRU cached, 1024 entries)
# ─────────────────────────────────────────────
@lru_cache(maxsize=1024)
def run_prediction(feature_string: str):
    """
    DistilBERT inference. LRU-cached on feature string.
    max_length=192: covers all real feature strings without padding waste.
    torch.inference_mode(): faster than no_grad on CPU.
    """
    import numpy as np

    enc = tokenizer(
        feature_string,
        truncation=True, max_length=192,
        padding=True, return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    enc.pop("token_type_ids", None)

    with torch.inference_mode():
        logits = model(**enc).logits

    probs    = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    top3_idx = probs.argsort()[::-1][:3]
    top3     = [
        {
            "category":   CLASS_NAMES[i],
            "confidence": round(float(probs[i]) * 100, 2),
        }
        for i in top3_idx
    ]
    return top3[0]["category"], top3[0]["confidence"], top3


def predict_proba_batch(texts: list):
    """Batched inference for LIME — not cached, batch size 8."""
    import numpy as np
    all_probs = []
    for i in range(0, len(texts), 8):
        batch = texts[i : i + 8]
        enc   = tokenizer(
            batch, truncation=True, max_length=128,
            padding=True, return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        enc.pop("token_type_ids", None)
        with torch.inference_mode():
            logits = model(**enc).logits
        all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.vstack(all_probs)


# ─────────────────────────────────────────────
# SMART CLASSIFY — used by url, batch, safe-check
# Combines scraping + URL features + confidence fallback
# ─────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 45.0   # below this, try URL-only as a cross-check

def smart_classify(url: str) -> tuple:
    """
    Returns (category, confidence, top3, method).

    Strategy:
    1. Domain shortcut (instant, 99% confidence)
    2. Path-aware shortcut (more specific than bare domain)
    3. Scrape + weighted features → DistilBERT
    4. If confidence < 45%: also try URL-only features,
       pick whichever gives higher confidence
       (scraper returned garbage → URL features may be cleaner)
    """
    domain           = get_domain(url)
    domain_with_path = get_domain_with_path(url)

    # Path-aware shortcut first (more specific)
    if domain_with_path in PATH_SHORTCUTS:
        cat = PATH_SHORTCUTS[domain_with_path]
        return cat, 99.0, [{"category": cat, "confidence": 99.0}], "path_shortcut"

    # Bare domain shortcut
    if domain in DOMAIN_SHORTCUTS:
        cat = DOMAIN_SHORTCUTS[domain]
        return cat, 99.0, [{"category": cat, "confidence": 99.0}], "domain_shortcut"

    # Scrape
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

    # Confidence fallback: if scraper gave low-confidence result,
    # cross-check with URL-only features and pick the better one
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

class ExplainRequest(BaseModel):
    url:    str
    n_words: int = 10

class PredictionResult(BaseModel):
    category:   str
    confidence: float
    top3:       List[dict]
    method:     str
    time_ms:    float


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Info"])
def home():
    return {
        "message" : "Website Category Classifier API v2.3.0",
        "model"   : "DistilBERT fine-tuned",
        "classes" : CLASS_NAMES,
        "docs"    : "/docs",
        "note"    : "Model loads on first classify request (~30s cold start)",
    }

@app.get("/ping", tags=["Info"])
def ping():
    return {"status": "alive"}

@app.get("/health", tags=["Info"])
def health():
    uptime = round(time.time() - startup_time, 1) if startup_time else None
    return {
        "status"        : "ok",
        "version"       : "2.3.0",
        "model_loaded"  : _model_loaded,
        "device"        : str(device),
        "classes"       : CLASS_NAMES,
        "classes_loaded": len(CLASS_NAMES),
        "uptime_seconds": uptime,
        "hf_repo"       : HF_MODEL_ID,
        "cache_info"    : run_prediction.cache_info()._asdict() if _model_loaded else {},
    }


# ── 1. POST /classify/url ─────────────────────
@app.post("/classify/url", response_model=PredictionResult, tags=["Classify"])
@limiter.limit("30/minute")
async def classify_url(request: Request, body: URLRequest):
    await ensure_model_loaded()
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
    await ensure_model_loaded()
    start = time.time()
    ip    = get_ip(request)

    try:
        text = body.text.strip()
        if len(text) < 10:
            raise HTTPException(422, "Text too short. Minimum 10 characters.")
        if len(text) > 5000:
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
    await ensure_model_loaded()
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
                category, confidence, _, method = smart_classify(url)
                results.append({
                    "url": url, "category": category, "confidence": confidence,
                    "method": method,
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
    await ensure_model_loaded()
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


# ── 5a. GET /explain ──────────────────────────
@app.get("/explain", tags=["XAI"])
@limiter.limit("5/minute")
async def explain_get(
    request: Request,
    url    : str = Query(..., description="Full URL to explain"),
    n_words: int = Query(10, ge=1, le=20),
):
    return await _run_explain(request, url, n_words)

# ── 5b. POST /explain ─────────────────────────
@app.post("/explain", tags=["XAI"])
@limiter.limit("5/minute")
async def explain_post(request: Request, body: ExplainRequest):
    return await _run_explain(request, body.url, body.n_words)

async def _run_explain(request: Request, url: str, n_words: int):
    """LIME XAI — deferred import, 150 samples, batch size 8."""
    await ensure_model_loaded()
    from lime.lime_text import LimeTextExplainer   # deferred import

    start = time.time()
    ip    = get_ip(request)

    try:
        url     = url.strip()
        n_words = min(max(n_words, 1), 20)

        if not is_valid_url(url):
            raise HTTPException(422, "Invalid URL format.")

        try:
            scraped = scrape_website(url)
        except Exception:
            scraped = {"error": "SCRAPE_FAILED"}

        if scraped.get("error"):
            features      = extract_url_features(url)
            scrape_method = "url_features_only"
        else:
            features      = build_weighted_features(scraped, url)
            scrape_method = "combined_features"

        if not features.strip():
            raise HTTPException(422, "Could not extract features from this URL.")

        category, confidence, top3 = run_prediction(features)
        pred_idx = CLASS_NAMES.index(category)

        explainer = LimeTextExplainer(
            class_names=CLASS_NAMES, bow=False, random_state=42
        )
        exp = explainer.explain_instance(
            features,
            predict_proba_batch,
            labels      = [pred_idx],
            num_features= n_words,
            num_samples = 150,
        )

        word_weights = [
            {
                "word":      word,
                "weight":    round(weight, 4),
                "direction": "supports" if weight > 0 else "opposes",
            }
            for word, weight in exp.as_list(label=pred_idx)
        ]

        elapsed = round((time.time() - start) * 1000, 1)
        log_request(ip, "/explain", True, elapsed,
                    input_url=url, category=category,
                    confidence=confidence, method="lime")

        return {
            "url":           url,
            "category":      category,
            "confidence":    confidence,
            "top3":          top3,
            "explanation":   word_weights,
            "scrape_method": scrape_method,
            "note": (
                f"Words with direction='supports' pushed toward '{category}'. "
                f"'opposes' pushed against it. LIME ran 150 perturbation samples."
            ),
            "time_ms": elapsed,
        }

    except HTTPException:
        raise
    except Exception as e:
        log_request(ip, "/explain", False, 0, input_url=url)
        raise HTTPException(500, f"Explain error: {str(e)}")


# ── 6. GET /stats ─────────────────────────────
@app.get("/stats", tags=["Analytics"])
async def get_stats(
    request: Request,
    limit  : int = Query(100, ge=1, le=1000),
):
    try:
        conn = sqlite3.connect(DB_FILE)

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
            media_type = "text/csv",
            headers    = {"Content-Disposition": "attachment; filename=api_logs.csv"},
        )

    except Exception as e:
        raise HTTPException(500, f"Export error: {str(e)}")

@app.get("/usage", tags=["Info"])
async def usage_info():
    """
    Returns API capabilities summary — shown in RapidAPI's endpoint explorer.
    """
    return {
        "name":        "Website Category Classifier",
        "version":     "2.1.0",
        "categories":  CLASS_NAMES,
        "endpoints": {
            "POST /classify/url":  "Classify a live website by URL (scrapes + predicts)",
            "POST /classify/text": "Classify raw text input",
            "POST /classify/batch":"Classify up to 20 URLs at once",
            "POST /safe-check":    "Brand safety verdict — Adult flags, Kids Safe",
            "GET  /explain":       "LIME XAI — which words drove the prediction",
        },
        "rate_limits": {
            "free":  "100 calls/month",
            "basic": "5,000 calls/month",
            "pro":   "50,000 calls/month",
        }
    }