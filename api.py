# ═══════════════════════════════════════════════════════════════════════════════
# Website Category Classifier API  —  v2.2.0
# Model   : DistilBERT fine-tuned (11 categories)
# Author  : SanandaDutta
#
# v2.2.0 memory fixes (Render free tier 512MB):
#   - Lazy model loading: loads on first request, NOT at startup
#     → port opens instantly, Render sees it as healthy
#     → model loads once on first real request (~30s cold start)
#   - Removed quantize_dynamic from startup: it temporarily doubles
#     RAM (holds original + quantized simultaneously → OOM)
#   - numpy/pandas/scipy imports deferred: only imported when needed
#   - CLASS_NAMES hardcoded: removes hf_hub_download + CSV parse at startup
#   - predict_proba_batch batch size reduced 16→8: safer for LIME on 512MB
#   - Removed duplicate route definitions (/, /health, /ping defined twice)
#   - LIME import deferred inside /explain: not loaded until first LIME call
#   - All functionality preserved: LIME, batch, safe-check, stats, explain
# ═══════════════════════════════════════════════════════════════════════════════

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

# Hardcoded — removes hf_hub_download + pandas CSV parse at startup.
# If you add/remove categories, update this list and redeploy.
CLASS_NAMES = [
    "Adult", "Arts", "Business", "Education", "Gaming",
    "Health", "Kids", "Lifestyle", "News", "Recreation", "Technology",
]

# Safety constants
ADULT_CATEGORIES = {"Adult"}
KIDS_CATEGORY    = "Kids"
SAFE_FOR_KIDS    = {"Education", "Kids", "Arts", "Recreation"}

# Global model handles — None until first request triggers lazy load
tokenizer      = None
model          = None
_model_loaded  = False
_model_loading = False   # prevents duplicate concurrent loads
startup_time   = None
device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# LAZY MODEL LOADER
# Called by every prediction endpoint before inference.
# Thread-safe enough for single-worker Render free tier.
# ─────────────────────────────────────────────
def ensure_model_loaded():
    global tokenizer, model, _model_loaded, _model_loading, startup_time

    if _model_loaded:
        return

    if _model_loading:
        # Another request triggered load — wait for it
        import time as _t
        for _ in range(120):   # wait up to 60s
            _t.sleep(0.5)
            if _model_loaded:
                return
        raise HTTPException(503, "Model is still loading. Retry in 30 seconds.")

    _model_loading = True
    t0 = time.time()

    try:
        if os.path.isdir(MODEL_DIR):
            print(f"Loading model from local: {MODEL_DIR}")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
            model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
            print("Model loaded from local cache")
        else:
            print(f"Loading model from HF Hub: {HF_MODEL_ID}")
            tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
            model     = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_ID)
            print("Model loaded from HuggingFace Hub")

        model.to(device)
        model.eval()

        # Warmup — prevents slow first real request
        dummy = tokenizer(
            "warmup", return_tensors="pt",
            truncation=True, max_length=64,
        )
        dummy = {k: v.to(device) for k, v in dummy.items()}
        dummy.pop("token_type_ids", None)
        with torch.no_grad():
            model(**dummy)

        elapsed = round(time.time() - t0, 1)
        startup_time  = time.time()
        _model_loaded = True
        print(f"Model ready in {elapsed}s on {device}")

    except Exception as e:
        _model_loading = False
        raise RuntimeError(f"Model load failed: {e}")

    finally:
        _model_loading = False


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
# LIFESPAN — minimal startup, just DB init
# Model loading moved to lazy loader above.
# Port opens in <2 seconds — Render sees healthy.
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 55)
    print("Website Category Classifier API  v2.2.0")
    print("=" * 55)
    init_db()
    print("SQLite ready")
    print("Port opening — model loads on first request")
    print("=" * 55)
    yield
    print("Shutting down...")


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
    version  = "2.2.0",
    lifespan = lifespan,
)

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
        "detail": "Max 30 req/min on classify endpoints. Max 5/min on batch and explain.",
    })


# ─────────────────────────────────────────────
# DOMAIN SHORTCUTS
# ─────────────────────────────────────────────
DOMAIN_SHORTCUTS = {
    # ── AI / Tech ──
    "claude.ai":"Technology","anthropic.com":"Technology",
    "openai.com":"Technology","chatgpt.com":"Technology",
    "mistral.ai":"Technology","huggingface.co":"Technology",
    "vercel.com":"Technology","supabase.com":"Technology",
    "figma.com":"Technology","linear.app":"Technology",
    "cloudflare.com":"Technology","digitalocean.com":"Technology",
    # ── News ──
    "ndtv.com":"News","thehindu.com":"News","hindustantimes.com":"News",
    "timesofindia.indiatimes.com":"News","indianexpress.com":"News",
    "scroll.in":"News","thewire.in":"News","theprint.in":"News",
    "bbc.com":"News","bbc.co.uk":"News","reuters.com":"News",
    "aajtak.in":"News","zeenews.india.com":"News","news18.com":"News",
    "twitter.com":"News","x.com":"News","facebook.com":"News",
    "reddit.com":"News",
    # ── Business ──
    "moneycontrol.com":"Business","economictimes.indiatimes.com":"Business",
    "livemint.com":"Business","business-standard.com":"Business",
    "zerodha.com":"Business","groww.in":"Business",
    "amazon.in":"Business","flipkart.com":"Business",
    "razorpay.com":"Business","paytm.com":"Business",
    "phonepe.com":"Business","indiamart.com":"Business","zoho.com":"Business",
    "amazon.com":"Business","ebay.com":"Business","meesho.com":"Business",
    "myntra.com":"Business","ajio.com":"Business","linkedin.com":"Business",
    # ── Technology ──
    "github.com":"Technology","stackoverflow.com":"Technology",
    "geeksforgeeks.org":"Technology","hackerrank.com":"Technology",
    "leetcode.com":"Technology","codechef.com":"Technology",
    "digit.in":"Technology","gadgets360.com":"Technology",
    "91mobiles.com":"Technology","beebom.com":"Technology",
    # ── Education ──
    "byjus.com":"Education","unacademy.com":"Education",
    "vedantu.com":"Education","coursera.org":"Education",
    "khanacademy.org":"Education","nptel.ac.in":"Education",
    "swayam.gov.in":"Education","wikipedia.org":"Education",
    "doubtnut.com":"Education","testbook.com":"Education",
    # ── Health ──
    "practo.com":"Health","1mg.com":"Health","netmeds.com":"Health",
    "apollohospitals.com":"Health","webmd.com":"Health",
    "healthline.com":"Health","pharmeasy.in":"Health","cult.fit":"Health",
    # ── Gaming ──
    "dream11.com":"Gaming","mpl.live":"Gaming","winzo.com":"Gaming",
    "zupee.com":"Gaming","rummycircle.com":"Gaming","adda52.com":"Gaming",
    "twitch.tv":"Gaming",
    # ── Recreation ──
    "cricbuzz.com":"Recreation","espncricinfo.com":"Recreation",
    "sportskeeda.com":"Recreation","indiahikes.com":"Recreation",
    "bcci.tv":"Recreation","iplt20.com":"Recreation",
    # ── Lifestyle ──
    "zomato.com":"Lifestyle","swiggy.com":"Lifestyle",
    "makemytrip.com":"Lifestyle","nykaa.com":"Lifestyle",
    "vogue.in":"Lifestyle","femina.in":"Lifestyle",
    "mensxp.com":"Lifestyle","shaadi.com":"Lifestyle",
    "instagram.com":"Lifestyle","pinterest.com":"Lifestyle",
    # ── Kids ──
    "firstcry.com":"Kids","nickelodeonindia.com":"Kids",
    "tinkle.in":"Kids","amarchitrakatha.com":"Kids",
    "disneyindia.in":"Kids","chuchuTV.com":"Kids",
    # ── Arts ──
    "gaana.com":"Arts","saavn.com":"Arts","filmfare.com":"Arts",
    "bollywoodhungama.com":"Arts","pratilipi.com":"Arts",
    "rekhta.org":"Arts","bookmyshow.com":"Arts",
    "youtube.com":"Arts","youtu.be":"Arts","netflix.com":"Arts",
    "primevideo.com":"Arts","hotstar.com":"Arts",
    "disneyplus.com":"Arts","zee5.com":"Arts",
    "sonyliv.com":"Arts","voot.com":"Arts","spotify.com":"Arts",
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

def is_valid_url(url: str) -> bool:
    return bool(re.match(
        r'^(https?://)?(([a-zA-Z0-9\-]+\.)+[a-zA-Z]{2,})(/.*)?$', url
    ))

NOISE_WORDS = {
    "cookie","cookies","consent","privacy","policy","terms",
    "children","child","coppa","gdpr","newsletter","subscribe",
    "advertisement","copyright","rights","reserved",
    "javascript","enabled","browser","please","enable","accept",
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
            "edu":"education university college academic",
            "gov":"government official public authority",
            "org":"organization nonprofit charity",
            "ac": "academic university college",
            "mil":"military government defense",
        }.get(tld, "")
        all_parts = domain_words * 3 + path_words + tld_signal.split()
        clean     = [w.lower() for w in all_parts
                     if len(w) > 2 and w.isalpha() and w.lower() not in NOISE_WORDS]
        return " ".join(clean)
    except Exception:
        return ""

def filter_noise(text: str) -> str:
    return " ".join(
        w for w in text.lower().split()
        if w not in NOISE_WORDS and len(w) > 2
    )


# ─────────────────────────────────────────────
# PREDICTION  (LRU cached)
# ─────────────────────────────────────────────
@lru_cache(maxsize=512)
def run_prediction(feature_string: str):
    """Runs DistilBERT inference. LRU-cached on feature string."""
    enc = tokenizer(
        feature_string,
        truncation=True, max_length=256,
        padding=True, return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    enc.pop("token_type_ids", None)

    with torch.no_grad():
        logits = model(**enc).logits

    import numpy as np
    probs    = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    top3_idx = probs.argsort()[::-1][:3]
    top3     = [
        {"category": CLASS_NAMES[i], "confidence": round(float(probs[i]) * 100, 2)}
        for i in top3_idx
    ]
    return top3[0]["category"], top3[0]["confidence"], top3


def predict_proba_batch(texts: list):
    """
    Batched inference for LIME — not cached.
    Batch size 8 (down from 16) to stay within 512MB RAM during LIME.
    """
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
        with torch.no_grad():
            logits = model(**enc).logits
        all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.vstack(all_probs)


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
    url: str
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
        "message" : "Website Category Classifier API v2.2.0",
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
        "version"       : "2.2.0",
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
    ensure_model_loaded()
    start = time.time()
    ip    = get_ip(request)

    try:
        url = body.url.strip()
        if not is_valid_url(url):
            raise HTTPException(422, "Invalid URL format.")

        domain = get_domain(url)

        if domain in DOMAIN_SHORTCUTS:
            category = DOMAIN_SHORTCUTS[domain]
            elapsed  = round((time.time() - start) * 1000, 1)
            log_request(ip, "/classify/url", True, elapsed,
                        input_url=url, category=category,
                        confidence=99.0, method="domain_shortcut")
            return PredictionResult(
                category=category, confidence=99.0,
                top3=[{"category": category, "confidence": 99.0}],
                method="domain_shortcut", time_ms=elapsed,
            )

        try:
            scraped = scrape_website(url)
        except Exception:
            scraped = {"error": "SCRAPE_FAILED"}

        if scraped.get("error"):
            features = extract_url_features(url)
            method   = "url_features_only"
        else:
            features = filter_noise(
                (build_feature_string(scraped) + " " + extract_url_features(url)).strip()
            )
            method = "combined_features"

        if not features.strip():
            raise HTTPException(422, "Could not extract any features from this URL.")

        category, confidence, top3 = run_prediction(features)
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
    ensure_model_loaded()
    start = time.time()
    ip    = get_ip(request)

    try:
        text = body.text.strip()
        if len(text) < 10:
            raise HTTPException(422, "Text too short. Minimum 10 characters.")
        if len(text) > 5000:
            text = text[:5000]

        category, confidence, top3 = run_prediction(text)
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
    ensure_model_loaded()
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
                domain = get_domain(url)

                if domain in DOMAIN_SHORTCUTS:
                    cat = DOMAIN_SHORTCUTS[domain]
                    results.append({
                        "url": url, "category": cat, "confidence": 99.0,
                        "method": "domain_shortcut",
                        "safe": cat not in ADULT_CATEGORIES,
                        "adult_flag": cat in ADULT_CATEGORIES,
                    })
                    continue

                try:
                    scraped = scrape_website(url)
                except Exception:
                    scraped = {"error": "SCRAPE_FAILED"}

                if scraped.get("error"):
                    features = extract_url_features(url)
                else:
                    features = filter_noise(
                        (build_feature_string(scraped) + " " + extract_url_features(url)).strip()
                    )

                if features.strip():
                    category, confidence, _ = run_prediction(features)
                    results.append({
                        "url": url, "category": category, "confidence": confidence,
                        "method": "ml_model",
                        "safe": category not in ADULT_CATEGORIES,
                        "adult_flag": category in ADULT_CATEGORIES,
                    })
                else:
                    results.append({
                        "url": url, "category": "Unknown", "confidence": 0.0,
                        "method": "no_features", "safe": None, "adult_flag": None,
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
    ensure_model_loaded()
    start = time.time()
    ip    = get_ip(request)

    try:
        url    = body.url.strip()
        if not is_valid_url(url):
            raise HTTPException(422, "Invalid URL format.")

        domain = get_domain(url)

        if domain in DOMAIN_SHORTCUTS:
            category, confidence, method = DOMAIN_SHORTCUTS[domain], 99.0, "domain_shortcut"
        else:
            try:
                scraped = scrape_website(url)
            except Exception:
                scraped = {"error": "SCRAPE_FAILED"}

            if scraped.get("error"):
                features = extract_url_features(url)
            else:
                features = filter_noise(
                    (build_feature_string(scraped) + " " + extract_url_features(url)).strip()
                )

            if not features.strip():
                raise HTTPException(422, "Could not extract features from URL.")

            category, confidence, _ = run_prediction(features)
            method = "ml_model"

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
    """
    LIME XAI explanation — shared by GET and POST /explain.
    LIME import is deferred here: lime is only loaded into memory
    when this endpoint is first called, not at startup.
    num_samples=150 (down from 200) reduces peak RAM during LIME.
    """
    ensure_model_loaded()

    # Deferred LIME import — saves ~30MB of RAM until first explain call
    from lime.lime_text import LimeTextExplainer

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
            features      = filter_noise(
                (build_feature_string(scraped) + " " + extract_url_features(url)).strip()
            )
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
            num_samples = 150,   # 150 vs 200: ~25% less RAM, still accurate
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

        # Parameterised — no SQL injection risk
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