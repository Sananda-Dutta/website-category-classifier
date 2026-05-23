# ═══════════════════════════════════════════════════════════════════════════════
# Website Category Classifier API  —  v2.0.0
# Model   : DistilBERT fine-tuned (11 categories)
# Author  : SanandaDutta
# HF Repo : SanandaDutta/website-category-distilbert
# Render  : website-category-classifier.onrender.com
#
# Endpoints (Layer 3 — Roadmap):
#   POST /classify/url     — scrape + predict, domain shortcuts
#   POST /classify/text    — raw text input, real probabilities
#   POST /classify/batch   — up to 20 URLs, CSV export built-in
#   POST /safe-check       — Adult/Kids safety flag + verdict
#   GET  /explain          — LIME word-level XAI explanation
#
# Infra (Layer 4 — Roadmap):
#   SQLite logging         — every call tracked
#   Rate limiting          — 30 req/min per IP (slowapi)
#   HuggingFace Hub        — model hosted free
#   LRU cache              — repeated URLs served instantly
#   Render + cron ping     — 24/7 uptime
# ═══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────
import os
import re
import sqlite3
import time
import torch
import numpy as np
import pandas as pd

from functools import lru_cache
from datetime import datetime
from urllib.parse import urlparse
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from slowapi.errors import RateLimitExceeded
from huggingface_hub import hf_hub_download

from rate_limiter import limiter
from scraper import scrape_website, build_feature_string

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
HF_MODEL_ID = "SanandaDutta/website-category-distilbert"
MODEL_DIR   = "./distilbert_final"       # local fallback
DB_FILE     = "usage_logs.db"           # SQLite (replaces CSV — survives restarts)

# Safety flags for /safe-check
ADULT_CATEGORIES = {"Adult"}
KIDS_CATEGORY    = "Kids"
SAFE_FOR_KIDS    = {"Education", "Kids", "Arts", "Recreation"}

# Global state
tokenizer   = None
model       = None
CLASS_NAMES = []
device      = None


# ─────────────────────────────────────────────
# SQLITE LOGGING
# Every API call is logged with full detail.
# Powers the /stats analytics endpoint.
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
    ip: str,
    endpoint: str,
    success: bool,
    time_ms: float,
    input_url:  str   = None,
    input_text: str   = None,
    category:   str   = None,
    confidence: float = None,
    method:     str   = None,
):
    """Non-crashing logger — API never fails because of a log error."""
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.execute("""
            INSERT INTO api_logs
                (timestamp, ip, endpoint, input_url, input_text,
                 category, confidence, success, time_ms, method)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            ip, endpoint,
            input_url,
            (input_text[:200] if input_text else None),  # cap stored text
            category, confidence,
            int(success), time_ms, method
        ))
        conn.commit()
        conn.close()
    except Exception:
        pass


# ─────────────────────────────────────────────
# LIFESPAN — startup + shutdown
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model, CLASS_NAMES, device

    print("=" * 60)
    print("🚀  Website Category Classifier API  v2.0.0")
    print("=" * 60)

    # Init SQLite
    init_db()
    print("✅ SQLite logging initialised →", DB_FILE)

    # Load device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📟 Device: {device}")

    # ── Load model: HuggingFace first, local fallback ──
    try:
        print(f"🌐 Loading model from HuggingFace: {HF_MODEL_ID}")
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
        model     = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_ID)
        print("✅ Model loaded from HuggingFace")

        labels_path = hf_hub_download(
            repo_id=HF_MODEL_ID,
            filename="label_classes.csv"
        )

    except Exception as hf_err:
        print(f"⚠️  HuggingFace load failed: {hf_err}")
        print(f"🔁 Falling back to local: {MODEL_DIR}")
        tokenizer   = AutoTokenizer.from_pretrained(MODEL_DIR)
        model       = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        labels_path = os.path.join(MODEL_DIR, "label_classes.csv")
        print("✅ Model loaded from local folder")

    model.to(device)
    model.eval()

    # ── Load class names ──
    try:
        labels_df   = pd.read_csv(labels_path, encoding="utf-8")
        CLASS_NAMES = [
            str(x).strip()
            for x in labels_df.iloc[:, 0].dropna().tolist()
        ]
        if not CLASS_NAMES:
            raise ValueError("label_classes.csv is empty")
        print(f"✅ Classes loaded ({len(CLASS_NAMES)}): {CLASS_NAMES}")
    except Exception as label_err:
        print(f"❌ Class name load error: {label_err}")
        CLASS_NAMES = []

    # ── Model warmup (avoids cold-start lag on first request) ──
    try:
        dummy = tokenizer(
            "warmup input", return_tensors="pt",
            truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            model(**dummy)
        print("✅ Model warmup complete")
    except Exception as warmup_err:
        print(f"⚠️  Warmup failed: {warmup_err}")

    print("=" * 60)
    print("🟢 API is ready")
    print("=" * 60)

    yield

    print("🔴 Shutting down...")


# ─────────────────────────────────────────────
# APP INIT
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
    version  = "2.0.0",
    lifespan = lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code = 429,
        content     = {
            "error":   "Rate limit exceeded.",
            "detail":  "Max 30 requests/minute per IP on classify endpoints. "
                       "Max 5/minute on batch and explain."
        }
    )


# ─────────────────────────────────────────────
# DOMAIN SHORTCUTS
# Well-known Indian + global sites → instant response, no scraping needed.
# ─────────────────────────────────────────────
DOMAIN_SHORTCUTS = {
    # ── News ──
    "ndtv.com": "News",
    "thehindu.com": "News",
    "hindustantimes.com": "News",
    "timesofindia.indiatimes.com": "News",
    "indianexpress.com": "News",
    "scroll.in": "News",
    "thewire.in": "News",
    "theprint.in": "News",
    "bbc.com": "News",
    "reuters.com": "News",
    "aajtak.in": "News",
    "zeenews.india.com": "News",
    "news18.com": "News",
    # ── Business ──
    "moneycontrol.com": "Business",
    "economictimes.indiatimes.com": "Business",
    "livemint.com": "Business",
    "business-standard.com": "Business",
    "zerodha.com": "Business",
    "groww.in": "Business",
    "amazon.in": "Business",
    "flipkart.com": "Business",
    "razorpay.com": "Business",
    "paytm.com": "Business",
    "phonepe.com": "Business",
    "indiamart.com": "Business",
    "zoho.com": "Business",
    # ── Technology ──
    "github.com": "Technology",
    "stackoverflow.com": "Technology",
    "geeksforgeeks.org": "Technology",
    "hackerrank.com": "Technology",
    "leetcode.com": "Technology",
    "codechef.com": "Technology",
    "digit.in": "Technology",
    "gadgets360.com": "Technology",
    "91mobiles.com": "Technology",
    "beebom.com": "Technology",
    # ── Education ──
    "byjus.com": "Education",
    "unacademy.com": "Education",
    "vedantu.com": "Education",
    "coursera.org": "Education",
    "khanacademy.org": "Education",
    "nptel.ac.in": "Education",
    "swayam.gov.in": "Education",
    "wikipedia.org": "Education",
    "doubtnut.com": "Education",
    "testbook.com": "Education",
    # ── Health ──
    "practo.com": "Health",
    "1mg.com": "Health",
    "netmeds.com": "Health",
    "apollohospitals.com": "Health",
    "webmd.com": "Health",
    "healthline.com": "Health",
    "pharmeasy.in": "Health",
    "cult.fit": "Health",
    # ── Gaming ──
    "dream11.com": "Gaming",
    "mpl.live": "Gaming",
    "winzo.com": "Gaming",
    "zupee.com": "Gaming",
    "rummycircle.com": "Gaming",
    "adda52.com": "Gaming",
    # ── Recreation ──
    "cricbuzz.com": "Recreation",
    "espncricinfo.com": "Recreation",
    "sportskeeda.com": "Recreation",
    "indiahikes.com": "Recreation",
    "bcci.tv": "Recreation",
    "iplt20.com": "Recreation",
    # ── Lifestyle ──
    "zomato.com": "Lifestyle",
    "swiggy.com": "Lifestyle",
    "makemytrip.com": "Lifestyle",
    "nykaa.com": "Lifestyle",
    "vogue.in": "Lifestyle",
    "femina.in": "Lifestyle",
    "mensxp.com": "Lifestyle",
    "shaadi.com": "Lifestyle",
    # ── Kids ──
    "firstcry.com": "Kids",
    "nickelodeonindia.com": "Kids",
    "tinkle.in": "Kids",
    "amarchitrakatha.com": "Kids",
    "disneyindia.in": "Kids",
    "chuChutv.com": "Kids",
    # ── Arts ──
    "gaana.com": "Arts",
    "saavn.com": "Arts",
    "filmfare.com": "Arts",
    "bollywoodhungama.com": "Arts",
    "pratilipi.com": "Arts",
    "rekhta.org": "Arts",
    "bookmyshow.com": "Arts",
}


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def get_ip(request: Request) -> str:
    """Extract real IP even behind Render's proxy."""
    forwarded = request.headers.get("x-forwarded-for")
    return forwarded.split(",")[0].strip() if forwarded else (
        request.client.host if request.client else "unknown"
    )

def get_domain(url: str) -> str:
    parsed = urlparse(url if url.startswith("http") else "https://" + url)
    return parsed.netloc.replace("www.", "").lower()

def is_valid_url(url: str) -> bool:
    pattern = re.compile(
        r'^(https?://)?'
        r'(([a-zA-Z0-9\-]+\.)+[a-zA-Z]{2,})'
        r'(/.*)?$'
    )
    return bool(re.match(pattern, url))

def extract_url_features(url: str) -> str:
    """
    Converts URL into token string matching prepare_data.py logic.
    Same feature engineering used during training.
    """
    try:
        parsed       = urlparse(url if url.startswith("http") else "http://" + url)
        domain       = parsed.netloc.replace("www.", "")
        path         = parsed.path
        tld          = domain.split(".")[-1] if "." in domain else ""
        domain_words = re.split(r'[.\-_]', domain)
        path_words   = re.split(r'[/\-_.]', path)

        tld_signal = {
            "edu": "education university college academic",
            "gov": "government official public authority",
            "org": "organization nonprofit charity",
            "ac":  "academic university college",
            "mil": "military government defense",
        }.get(tld, "")

        all_parts = domain_words * 3 + path_words + tld_signal.split()
        clean     = [w.lower() for w in all_parts if len(w) > 2 and w.isalpha()]
        return " ".join(clean)
    except Exception:
        return ""


# ─────────────────────────────────────────────
# MODEL PREDICTION
# LRU cache: repeated identical inputs skip inference entirely.
# ─────────────────────────────────────────────
@lru_cache(maxsize=1000)
def run_prediction(feature_string: str):
    """
    DistilBERT inference with softmax probabilities.
    Returns (category, confidence_%, top3_list)
    Cached for 1000 unique inputs — speeds up repeated URLs.
    """
    enc = tokenizer(
        feature_string,
        truncation    = True,
        max_length    = 512,
        padding       = True,
        return_tensors= "pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**enc).logits

    probs    = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    top3_idx = np.argsort(probs)[::-1][:3]

    top3 = [
        {
            "category":   CLASS_NAMES[i],
            "confidence": round(float(probs[i]) * 100, 1)
        }
        for i in top3_idx
    ]

    best_idx   = int(np.argmax(probs))
    category   = CLASS_NAMES[best_idx]
    confidence = round(float(probs[best_idx]) * 100, 1)

    return category, confidence, top3


def predict_proba_batch(texts: List[str]) -> np.ndarray:
    """
    Batched softmax — used internally by LIME explainer.
    Not cached (LIME generates perturbed variants each time).
    """
    all_probs = []
    for i in range(0, len(texts), 16):
        batch = texts[i : i + 16]
        enc   = tokenizer(
            batch, truncation=True, max_length=512,
            padding=True, return_tensors="pt"
        ).to(device)
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

class PredictionResult(BaseModel):
    category:   str
    confidence: float
    top3:       List[dict]
    method:     str
    time_ms:    float


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

# ── Root ──────────────────────────────────────
@app.get("/", tags=["Info"])
def home():
    return {
        "message" : "Website Category Classifier API 🚀",
        "version" : "2.0.0",
        "model"   : "DistilBERT fine-tuned",
        "classes" : CLASS_NAMES,
        "docs"    : "/docs",
        "endpoints": [
            "POST /classify/url    — scrape + predict from URL",
            "POST /classify/text   — predict from raw text",
            "POST /classify/batch  — up to 20 URLs + CSV export",
            "POST /safe-check      — Adult/Kids safety flag + verdict",
            "GET  /explain?url=    — LIME XAI word explanation",
            "GET  /stats           — API usage analytics",
            "GET  /stats/export    — download logs as CSV",
            "GET  /health          — model status",
        ]
    }


# ── Health ────────────────────────────────────
@app.get("/health", tags=["Info"])
def health():
    return {
        "status"         : "ok",
        "model_loaded"   : model is not None,
        "classes_loaded" : len(CLASS_NAMES),
        "classes"        : CLASS_NAMES,
        "device"         : str(device),
        "hf_repo"        : HF_MODEL_ID,
        "cache_info"     : run_prediction.cache_info()._asdict(),
    }


# ── 1. POST /classify/url ─────────────────────
@app.post("/classify/url", response_model=PredictionResult, tags=["Classify"])
@limiter.limit("30/minute")
async def classify_url(request: Request, body: URLRequest):
    """
    Classify a website by its URL.
    - Known domains → instant response via shortcut table
    - Unknown domains → scrape page content + extract URL features → DistilBERT
    - Scrape fails → URL features only (still works, just less accurate)
    """
    start = time.time()
    ip    = get_ip(request)

    try:
        url = body.url.strip()

        if not is_valid_url(url):
            raise HTTPException(422, "Invalid URL format.")

        domain = get_domain(url)

        # ── Fast path: known domain ──
        if domain in DOMAIN_SHORTCUTS:
            category = DOMAIN_SHORTCUTS[domain]
            elapsed  = round((time.time() - start) * 1000, 1)
            log_request(ip, "/classify/url", True, elapsed,
                        input_url=url, category=category,
                        confidence=99.0, method="domain_shortcut")
            return PredictionResult(
                category=category, confidence=99.0,
                top3=[{"category": category, "confidence": 99.0}],
                method="domain_shortcut", time_ms=elapsed
            )

        # ── Scrape + feature extraction ──
        try:
            scraped = scrape_website(url)
        except Exception:
            scraped = {"error": "SCRAPE_FAILED"}

        if scraped.get("error"):
            features = extract_url_features(url)
            method   = "url_features_only"
        else:
            content  = build_feature_string(scraped)
            features = (content + " " + extract_url_features(url)).strip()
            method   = "combined_features"

        if not features.strip():
            raise HTTPException(422, "Could not extract any features from this URL.")

        category, confidence, top3 = run_prediction(features)
        elapsed = round((time.time() - start) * 1000, 1)

        log_request(ip, "/classify/url", True, elapsed,
                    input_url=url, category=category,
                    confidence=confidence, method=method)

        return PredictionResult(
            category=category, confidence=confidence,
            top3=top3, method=method, time_ms=elapsed
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
    """
    Classify raw text — page title, meta description, scraped content, keywords.
    Min 10 chars. Text over 5000 chars is trimmed to first 5000.
    """
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
            top3=top3, method="text_input", time_ms=elapsed
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
    """
    Classify up to 20 URLs in one request.
    Returns JSON results + a CSV string ready for Excel export.
    Use case: brand safety audits, ad network screening, bulk analysis.
    """
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
                    results.append({
                        "url"       : url,
                        "category"  : DOMAIN_SHORTCUTS[domain],
                        "confidence": 99.0,
                        "method"    : "domain_shortcut",
                        "safe"      : DOMAIN_SHORTCUTS[domain] not in ADULT_CATEGORIES,
                        "adult_flag": DOMAIN_SHORTCUTS[domain] in ADULT_CATEGORIES,
                    })
                    continue

                try:
                    scraped = scrape_website(url)
                except Exception:
                    scraped = {"error": "SCRAPE_FAILED"}

                features = (
                    build_feature_string(scraped) + " " + extract_url_features(url)
                ).strip() if not scraped.get("error") else extract_url_features(url)

                if features.strip():
                    category, confidence, _ = run_prediction(features)
                    results.append({
                        "url"       : url,
                        "category"  : category,
                        "confidence": confidence,
                        "method"    : "ml_model",
                        "safe"      : category not in ADULT_CATEGORIES,
                        "adult_flag": category in ADULT_CATEGORIES,
                    })
                else:
                    results.append({
                        "url"       : url,
                        "category"  : "Unknown",
                        "confidence": 0.0,
                        "method"    : "no_features",
                        "safe"      : None,
                        "adult_flag": None,
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

        # CSV string for direct Excel paste / download
        csv_lines = ["url,category,confidence,method,safe,adult_flag"]
        for r in results:
            csv_lines.append(
                f"{r['url']},{r['category']},{r['confidence']},"
                f"{r['method']},{r['safe']},{r['adult_flag']}"
            )

        return {
            "total"      : len(results),
            "time_ms"    : elapsed,
            "results"    : results,
            "csv_export" : "\n".join(csv_lines),
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
    """
    Safety classification for:
    - Parental controls
    - Ad network brand safety
    - Firewall URL filtering

    Returns:
      safe          → True = general audience safe
      adult_flag    → True = Adult content detected (block recommended)
      kids_safe     → True = explicitly Kids category
      safe_for_kids → True = Education / Kids / Arts / Recreation
      verdict       → human-readable string with emoji
    """
    start = time.time()
    ip    = get_ip(request)

    try:
        url    = body.url.strip()

        if not is_valid_url(url):
            raise HTTPException(422, "Invalid URL format.")

        domain = get_domain(url)

        if domain in DOMAIN_SHORTCUTS:
            category   = DOMAIN_SHORTCUTS[domain]
            confidence = 99.0
            method     = "domain_shortcut"
        else:
            try:
                scraped = scrape_website(url)
            except Exception:
                scraped = {"error": "SCRAPE_FAILED"}

            features = (
                build_feature_string(scraped) + " " + extract_url_features(url)
            ).strip() if not scraped.get("error") else extract_url_features(url)

            if not features.strip():
                raise HTTPException(422, "Could not extract features from URL.")

            category, confidence, _ = run_prediction(features)
            method = "ml_model"

        adult_flag    = category in ADULT_CATEGORIES
        kids_safe     = category == KIDS_CATEGORY
        safe_for_kids = category in SAFE_FOR_KIDS
        safe          = not adult_flag

        verdict = (
            "🔴 ADULT — block recommended"  if adult_flag    else
            "🟢 KIDS SAFE"                  if kids_safe     else
            "🟡 SAFE FOR KIDS"              if safe_for_kids else
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
            "safe"         : safe,
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


# ── 5. GET /explain  (LIME XAI) ───────────────
@app.get("/explain", tags=["XAI"])
@limiter.limit("5/minute")
async def explain(
    request: Request,
    url    : str = Query(..., description="Full URL to explain"),
    n_words: int = Query(10,  description="Number of top words (max 20)")
):
    """
    Explains which words drove the DistilBERT prediction for a URL.
    Uses LIME (Local Interpretable Model-agnostic Explanations).

    Slower than other endpoints (~10–20s per call).
    Rate limited to 5/minute to protect server resources.

    Supports the GET /explain endpoint shown in your roadmap (XAI column).
    """
    start = time.time()
    ip    = get_ip(request)

    try:
        from lime.lime_text import LimeTextExplainer

        n_words = min(max(n_words, 1), 20)

        try:
            scraped = scrape_website(url)
        except Exception:
            scraped = {"error": "SCRAPE_FAILED"}

        features = (
            build_feature_string(scraped) + " " + extract_url_features(url)
        ).strip() if not scraped.get("error") else extract_url_features(url)

        if not features.strip():
            raise HTTPException(422, "Could not extract features from URL.")

        category, confidence, top3 = run_prediction(features)
        pred_idx = CLASS_NAMES.index(category)

        explainer = LimeTextExplainer(
            class_names  = CLASS_NAMES,
            bow          = False,
            random_state = 42
        )
        exp = explainer.explain_instance(
            features,
            predict_proba_batch,
            labels      = [pred_idx],
            num_features= n_words,
            num_samples = 200     # keep low for API speed; raise for accuracy
        )

        word_weights = [
            {
                "word"     : word,
                "weight"   : round(weight, 4),
                "direction": "supports" if weight > 0 else "opposes",
            }
            for word, weight in exp.as_list(label=pred_idx)
        ]

        elapsed = round((time.time() - start) * 1000, 1)
        log_request(ip, "/explain", True, elapsed,
                    input_url=url, category=category,
                    confidence=confidence, method="lime")

        return {
            "url"        : url,
            "category"   : category,
            "confidence" : confidence,
            "top3"       : top3,
            "explanation": word_weights,
            "note"       : (
                "Words with direction='supports' pushed the prediction toward "
                f"'{category}'. Words with 'opposes' pushed against it."
            ),
            "time_ms"    : elapsed,
        }

    except HTTPException:
        raise
    except Exception as e:
        log_request(ip, "/explain", False, 0, input_url=url)
        raise HTTPException(500, f"Explain error: {str(e)}")


# ── 6. GET /stats  (Analytics dashboard) ──────
@app.get("/stats", tags=["Analytics"])
async def get_stats(
    request: Request,
    limit  : int = Query(100, le=1000, description="Max recent requests to return")
):
    """
    API usage analytics — powers the Analytics Dashboard in your roadmap.
    Shows: total calls, success rate, category breakdown, per-endpoint stats.
    """
    try:
        conn = sqlite3.connect(DB_FILE)

        total         = conn.execute("SELECT COUNT(*) FROM api_logs").fetchone()[0]
        success_count = conn.execute(
            "SELECT COUNT(*) FROM api_logs WHERE success=1"
        ).fetchone()[0]

        by_endpoint = conn.execute("""
            SELECT endpoint,
                   COUNT(*)              AS calls,
                   ROUND(AVG(time_ms),1) AS avg_ms,
                   SUM(CASE WHEN success=0 THEN 1 ELSE 0 END) AS errors
            FROM api_logs
            GROUP BY endpoint
            ORDER BY calls DESC
        """).fetchall()

        by_category = conn.execute("""
            SELECT category, COUNT(*) AS count
            FROM api_logs
            WHERE category IS NOT NULL AND category != ''
            GROUP BY category
            ORDER BY count DESC
        """).fetchall()

        recent = conn.execute(f"""
            SELECT timestamp, ip, endpoint, input_url,
                   category, confidence, success, time_ms, method
            FROM api_logs
            ORDER BY id DESC
            LIMIT {limit}
        """).fetchall()

        conn.close()

        return {
            "summary": {
                "total_requests": total,
                "success_count" : success_count,
                "error_count"   : total - success_count,
                "success_rate"  : round(success_count / total * 100, 1) if total else 0,
            },
            "by_endpoint": [
                {
                    "endpoint": r[0], "calls": r[1],
                    "avg_ms": r[2], "errors": r[3]
                }
                for r in by_endpoint
            ],
            "by_category": [
                {"category": r[0], "count": r[1]}
                for r in by_category
            ],
            "recent_requests": [
                {
                    "timestamp" : r[0], "ip"        : r[1],
                    "endpoint"  : r[2], "input_url" : r[3],
                    "category"  : r[4], "confidence": r[5],
                    "success"   : bool(r[6]), "time_ms": r[7],
                    "method"    : r[8],
                }
                for r in recent
            ],
        }

    except Exception as e:
        raise HTTPException(500, f"Stats error: {str(e)}")


# ── 7. GET /stats/export  (CSV download) ──────
@app.get("/stats/export", tags=["Analytics"])
async def export_logs():
    """
    Download all API logs as a CSV file.
    Useful for the Analytics Dashboard PDF export feature in your roadmap.
    """
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
                f'"{str(x)}"' if x is not None else '""'
                for x in r
            ))

        return StreamingResponse(
            iter(["\n".join(lines)]),
            media_type = "text/csv",
            headers    = {
                "Content-Disposition": "attachment; filename=api_logs.csv"
            }
        )

    except Exception as e:
        raise HTTPException(500, f"Export error: {str(e)}")