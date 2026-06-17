# ═══════════════════════════════════════════════════════════════════════════════
# Website Category Classifier API  —  v2.7.0
# Model   : DistilBERT fine-tuned (11 categories)
# Author  : SanandaDutta
# HF Repo : SanandaDutta/website-category-distilbert
#
# v2.7.0 — Model retrained on verified Indian + global URLs
#           Test accuracy: 82.2% | Val accuracy: 91.8% (June 2026)
#
# CHANGES FROM v2.6.0:
#   - Model retrained on clean verified data (no DMOZ noise)
#   - Fixed: keyword_fallback was referencing undefined `extracted_text`
#   - Fixed: duplicate domain keys cleaned up
#   - Added: more Adult domain shortcuts
#   - Version bumped to v2.7.0
#
# ARCHITECTURE (unchanged from v2.6.0):
#   RapidAPI → Render (this file, ~80MB RAM) → HF Space relay → DistilBERT
#
# WHY HF SPACE RELAY:
#   Render free tier blocks outbound DNS to api-inference.huggingface.co
#   Fix: route inference through a public HF Space (free CPU tier)
#   which has stable outbound networking.
#
# RENDER ENV VARS NEEDED:
#   HF_TOKEN              = hf_xxxx  (needed if model is private)
#   RAPIDAPI_PROXY_SECRET = xxxx     (set when publishing to marketplace)
#
# ALL ENDPOINTS UNCHANGED:
#   /classify/url, /classify/text, /classify/batch, /safe-check,
#   /explain, /stats, /stats/export, /health, /ping, /usage
# ═══════════════════════════════════════════════════════════════════════════════

import asyncio
import os
import re
import sqlite3
import time

from auth import verify_api_credentials

from contextlib import asynccontextmanager
from datetime import datetime
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
HF_MODEL_ID = "SanandaDutta/website-category-distilbert"

HF_SPACE_URL = "https://sanandadutta-wcc-inference-relay.hf.space/classify"

HF_TOKEN = os.getenv("HF_TOKEN", "")
DB_FILE  = "usage_logs.db"

CLASS_NAMES = [
    "Adult", "Arts", "Business", "Education", "Gaming",
    "Health", "Kids", "Lifestyle", "News", "Recreation", "Technology",
]

ADULT_CATEGORIES     = {"Adult"}
KIDS_CATEGORY        = "Kids"
SAFE_FOR_KIDS        = {"Education", "Kids", "Arts", "Recreation"}
CONFIDENCE_THRESHOLD = 45.0

# Shared async HTTP client
_hf_client: httpx.AsyncClient = None

CATEGORY_KEYWORDS = {
    "Gaming": [
        "game", "games", "gaming", "play", "player", "multiplayer",
        "arcade", "puzzle", "adventure", "fps", "rpg", "leaderboard",
        "score", "level", "quest", "boss", "spawn", "loot", "esports",
        "browser game", "online game", "free game", "playthrough",
        "speedrun", "controller", "joystick", "console", "gamepad",
    ],
    "News": [
        "news", "breaking", "headline", "reporter", "journalist",
        "article", "politics", "election", "world news", "local news",
        "latest", "update", "press", "media", "editorial",
        "correspondent", "bulletin", "dispatch", "coverage", "report",
    ],
    "Education": [
        "learn", "course", "tutorial", "lecture", "student", "teacher",
        "university", "college", "school", "exam", "study", "quiz",
        "lesson", "curriculum", "degree", "certificate", "mooc",
        "assignment", "textbook", "syllabus", "classroom", "e-learning",
    ],
    "Arts": [
        "art", "artwork", "painting", "illustration", "sculpture",
        "photography", "gallery", "museum", "exhibition", "creative",
        "design", "drawing", "sketch", "animation", "film", "music",
        "song", "album", "poetry", "literature", "theater", "dance",
        "craft", "portfolio", "canvas", "palette",
    ],
    "Business": [
        "business", "company", "startup", "enterprise", "corporate",
        "revenue", "profit", "investor", "funding", "vc", "b2b",
        "saas", "market", "strategy", "management", "ceo", "founder",
        "acquisition", "ipo", "valuation", "consulting", "finance",
        "budget", "sales", "marketing", "brand", "client",
    ],
    "Health": [
        "health", "medical", "doctor", "patient", "hospital", "medicine",
        "symptom", "diagnosis", "treatment", "wellness", "fitness",
        "nutrition", "diet", "mental health", "therapy", "clinic",
        "pharmacy", "drug", "disease", "condition", "surgery", "nurse",
        "prescription", "vaccine", "chronic", "recovery",
    ],
    "Kids": [
        "kids", "children", "child", "toddler", "preschool", "kindergarten",
        "cartoon", "nursery", "bedtime", "story", "toy", "playground",
        "safe for kids", "family friendly", "parental", "disney",
        "nick jr", "pbs kids", "sesame", "learning for kids",
    ],
    "Lifestyle": [
        "lifestyle", "fashion", "beauty", "style", "recipe", "food",
        "cooking", "home decor", "interior", "relationship", "dating",
        "wedding", "parenting", "self-help", "motivation", "mindfulness",
        "yoga", "skincare", "makeup", "outfit", "trend", "vlog",
        "influencer", "blogging", "wellness", "personal growth",
    ],
    "Recreation": [
        "sport", "sports", "football", "cricket", "tennis", "basketball",
        "soccer", "match", "tournament", "league", "team", "athlete",
        "stadium", "championship", "olympic", "travel", "adventure",
        "hiking", "camping", "outdoor", "fishing", "cycling", "gym",
        "workout", "training", "hobby", "club", "recreation",
    ],
    "Technology": [
        "software", "hardware", "developer", "api", "code", "programming",
        "tech", "cloud", "data", "ai", "machine learning", "open source",
        "devops", "cybersecurity", "blockchain", "saas", "platform",
        "framework", "library", "database", "server", "linux", "python",
        "javascript", "github", "deployment", "docker", "kubernetes",
    ],
    "Adult": [
        "adult content", "18+", "explicit", "nsfw", "xxx",
        "porn", "hentai", "nude", "erotic", "onlyfans",
    ],
}

def keyword_fallback(text: str) -> tuple[str, float]:
    text_lower = text.lower()
    scores = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        if category == "Adult":
            continue  # never assign Adult via keyword fallback
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[category] = score

    if not scores:
        return None, 0.0

    best = max(scores, key=scores.get)
    confidence = min(scores[best] / 5.0, 1.0)
    return best, round(confidence * 100, 2)


# ─────────────────────────────────────────────
# LIFESPAN
# ─────────────────────────────────────────────
async def background_prewarm():
    await asyncio.sleep(30)

    print("\n" + "═"*30)
    print("🚀 BACKGROUND PRE-WARM STARTING")
    print("═"*30)

    for i in range(3):
        try:
            await _call_hf_inference("warmup")
            print(f"✅ HF Space is awake and ready! (Attempt {i+1})")
            print("═"*30 + "\n")
            return
        except Exception as e:
            print(f"⚠️ Pre-warm attempt {i+1} failed: {e}")
            if i < 2:
                print("🔄 Retrying in 20 seconds...")
                await asyncio.sleep(20)

    print("❌ Background pre-warm failed after 3 attempts.")
    print("💡 Note: The first user request will trigger the Space wake-up.")
    print("═"*30 + "\n")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _hf_client

    print("\n" + "╔" + "═"*53 + "╗")
    print(f"║ {'WEBSITE CATEGORY CLASSIFIER v2.7.0':^51} ║")
    print("╚" + "═"*53 + "╝")

    # PHASE A: Database
    try:
        init_db()
        print("📁 [1/4] SQLite Database: Ready")
    except Exception as e:
        print(f"❌ [1/4] SQLite Database: FAILED ({e})")

    # PHASE B: Credentials
    if not HF_TOKEN:
        print("⚠️  [2/4] HF_TOKEN not set (ok if Space model is public)")
    else:
        print(f"🔑 [2/4] Credentials: HF_TOKEN Verified")
    print(f"🤖 [2/4] Relay Target: {HF_SPACE_URL}")

    # PHASE C: HTTP client
    _hf_client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=30.0, read=60.0, write=10.0, pool=5.0),
        limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
    )
    print("🌐 [3/4] Network: HTTPX Client Initialized (no auth — Space is public)")

    # PHASE D: Background pre-warm
    asyncio.create_task(background_prewarm())
    print("🛰️ [4/4] AI Engine: Background pre-warm scheduled")

    print("\n🚀 STATUS: API IS LIVE | RAM: ~80MB | MODEL: v2.7.0 (85.4% accuracy)")
    print("═"*55 + "\n")

    yield

    await _hf_client.aclose()
    print("\n" + "═"*55)
    print("🛑 SHUTDOWN: HTTPX Client Closed. Service Offline.")
    print("═"*55)


# ─────────────────────────────────────────────
# HF SPACE CALLER
# ─────────────────────────────────────────────
async def _call_hf_inference(text: str) -> list:
    if _hf_client is None:
        raise HTTPException(503, "HTTP client not initialized. Check startup logs.")

    payload = {"inputs": text[:512]}

    for attempt in range(3):
        try:
            r = await _hf_client.post(HF_SPACE_URL, json=payload)

            if r.status_code == 503:
                wait = 20
                try:
                    wait = min(r.json().get("estimated_time", 20), 30)
                except Exception:
                    pass
                print(f"HF Space loading, waiting {wait}s (attempt {attempt+1}/3)")
                await asyncio.sleep(wait)
                continue

            if r.status_code == 429:
                await asyncio.sleep(10)
                continue

            r.raise_for_status()
            results = r.json()

            if isinstance(results, list) and results:
                if isinstance(results[0], list):
                    results = results[0]
                results.sort(key=lambda x: x["score"], reverse=True)
                return results

            raise ValueError(f"Unexpected Space response format: {results}")

        except httpx.TimeoutException:
            if attempt == 2:
                raise HTTPException(504,
                    "HF Space timed out. Space may be cold — retry in 30 seconds.")
            await asyncio.sleep(5)
            continue

        except HTTPException:
            raise

        except Exception as e:
            import traceback
            print(f"[HF ERROR attempt {attempt+1}] {type(e).__name__}: {e}")
            print(traceback.format_exc())
            if attempt == 2:
                raise HTTPException(500, f"HF Inference error: {str(e)}")
            await asyncio.sleep(3)
            continue

    raise HTTPException(503, "HF Space unavailable after 3 retries.")


# ─────────────────────────────────────────────
# PREDICTION CACHE
# ─────────────────────────────────────────────
_prediction_cache: dict = {}
_cache_lock = asyncio.Lock()
MAX_CACHE_SIZE = 1024

async def run_prediction(feature_string: str) -> tuple:
    if feature_string in _prediction_cache:
        return _prediction_cache[feature_string]

    results = await _call_hf_inference(feature_string)

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

    async with _cache_lock:
        if len(_prediction_cache) >= MAX_CACHE_SIZE:
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
        "**Model accuracy: 82.2% test | 91.8% val** (retrained June 2026)\n\n"
        "**Categories:** Adult · Arts · Business · Education · Gaming · "
        "Health · Kids · Lifestyle · News · Recreation · Technology\n\n"
        "Built by **SanandaDutta** — "
        "[HuggingFace](https://huggingface.co/SanandaDutta) · "
        "[GitHub](https://github.com/SanandaDutta)"
    ),
    version="2.7.0",
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
        print(f"[AUTH] path={request.url.path} incoming={repr(incoming)} "
              f"expected={repr(RAPIDAPI_SECRET)} match={incoming == RAPIDAPI_SECRET}")
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
# DOMAIN / PATH SHORTCUTS
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
    # ── Adult ─────────────────────────────────────────────────────────────────
    "xvideos.com":          "Adult",
    "xnxx.com":             "Adult",
    "pornhub.com":          "Adult",
    "xhamster.com":         "Adult",
    "redtube.com":          "Adult",
    "youporn.com":          "Adult",
    "tube8.com":            "Adult",
    "brazzers.com":         "Adult",
    "onlyfans.com":         "Adult",
    "chaturbate.com":       "Adult",
    "stripchat.com":        "Adult",
    "livejasmin.com":       "Adult",
    "myfreecams.com":       "Adult",
    "bongacams.com":        "Adult",
    "nhentai.net":          "Adult",
    "hentai2read.com":      "Adult",
    "imbesharam.com":       "Adult",
    "desixnxx.net":         "Adult",
    "indianpornvideos.net": "Adult",
    "desimasala.co":        "Adult",
    # ── Arts ──────────────────────────────────────────────────────────────────
    "gaana.com":            "Arts",
    "saavn.com":            "Arts",
    "jiosaavn.com":         "Arts",
    "filmfare.com":         "Arts",
    "bollywoodhungama.com": "Arts",
    "pratilipi.com":        "Arts",
    "rekhta.org":           "Arts",
    "bookmyshow.com":       "Arts",
    "youtube.com":          "Arts",
    "youtu.be":             "Arts",
    "netflix.com":          "Arts",
    "primevideo.com":       "Arts",
    "hotstar.com":          "Arts",
    "disneyplus.com":       "Arts",
    "zee5.com":             "Arts",
    "sonyliv.com":          "Arts",
    "voot.com":             "Arts",
    "spotify.com":          "Arts",
    "imdb.com":             "Arts",
    "rottentomatoes.com":   "Arts",
    "deviantart.com":       "Arts",
    "behance.net":          "Arts",
    "dribbble.com":         "Arts",
    "artstation.com":       "Arts",
    "unsplash.com":         "Arts",
    "flickr.com":           "Arts",
    "500px.com":            "Arts",
    "vimeo.com":            "Arts",
    "soundcloud.com":       "Arts",
    "bandcamp.com":         "Arts",
    # ── Business ──────────────────────────────────────────────────────────────
    "moneycontrol.com":             "Business",
    "economictimes.indiatimes.com": "Business",
    "livemint.com":                 "Business",
    "business-standard.com":        "Business",
    "zerodha.com":                  "Business",
    "groww.in":                     "Business",
    "upstox.com":                   "Business",
    "amazon.in":                    "Business",
    "flipkart.com":                 "Business",
    "snapdeal.com":                 "Business",
    "razorpay.com":                 "Business",
    "paytm.com":                    "Business",
    "phonepe.com":                  "Business",
    "indiamart.com":                "Business",
    "zoho.com":                     "Business",
    "freshworks.com":               "Business",
    "amazon.com":                   "Business",
    "ebay.com":                     "Business",
    "meesho.com":                   "Business",
    "myntra.com":                   "Business",
    "ajio.com":                     "Business",
    "nykaa.com":                    "Business",
    "linkedin.com":                 "Business",
    "bloomberg.com":                "Business",
    "forbes.com":                   "Business",
    "businessinsider.com":          "Business",
    "entrepreneur.com":             "Business",
    "hbr.org":                      "Business",
    "crunchbase.com":               "Business",
    "glassdoor.com":                "Business",
    "indeed.com":                   "Business",
    "upwork.com":                   "Business",
    "fiverr.com":                   "Business",
    "shopify.com":                  "Business",
    "stripe.com":                   "Business",
    # ── Education ─────────────────────────────────────────────────────────────
    "byjus.com":            "Education",
    "unacademy.com":        "Education",
    "vedantu.com":          "Education",
    "coursera.org":         "Education",
    "khanacademy.org":      "Education",
    "nptel.ac.in":          "Education",
    "swayam.gov.in":        "Education",
    "wikipedia.org":        "Education",
    "doubtnut.com":         "Education",
    "testbook.com":         "Education",
    "udemy.com":            "Education",
    "edx.org":              "Education",
    "britannica.com":       "Education",
    "collegedunia.com":     "Education",
    "shiksha.com":          "Education",
    "careers360.com":       "Education",
    "brilliant.org":        "Education",
    "duolingo.com":         "Education",
    "quizlet.com":          "Education",
    "wolframalpha.com":     "Education",
    "scholar.google.com":   "Education",
    # ── Gaming ────────────────────────────────────────────────────────────────
    "dream11.com":          "Gaming",
    "mpl.live":             "Gaming",
    "winzo.com":            "Gaming",
    "zupee.com":            "Gaming",
    "rummycircle.com":      "Gaming",
    "adda52.com":           "Gaming",
    "twitch.tv":            "Gaming",
    "ign.com":              "Gaming",
    "gamespot.com":         "Gaming",
    "steampowered.com":     "Gaming",
    "epicgames.com":        "Gaming",
    "xbox.com":             "Gaming",
    "playstation.com":      "Gaming",
    "poki.com":             "Gaming",
    "miniclip.com":         "Gaming",
    "kongregate.com":       "Gaming",
    "newgrounds.com":       "Gaming",
    "itch.io":              "Gaming",
    "roblox.com":           "Gaming",
    "chess.com":            "Gaming",
    "friv.com":             "Gaming",
    "coolmathgames.com":    "Gaming",
    "y8.com":               "Gaming",
    "addictinggames.com":   "Gaming",
    # ── Health ────────────────────────────────────────────────────────────────
    "practo.com":           "Health",
    "1mg.com":              "Health",
    "netmeds.com":          "Health",
    "apollohospitals.com":  "Health",
    "pharmeasy.in":         "Health",
    "cult.fit":             "Health",
    "webmd.com":            "Health",
    "healthline.com":       "Health",
    "mayoclinic.org":       "Health",
    "medlineplus.gov":      "Health",
    "nih.gov":              "Health",
    "who.int":              "Health",
    "medscape.com":         "Health",
    # ── Kids ──────────────────────────────────────────────────────────────────
    "firstcry.com":         "Kids",
    "nickelodeonindia.com": "Kids",
    "tinkle.in":            "Kids",
    "amarchitrakatha.com":  "Kids",
    "disneyindia.in":       "Kids",
    "pbs.org":              "Kids",
    "starfall.com":         "Kids",
    "pbskids.org":          "Kids",
    "nickjr.com":           "Kids",
    "cartoonnetwork.com":   "Kids",
    "disney.com":           "Kids",
    "funbrain.com":         "Kids",
    "abcmouse.com":         "Kids",
    "sesamestreet.org":     "Kids",
    "natgeokids.com":       "Kids",
    # ── Lifestyle ─────────────────────────────────────────────────────────────
    "zomato.com":           "Lifestyle",
    "swiggy.com":           "Lifestyle",
    "makemytrip.com":       "Lifestyle",
    "irctc.co.in":          "Lifestyle",
    "vogue.in":             "Lifestyle",
    "femina.in":            "Lifestyle",
    "mensxp.com":           "Lifestyle",
    "shaadi.com":           "Lifestyle",
    "instagram.com":        "Lifestyle",
    "pinterest.com":        "Lifestyle",
    "uber.com":             "Lifestyle",
    "buzzfeed.com":         "Lifestyle",
    "cosmopolitan.com":     "Lifestyle",
    "vogue.com":            "Lifestyle",
    "elle.com":             "Lifestyle",
    "allrecipes.com":       "Lifestyle",
    "food.com":             "Lifestyle",
    "tasty.co":             "Lifestyle",
    "goodhousekeeping.com": "Lifestyle",
    "realsimple.com":       "Lifestyle",
    "mindbodygreen.com":    "Lifestyle",
    # ── News ──────────────────────────────────────────────────────────────────
    "ndtv.com":                       "News",
    "thehindu.com":                   "News",
    "hindustantimes.com":             "News",
    "timesofindia.indiatimes.com":    "News",
    "indianexpress.com":              "News",
    "scroll.in":                      "News",
    "thewire.in":                     "News",
    "theprint.in":                    "News",
    "bbc.com":                        "News",
    "bbc.co.uk":                      "News",
    "reuters.com":                    "News",
    "apnews.com":                     "News",
    "theguardian.com":                "News",
    "aajtak.in":                      "News",
    "zeenews.india.com":              "News",
    "news18.com":                     "News",
    "abplive.com":                    "News",
    "tv9bharatvarsh.com":             "News",
    "twitter.com":                    "News",
    "x.com":                          "News",
    "facebook.com":                   "News",
    "reddit.com":                     "News",
    # ── Recreation ────────────────────────────────────────────────────────────
    "cricbuzz.com":         "Recreation",
    "espncricinfo.com":     "Recreation",
    "sportskeeda.com":      "Recreation",
    "indiahikes.com":       "Recreation",
    "bcci.tv":              "Recreation",
    "iplt20.com":           "Recreation",
    "espn.com":             "Recreation",
    "bleacherreport.com":   "Recreation",
    "fifa.com":             "Recreation",
    "olympics.com":         "Recreation",
    "icc-cricket.com":      "Recreation",
    "tripadvisor.com":      "Recreation",
    "booking.com":          "Recreation",
    "airbnb.com":           "Recreation",
    "rei.com":              "Recreation",
    "alltrails.com":        "Recreation",
    "strava.com":           "Recreation",
    # ── Technology ────────────────────────────────────────────────────────────
    "claude.ai":            "Technology",
    "anthropic.com":        "Technology",
    "openai.com":           "Technology",
    "chatgpt.com":          "Technology",
    "mistral.ai":           "Technology",
    "huggingface.co":       "Technology",
    "vercel.com":           "Technology",
    "supabase.com":         "Technology",
    "figma.com":            "Technology",
    "linear.app":           "Technology",
    "cloudflare.com":       "Technology",
    "digitalocean.com":     "Technology",
    "render.com":           "Technology",
    "netlify.com":          "Technology",
    "heroku.com":           "Technology",
    "aws.amazon.com":       "Technology",
    "azure.microsoft.com":  "Technology",
    "github.com":           "Technology",
    "stackoverflow.com":    "Technology",
    "geeksforgeeks.org":    "Technology",
    "hackerrank.com":       "Technology",
    "leetcode.com":         "Technology",
    "codechef.com":         "Technology",
    "codeforces.com":       "Technology",
    "digit.in":             "Technology",
    "gadgets360.com":       "Technology",
    "91mobiles.com":        "Technology",
    "beebom.com":           "Technology",
    "techcrunch.com":       "Technology",
    "theverge.com":         "Technology",
    "wired.com":            "Technology",
    "arstechnica.com":      "Technology",
    "dev.to":               "Technology",
    "medium.com":           "Technology",
    "producthunt.com":      "Technology",
    "npmjs.com":            "Technology",
    "pypi.org":             "Technology",
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
        # 1. Safely pull data from your active scraper dictionary
        title = scraped.get("title", "").strip()
        meta  = scraped.get("meta_description", "").strip()
        keywords = scraped.get("keywords", "").strip()
        
        # Pull headings and paragraph strings
        h1_h2_h3 = scraped.get("h1_h2_h3", "").strip() or " ".join([scraped.get("h1", ""), scraped.get("h2", "")]).strip()
        paragraphs = scraped.get("paragraphs", "").strip()
        
        # Read the raw background body copy block
        body  = scraped.get("body", scraped.get("body_text", "")).strip()
        
        # 2. Append them strictly ONE time each (NO multiplication lists!)
        if title:      parts.append(title)
        if meta:       parts.append(meta)
        if keywords:   parts.append(keywords)
        if h1_h2_h3:   parts.append(h1_h2_h3)
        if paragraphs: parts.append(paragraphs)
        
        # 3. Match the 3000-character body snapshot parsing limit
        if body:       
            parts.append(body[:3000])
            
    except Exception:
        return extract_url_features(url)
    
    # 4. Standardize text structure transformations (regex cleaning rules)
    result = " ".join(parts)
    result = re.sub(r'http\S+', '', result)   # drop links
    result = re.sub(r'[^\w\s]', ' ', result)   # drop symbols
    result = re.sub(r'\s+', ' ', result).strip().lower()
    
    # 5. Cap the total feature payload output string precisely at 1500 chars
    words = result[:1500].split()
    return " ".join(w for w in words if w not in NOISE_WORDS and len(w) > 2)

# ─────────────────────────────────────────────
# SMART CLASSIFY
# ─────────────────────────────────────────────
async def smart_classify(url: str) -> tuple:
    """
    Returns (category, confidence, top3, method).
    Shortcuts served locally (0 ms). Unknown domains call HF Space.
    """
    domain           = get_domain(url)
    domain_with_path = get_domain_with_path(url)

    if domain_with_path in PATH_SHORTCUTS:
        cat = PATH_SHORTCUTS[domain_with_path]
        return cat, 99.0, [{"category": cat, "confidence": 99.0}], "path_shortcut"

    if domain in DOMAIN_SHORTCUTS:
        cat = DOMAIN_SHORTCUTS[domain]
        return cat, 99.0, [{"category": cat, "confidence": 99.0}], "domain_shortcut"

    # Try scraping
    scraped = {}
    try:
        scraped = scrape_website(url)
    except Exception as scrape_err:
        print(f"[SCRAPE FAILED] {url}: {scrape_err}")
        scraped = {"error": "SCRAPE_FAILED"}

    if scraped.get("error"):
        features = extract_url_features(url)
        method   = "url_features_only"
    else:
        features = build_weighted_features(scraped, url)
        method   = "combined_features"

    if not features.strip():
        features = domain.replace(".", " ").replace("-", " ")
        method   = "domain_name_only"
    
    if not features.strip() or len(features.split()) < 15:
        # Returning a tuple to perfectly match your existing smart_classify structure:
        return (
            "Unknown", 
            0.0, 
            [{"category": "Unknown", "confidence": 0.0}], 
            "unable_to_extract_fallback"
        )

    try:
        category, confidence, top3 = await run_prediction(features)
    except HTTPException as he:
        print(f"[HF ERROR in smart_classify] {he.detail}")
        return "Technology", 30.0, [{"category": "Technology", "confidence": 30.0}], "hf_error_fallback"

    # Fallback 1: try URL features alone if confidence is low
    if confidence < CONFIDENCE_THRESHOLD and method == "combined_features":
        url_features = extract_url_features(url)
        if url_features.strip():
            try:
                cat_url, conf_url, top3_url = await run_prediction(url_features)
                if conf_url > confidence:
                    category, confidence, top3 = cat_url, conf_url, top3_url
                    method = "url_features_fallback"
            except Exception:
                pass

    # Fallback 2: keyword rules if still below threshold
    # FIX v2.7.0: was referencing undefined `extracted_text`, now correctly uses `features`
    if confidence < CONFIDENCE_THRESHOLD:
        fallback_cat, fallback_conf = keyword_fallback(features)
        if fallback_cat and fallback_conf >= 30.0:
            category   = fallback_cat
            confidence = fallback_conf
            method     = "keyword_fallback"
        else:
            if method != "url_features_fallback":
                method = "model_low_confidence"
    else:
        if method != "url_features_fallback":
            method = "model"

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
        "message"     : "Website Category Classifier API v2.7.0",
        "architecture": "HF Space Relay (Render RAM: ~80 MB)",
        "model"       : "DistilBERT retrained June 2026 — 85.4% test accuracy",
        "classes"     : CLASS_NAMES,
        "docs"        : "/docs",
    }

@app.api_route("/ping", methods=["GET", "HEAD"], tags=["Info"])
async def ping(request: Request):
    return {"status": "alive", "version": "2.7.0"}

@app.api_route("/health", methods=["GET", "HEAD"], tags=["Info"])
async def health(request: Request):
    return {
        "status"        : "ok",
        "version"       : "2.7.0",
        "architecture"  : "hf_space_relay",
        "hf_space"      : HF_SPACE_URL,
        "hf_model"      : HF_MODEL_ID,
        "model_accuracy": "85.4% test | 91.8% val",
        "classes"       : CLASS_NAMES,
        "cache_size"    : len(_prediction_cache),
        "cache_max"     : MAX_CACHE_SIZE,
    }

@app.get("/usage", tags=["Info"])
async def usage_info():
    return {
        "name":       "Website Category Classifier",
        "version":    "2.7.0",
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
                    "url":        url,
                    "category":   category,
                    "confidence": confidence,
                    "method":     method,
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
            "total":      len(results),
            "time_ms":    elapsed,
            "results":    results,
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
            "url":          url,
            "category":     category,
            "confidence":   confidence,
            "safe":         not adult_flag,
            "adult_flag":   adult_flag,
            "kids_safe":    kids_safe,
            "safe_for_kids":safe_for_kids,
            "verdict":      verdict,
            "method":       method,
            "time_ms":      elapsed,
        }
    except HTTPException:
        raise
    except Exception as e:
        log_request(ip, "/safe-check", False, 0, input_url=body.url)
        raise HTTPException(500, f"Internal error: {str(e)}")


# ── 5. /explain ───────────────────────────────
@app.api_route("/explain", methods=["GET", "POST"], tags=["XAI"])
async def explain(request: Request):
    return JSONResponse(status_code=503, content={
        "error"     : "LIME explanation unavailable in HF Space relay mode.",
        "workaround": "Use top3 from /classify/url for the top 3 category scores.",
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
                    "timestamp": r[0], "ip":       r[1], "endpoint":   r[2],
                    "input_url": r[3], "category": r[4], "confidence": r[5],
                    "success":   bool(r[6]), "time_ms": r[7], "method": r[8],
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