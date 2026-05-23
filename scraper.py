import requests
from bs4 import BeautifulSoup
import re
import random

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
]

def get_headers():
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

# Sites known to block scrapers — classify by URL only
BLOCKED_SITES = [
    "amazon.com", "amazon.in", "amazon.co.uk",
    "facebook.com", "instagram.com", "twitter.com",
    "linkedin.com", "tiktok.com"
]

def is_blocked_site(url):
    return any(site in url.lower() for site in BLOCKED_SITES)


def scrape_website(url):
    if not url.startswith("http"):
        url = "https://" + url

    # Skip known blocked sites
    if is_blocked_site(url):
        return {
            "url": url,
            "title": "", "meta_description": "",
            "meta_keywords": "", "h1": "", "h2": "", "body": "",
            "error": "BLOCKED"
        }

    try:
        response = requests.get(
            url,
            headers=get_headers(),
            timeout=10,
            allow_redirects=True
        )
        response.raise_for_status()

        # 🚨 Handle bot-blocking / empty responses
        if not response.text or len(response.text) < 200:
            return {"error": "Empty or blocked response"}

    except requests.exceptions.Timeout:
        return {"error": "Timeout"}
    except requests.exceptions.ConnectionError:
        return {"error": "Connection error"}
    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP error: {e}"}
    except Exception as e:
        return {"error": f"Error: {str(e)}"}

    soup = BeautifulSoup(response.text, "lxml")

    # ✅ SAFE title extraction (fixes NoneType crash)
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    # ✅ Meta description (safe)
    meta_desc = ""
    meta_tag = soup.find("meta", attrs={"name": "description"})
    if meta_tag and meta_tag.get("content"):
        meta_desc = meta_tag.get("content").strip()

    # ✅ Meta keywords (safe)
    meta_keywords = ""
    kw_tag = soup.find("meta", attrs={"name": "keywords"})
    if kw_tag and kw_tag.get("content"):
        meta_keywords = kw_tag.get("content").strip()

    # ✅ Headings (important for classification)
    h1 = " ".join([
        h.get_text(strip=True)
        for h in soup.find_all("h1")
        if h.get_text()
    ])

    h2 = " ".join([
        h.get_text(strip=True)
        for h in soup.find_all("h2")
        if h.get_text()
    ])

    # ✅ Remove noisy tags
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    # ✅ Body text (limit size for performance)
    body_text = soup.get_text(separator=" ", strip=True)
    body_words = body_text.split()[:1000]   # 🔥 increased from 500 → 1000
    body = " ".join(body_words)

    return {
        "url": url,
        "title": title,
        "meta_description": meta_desc,
        "meta_keywords": meta_keywords,
        "h1": h1,
        "h2": h2,
        "body": body,
        "error": None
    }


def extract_domain(url):
    try:
        return url.split("//")[-1].split("/")[0].replace("www.", "")
    except:
        return ""


def build_feature_string(scraped):
    parts = []

    # ─────────────────────────────
    # 1. Core content features
    # ─────────────────────────────
    title = scraped.get("title", "")
    meta_desc = scraped.get("meta_description", "")
    meta_keywords = scraped.get("meta_keywords", "")
    h1 = scraped.get("h1", "")
    h2 = scraped.get("h2", "")
    body = scraped.get("body", "")[:1500]  # 🔥 increase context

    # ─────────────────────────────
    # 2. Domain feature (VERY IMPORTANT)
    # ─────────────────────────────
    url = scraped.get("url", "")
    domain = extract_domain(url)

    # Boost domain importance
    parts.append(domain)
    parts.append(domain)

    # ─────────────────────────────
    # 3. Weighted important features
    # ─────────────────────────────
    parts.append(title)
    parts.append(title)        # 🔥 boost title

    parts.append(meta_desc)

    parts.append(h1)
    parts.append(h1)           # 🔥 boost headings

    parts.append(h2)

    parts.append(meta_keywords)

    # Body content (less weight but important)
    parts.append(body)

    # ─────────────────────────────
    # 4. Combine safely
    # ─────────────────────────────
    features = " ".join([p for p in parts if p and p.strip()])
    features_lower = features.lower()

    # ─────────────────────────────
    # 5. Smart keyword boosting
    # ─────────────────────────────

    # 📰 News
    if any(x in features_lower for x in ["news", "breaking", "headline", "journal"]):
        features += " news news news news"

    # 🛒 E-commerce
    if any(x in features_lower for x in ["shop", "buy", "cart", "product", "sale", "order"]):
        features += " ecommerce ecommerce ecommerce ecommerce"

    # 💻 Technology
    if any(x in features_lower for x in ["tech", "software", "developer", "code", "programming", "api"]):
        features += " technology technology technology"

    # 🏥 Health
    if any(x in features_lower for x in ["health", "doctor", "medical", "clinic", "hospital"]):
        features += " health health"

    # 🎓 Education
    if any(x in features_lower for x in ["course", "learn", "education", "exam", "university"]):
        features += " education education"

    # ⚽ Sports
    if any(x in features_lower for x in ["sport", "cricket", "football", "match", "score"]):
        features += " sports sports"

    # 💼 Business
    if any(x in features_lower for x in ["business", "market", "finance", "stock", "startup"]):
        features += " business business"

    # 🏛️ Government / fallback
    if any(x in features_lower for x in ["gov", "government", "ministry", "policy"]):
        features += " other other"

    # ─────────────────────────────
    return features