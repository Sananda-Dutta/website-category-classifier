import requests
from bs4 import BeautifulSoup
import re
import os

# ---------------------------------------------------------------------------
# NEW: URL of your HuggingFace scraper microservice (Playwright + stealth +
# Browserless fallback). Set this as an environment variable on Render
# named SCRAPER_SERVICE_URL. The hardcoded value below is just a fallback
# default so local testing still works without setting the env var.
# ---------------------------------------------------------------------------
SCRAPER_SERVICE_URL = os.environ.get(
    "SCRAPER_SERVICE_URL",
    "https://sanandadutta-wcc-scraper.hf.space"
)

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

    # Skip known blocked sites (no point even asking the scraper service)
    if is_blocked_site(url):
        return {
            "url": url,
            "title": "", "meta_description": "",
            "meta_keywords": "", "h1": "", "h2": "", "body": "",
            "error": "BLOCKED"
        }

    # -----------------------------------------------------------------
    # CHANGED: instead of requests.get(url) directly, we POST the target
    # URL to our scraper microservice, which does the actual fetching
    # (via a real browser, so it isn't blocked by target sites the way
    # Render's own IP would be). It hands back the full raw HTML.
    # -----------------------------------------------------------------
    try:
        response = requests.post(
            f"{SCRAPER_SERVICE_URL}/scrape",
            json={"url": url},
            timeout=30  # browser rendering takes longer than a plain GET
        )
        response.raise_for_status()
        result = response.json()

        if not result.get("success"):
            return {"error": result.get("error", "Scraper service failed")}

        html_content = result.get("html", "")

        # Handle bot-blocking / empty responses (same guard as before)
        if not html_content or (len(html_content) < 800 and len(html_content) > 500):
            return {"error": "Empty or blocked response"}

    except requests.exceptions.Timeout:
        return {"error": "Timeout"}
    except requests.exceptions.ConnectionError:
        return {"error": "Connection error"}
    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP error: {e}"}
    except Exception as e:
        return {"error": f"Error: {str(e)}"}

    # -----------------------------------------------------------------
    # EVERYTHING BELOW THIS LINE IS UNCHANGED from your original file.
    # It parses html_content exactly the same way it used to parse
    # response.text — the scraper service just gets us clean, unblocked
    # HTML to hand to BeautifulSoup.
    # -----------------------------------------------------------------
    soup = BeautifulSoup(html_content, "html.parser")

    # Title extraction
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    # Body text raw capture (used for the bot check)
    body_text = soup.get_text(separator=" ", strip=True)

    # =====================================================================
    # BOT CHALLENGE INTERCEPTION
    # =====================================================================
    check_payload = f"{title} {body_text}".lower()
    bot_signatures = [
        "checking your browser",
        "enable javascript",
        "are you human",
        "access denied",
        "cloudflare",
        "captcha"
    ]
    if any(sig in check_payload for sig in bot_signatures):
        return {
            "url": url,
            "title": "", "meta_description": "",
            "meta_keywords": "", "h1": "", "h2": "", "body": "",
            "error": "BOT_CHALLENGE"
        }
    # =====================================================================

    # Meta description
    meta_desc = ""
    for attr in [{"name": "description"}, {"property": "og:description"}]:
        meta_tag = soup.find("meta", attrs=attr)
        if meta_tag and meta_tag.get("content"):
            meta_desc = meta_tag.get("content").strip()
            break

    # Meta keywords
    meta_keywords = ""
    kw_tag = soup.find("meta", attrs={"name": "keywords"})
    if kw_tag and kw_tag.get("content"):
        meta_keywords = kw_tag.get("content").strip()

    # Headings (h1, h2, h3 up to 10 tags total)
    headings_list = []
    for tag in soup.find_all(["h1", "h2", "h3"])[:10]:
        text = tag.get_text(strip=True)
        if text:
            headings_list.append(text)
    h1 = " ".join(headings_list)
    h2 = ""  # kept as empty string so it doesn't break other dictionary lookups

    # Extract paragraphs
    paragraphs_list = []
    for p in soup.find_all("p")[:20]:
        text = p.get_text(strip=True)
        if len(text.split()) >= 4:
            paragraphs_list.append(text)
    paragraphs_text = " ".join(paragraphs_list)

    # Remove noisy tags before grabbing full body dump
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    # Body text capture window (3000 chars)
    body_text = soup.get_text(separator=" ", strip=True)
    body = re.sub(r'\s+', ' ', body_text)[:3000]

    # Mix the paragraph text features directly into the body string
    combined_body = f"{paragraphs_text} {body}".strip()

    return {
        "url": url,
        "title": title,
        "meta_description": meta_desc,
        "meta_keywords": meta_keywords,
        "h1": h1,
        "h2": h2,
        "body": combined_body,
        "error": None
    }

def extract_domain(url):
    try:
        return url.split("//")[-1].split("/")[0].replace("www.", "")
    except:
        return ""


def build_feature_string(scraped):
    parts = []

    title = scraped.get("title", "").strip()
    meta_desc = scraped.get("meta_description", "").strip()
    meta_keywords = scraped.get("meta_keywords", "").strip()
    h1 = scraped.get("h1", "").strip()
    body = scraped.get("body", "").strip()

    url = scraped.get("url", "")
    domain = extract_domain(url)

    if domain:        parts.append(domain)
    if title:         parts.append(title)
    if meta_desc:     parts.append(meta_desc)
    if h1:            parts.append(h1)
    if meta_keywords: parts.append(meta_keywords)
    if body:          parts.append(body)

    features = " ".join([p for p in parts if p])
    result = re.sub(r'http\S+', '', features)
    result = re.sub(r'[^\w\s]', ' ', result)
    result = re.sub(r'\s+', ' ', result).strip().lower()

    return result[:1500]