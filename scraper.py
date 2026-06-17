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

        # Handle bot-blocking / empty responses
        if not response.text or (len(response.text) < 800 and len(response.text) > 500):
            return {"error": "Empty or blocked response"}

    except requests.exceptions.Timeout:
        return {"error": "Timeout"}
    except requests.exceptions.ConnectionError:
        return {"error": "Connection error"}
    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP error: {e}"}
    except Exception as e:
        return {"error": f"Error: {str(e)}"}

    soup = BeautifulSoup(response.text, "html.parser")  # Swapped to html.parser to match training exactly

    # Title extraction
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    
    # Body text raw capture (We fetch this up here now so we can use it for the bot check)
    body_text = soup.get_text(separator=" ", strip=True)

    # =====================================================================
    # 🔥 NEW CODE ADDEED HERE: BOT CHALLENGE INTERCEPTION
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

    # Headings (Updated to look for h1, h2, h3 up to 10 tags total)
    headings_list = []
    for tag in soup.find_all(["h1", "h2", "h3"])[:10]:
        text = tag.get_text(strip=True)
        if text:
            headings_list.append(text)
    h1 = " ".join(headings_list)
    h2 = ""  # Keeping h2 as empty string so it doesn't break other dictionary lookups

    # --- NEW REQUIRED THING: Extract Paragraphs (<p>) ---
    paragraphs_list = []
    for p in soup.find_all("p")[:20]:
        text = p.get_text(strip=True)
        if len(text.split()) >= 4:
            paragraphs_list.append(text)
    paragraphs_text = " ".join(paragraphs_list)

    # Remove noisy tags before grabing full body dump
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    # Body text capture window adjusted to 3000 chars
    body_text = soup.get_text(separator=" ", strip=True)
    body = re.sub(r'\s+', ' ', body_text)[:3000]

    # Mix the new paragraph text features directly into the body string for safety
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

    # Pull core features
    title = scraped.get("title", "").strip()
    meta_desc = scraped.get("meta_description", "").strip()
    meta_keywords = scraped.get("meta_keywords", "").strip()
    h1 = scraped.get("h1", "").strip()
    body = scraped.get("body", "").strip()

    # Extract Domain string
    url = scraped.get("url", "")
    domain = extract_domain(url)

    # Stack things up EXACTLY ONE TIME (No multiplication list hacks, no artificial boosts)
    if domain:        parts.append(domain)
    if title:         parts.append(title)
    if meta_desc:     parts.append(meta_desc)
    if h1:            parts.append(h1)
    if meta_keywords: parts.append(meta_keywords)
    if body:          parts.append(body)

    # Standard clean up sequence
    features = " ".join([p for p in parts if p])
    result = re.sub(r'http\S+', '', features)
    result = re.sub(r'[^\w\s]', ' ', result)
    result = re.sub(r'\s+', ' ', result).strip().lower()

    # Enforce the final 1500 character window slice cap
    return result[:1500]