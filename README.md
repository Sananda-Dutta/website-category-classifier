# 🌐 Website Category Classifier (V2 Production-Ready)

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![Google Cloud Run](https://img.shields.io/badge/Google_Cloud_Run-4285F4?style=for-the-badge&logo=googlecloud&logoColor=white)](https://cloud.google.com/run)
[![Render](https://img.shields.io/badge/Render-46E3B7?style=for-the-badge&logo=render&logoColor=black)](https://render.com)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Playwright](https://img.shields.io/badge/Playwright-45BA4B?style=for-the-badge&logo=playwright&logoColor=white)](https://playwright.dev/)
[![Browserless](https://img.shields.io/badge/Browserless-000000?style=for-the-badge&logo=googlechrome&logoColor=white)](https://www.browserless.io/)
[![RapidAPI](https://img.shields.io/badge/RapidAPI-0055DA?style=for-the-badge&logo=rapid&logoColor=white)](https://rapidapi.com/)
[![UptimeRobot](https://img.shields.io/badge/UptimeRobot-5CDD8B?style=for-the-badge&logo=uptimerobot&logoColor=white)](https://uptimerobot.com/)

Classify any website into one of 11 distinct verticals in real time. A production-grade, high-performance engine built for **ad-tech brand safety**, **contextual targeting**, and **EdTech content filtering** use cases. 

Give it a URL. Get back a category, a confidence score, and an immediate safety verdict—no expensive scraping infrastructure needed on your end.

🚀 **Live Commercial Gateway:** [Available on RapidAPI](https://rapidapi.com/SanandaDutta/api/website-category-classifier)

---

## 🏗️ Architecture Blueprint

The application is decoupled into a resilient, high-concurrency microservice cluster to guarantee maximum uptime, fast ingestion speeds, and strict time budgeting:
```text
        Client Request
              │
              ▼
  RapidAPI Gateway (Auth + Rate Limiting)
              │
              ▼
  Render Gateway (FastAPI — Routing, Caching, Fallback Core)
              │
              ├──▶ Scraper Service (Railway Cluster)
              │      Playwright + Stealth-Mode Headless Browser
              │      Automatic fallback to Browserless.io for hard-to-scrape sites
              │
              └──▶ Classifier Service (Railway Cluster)
                     Fine-tuned DistilBERT In-Memory Weights (11 Verticals)
```

> **Why a dedicated scraper service?** Naive server-side scraping gets IP-blocked by modern sites almost immediately. Routing scrape requests through a real, stealth-configured headless browser (with paid cloud fallbacks for hardened targets) means the classifier actually evaluates real page context instead of parsing empty strings or anti-bot challenge walls.

---

## ⚡ Key Improvements in Version 2

*   **Synchronized Training & Inference Pipeline:** Eliminated historical train-inference skew by implementing a unified flat text architecture across both training dataset compilation (`phase3_scrape_master_v2_final.py`) and live production endpoints.
*   **Pristine Natural Text Alignment:** Stripped away legacy V1 artificial keyword multipliers and tail-end string injection hacks, forcing the transformer model to rely strictly on natural linguistic distributions.
*   **Granular Context Capture:** Added automated paragraph extraction structures (`<p>` tag sampling) and widened the token intake window up to 3000 characters to retain high-density business context.

---

## ✨ Features

*   **Multi-Layer Smart Classification Engine (`smart_classify`):**
    *   *Instant Routing (0ms):* Automatically checks local deterministic paths and domain-level short-circuit lists first.
    *   *Real-time Asynchronous Scraper:* Fetches titles, meta headers, semantic headings ($H_1, H_2, H_3$), and paragraph body copy concurrently.
*   **Dual Fallback Loops:** Automatically falls back to localized URL token features if a domain blocks scraping traffic or if the model records low-confidence thresholds.
*   **Production-Grade Text Sanitization:** Active regex pipelines isolate clean web features, eliminate structural punctuation, and filter out domain-level semantic noise words.
*   **High-Concurrency Scraping Architecture:** Features random rotational User-Agent headers and native request tracking logic to bypass anti-scraping flags.
*   **Interactive API Playground:** Native integration with OpenAPI/Swagger specifications.

---

## 📊 Dataset Profile & Model Training

The underlying engine handles a balanced schema targeting **11 valid classes**:
`Adult` · `Arts` · `Business` · `Education` · `Gaming` · `Health` · `Kids` · `Lifestyle` · `News` · `Recreation` · `Technology`

The comprehensive master production database (`master_scraped_v2.csv`) integrates 5 target assets:
1.  **DMOZ Cleaned Directory:** High-volume baseline repository for standard internet safety signatures.
2.  **Indian URLs Dataset:** Regional domain footprint optimization (e.g., `.in` and `.co.in`) to catch localized context nuances.
3.  **Manual Adult/Only Blacklist:** Explicit safety override payload ensuring robust filter protection bounds.
4.  **Targeted Top-Up Vectors:** Dynamically injected training payloads used to balance weak categories and enforce stable multi-class prediction metrics.
5.  **Master Production Archive:** Unified, deduplicated database containing parsed text layouts.

### 📈 Model Verification
*   **Core Validation Accuracy:** `91.8%` across global validation vectors.
*   **Held-out Generalization Test Accuracy:** `82.2% - 85.4%` across raw, un-sanitized real-world production web data.
*   **Data Consistency Bounds:** Regulated mean duplication ratio under `0.45` to ensure your transformers process actual web context instead of duplicate navigation menus.
*   **Model Card Registry:** `huggingface.co/SanandaDutta/website-category-distilbert`

---

## ⚡ Quick Start: RapidAPI Integration Example

Integrate real-time URL classification into your production application with a single API call:

### cURL
```bash
curl -X POST "[https://website-category-classifier.p.rapidapi.com/classify/url](https://website-category-classifier.p.rapidapi.com/classify/url)" \
  -H "Content-Type: application/json" \
  -H "X-RapidAPI-Key: 19459d36b0msh5f8bf9042600944p15d9b0jsnff32ec11ef9a" \
  -H "X-RapidAPI-Host: website-category-classifier.p.rapidapi.com" \
  -d '{"url": "[https://github.com](https://github.com/Sananda-Dutta/website-category-classifier)"}'
```
##Python Code
```python
import requests

url = "[https://website-category-classifier.p.rapidapi.com/classify/url](https://website-category-classifier.p.rapidapi.com/classify/url)"
headers = {
    "Content-Type": "application/json",
    "X-RapidAPI-Key": "19459d36b0msh5f8bf9042600944p15d9b0jsnff32ec11ef9a",
    "X-RapidAPI-Host": "website-category-classifier.p.rapidapi.com"
}
payload = {"url": "[https://github.com](https://github.com)"}

response = requests.post(url, json=payload, headers=headers)
print(response.json())
```
##JSON response reload
```json
{
  "url": "[https://github.com](https://github.com)",
  "category": "Technology",
  "confidence": 94.2,
  "top3": [
    {"category": "Technology", "confidence": 94.2},
    {"category": "Business", "confidence": 3.1},
    {"category": "Education", "confidence": 1.4}
  ],
  "method": "domain_shortcut",
  "safe_for_work": true,
  "time_ms": 12.4
}
```
## 📖 API Endpoint Blueprint

| Endpoint | Method | Payload | Description |
| :--- | :---: | :--- | :--- |
| `/classify/url` | `POST` | `{"url": "site.com"}` | Passes inputs through primary V2 scraper and returns model classifications. |
| `/classify/text` | `POST` | `{"text": "raw body content"}` | Classifies raw string tokens directly, bypassing the scraper step. |
| `/classify/batch` | `POST` | `{"urls": ["site1.com", "site2.com"]}` | Concurrently batches inference lists up to 20 URLs. |
| `/safe-check` | `POST` | `{"url": "site.com"}` | Returns a direct Adult/Kids-safe binary verdict for content filtering. |
| `/explain` | `GET` | `?url=example.com` | Exposes underlying metadata metrics and token feature assembly maps. |
| `/health` | `GET` | *None* | Baseline status heartbeat checking live connectivity to cluster nodes. |

### 🛠️ Tech Stack
* **Core Frameworks:** Python | FastAPI | Uvicorn | PyTorch
* **Transformer Core:** DistilBERT (Fine-tuned via Hugging Face Transformers)
* **Scraping Infrastructure:** Playwright + `playwright-stealth` | BeautifulSoup4 | Browserless.io
* **Hosting Architecture:** Render (Gateway Edge Node) | Railway (Scraper & ML Classifier Microservices)
* **Commercial Distribution:** RapidAPI

### ⚙️ Local Development Setup
If you want to pull down the repository and deploy the stack locally inside your local workspace environment:
```bash
# 1. Clone & Set Up Environment
git clone [https://github.com/Sananda-Dutta/website-category-classifier.git](https://github.com/Sananda-Dutta/website-category-classifier.git)
cd website-category-classifier

# 2. Install Standard Dependencies
pip install -r requirements.txt

# 3. Launch Local Server Instance
uvicorn api:app --reload
```
Once the local Uvicorn workers warm up, access the native interactive swagger playground documentation directly at: 
`📌 http://127.0.0.1:8000/docs`

### 🗺️ Commercial Roadmap
* [x] Model training, metric optimization & baseline fine-tuning.
* [x] Resilient scraping pipeline implementation (Stealth + Cloud Fallback).
* [x] Core Production Cluster Migration (Offloading to Render/Railway Infrastructure).
* [x] RapidAPI Marketplace Listing Deployment.
* [ ] Cross-listing integration across secondary distribution channels (APILayer, Apify, Eden AI).
* [ ] Finalizing expanded open-source public model card documentation

## 👤 Author

Built with 💻 by **Sananda Dutta** — [HuggingFace](https://huggingface.co/SanandaDutta) · [LinkedIn](https://www.linkedin.com/in/sananda-dutta-71a26a346/)

### 📄 License
This project is open-source software licensed under the `MIT License`.
