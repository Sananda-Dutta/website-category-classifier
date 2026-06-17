# 🌐 Website Category Classifier (V2 Production-Ready)
A production-grade, high-performance Website Category Classification engine built with a DistilBERT-powered NLP backbone,
 a lightweight FastAPI interface, and native asynchronous execution.This system uses advanced text preprocessing to 
 classify websites into 11 distinct verticals based on real-time HTML scraping, with fallback intelligence to handle
  anti-bot frameworks and empty page loads.

## 🚀 Key Improvements in Version 2Synchronized:
- Training & Inference Pipeline: Eliminated the historical train-inference skew by implementing a unified flat text architecture across both dataset compilation (phase3_scrape_master_v2_final.py) and live production endpoints.
-Pristine Natural Text Alignment: Stripped away legacy V1 artificial keyword multipliers and tail-end string injection hacks, forcing the transformer model to rely strictly on natural linguistic distributions.
-Granular Context Capture: Added automated paragraph extraction structures (<p> tag sampling) and widened the token intake window up to 3000 characters to retain high-density business context.

## ✨ Features:
-Multi-Layer Smart Classification Engine (smart_classify):Instant Routing (0ms): Checks local deterministic path and domain-level short-circuit lists first.Real-time Asynchronous Scraper: Fetches titles, meta headers, semantic headings ($H_1, H_2, H_3$), and paragraph body copy concurrently.
-Dual Fallback Fallback Loops: Automatically falls back to localized URL token features if a domain blocks scraping traffic or if the model records low-confidence thresholds.
-Production-Grade Text Sanitization: Active regex pipelines to isolate clean web features, eliminate structural punctuation, and filter out domain-level semantic noise words.
-High-Concurrency Scraping Architecture: Features random rotational User-Agent headers and native request tracking logic to bypass anti-scraping flags.Interactive API Playground: Native integration with OpenAPI specifications.

## 🛠️ Tech Stack
-Core Language:Python
-Web Framework: FastAPI + Uvicorn
-Deep Learning Framework: PyTorch
-Transformer Architecture: DistilBERT (Fine-tuned via Hugging Face Transformers)
-HTML Parsing Engine: BeautifulSoup4
-Hosting Ecosystem: Hugging Face Hub (Model Hub hosting) & Render / Space deployment pipelines

## 📊 Dataset Profile & Production Balance
The model is optimized to recognize 11 target valid classes:Adult, Arts, Business, Education, Gaming, Health, Kids, Lifestyle, News, Recreation, and Technology.The system's underlying training pipeline integrates five distinct target assets:DMOZ Cleaned Directory: High-volume baseline repository for standard internet safety signatures.Indian URLs Dataset: Regional domain footprint optimization (e.g., .in and .co.in) to catch localized context nuances.Manual Adult/Only Blacklist: Explicit safety override payload ensuring robust filter protection bounds.Targeted Top-Up Vectors: Dynamically injected training payloads used to balance weak categories and enforce stable multi-class prediction metrics.Master Production Archive (master_scraped_v2.csv): Unified, deduplicated database containing parsed text layouts.

## ⚡ Performance & Model VerificationCore Test Accuracy: 85.4% across global validation vectors.Data Consistency Bounds: Regulated mean duplication ratio under 0.45 to ensure your transformers process actual web context instead of duplicate navigation menus.

## 💻 Installation & Local Execution1. Clone & Set Up EnvironmentBashgit clone https://github.com/YOUR_USERNAME/website-category-classifier-final.git

cd website-category-classifier-final
2. Install Standard DependenciesBashpip install -r requirements.txt
3. Launch Local Server InstanceBashuvicorn api:app --reload

## 📖 API Endpoint BlueprintOnce the server is running locally, access the interactive documentation at:📌 http://127.0.0.1:8000/docsKey Production Endpoints Available:RouteMethodPayload ContextExecution Target/classifyPOST{"url": "example.com"}Passes inputs through the primary optimized V2 scraper and returns model classifications./batch_classifyPOST{"urls": ["site1.com", "site2.com"]}Concurrently batches inference lists using clean multi-threaded execution./explainGET?url=example.comExposes underlying metadata metrics, tracking feature string assembly maps and structural execution modes./healthGETNoneBaseline status heartbeat checking live connectivity to Hugging Face Spaces.