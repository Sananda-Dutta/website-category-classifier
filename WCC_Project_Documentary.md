# Website Category Classifier (WCC)
## A Full Project Documentary — From First Prototype to Production API

**Author:** Sananda Dutta
**Model card:** huggingface.co/SanandaDutta/website-category-distilbert
**Status:** Live in production, monetized on RapidAPI, hosted on Google Cloud Run

---

## 1. Origin — What Problem Was This Solving?

The project started from a simple observation: a huge number of downstream systems — ad networks, parental-control apps, brand-safety platforms, EdTech tools — all need to answer one question before they can act: *"What kind of website is this?"*

Most real-world solutions to that question are either manually curated domain lists (which go stale the moment a site changes or a new one appears) or crude keyword filters (which are trivial to fool, since a page doesn't need to contain a "banned word" to belong to an unwanted category). There was a clear gap for something that looked at a page's *actual live content* and made a real-time judgment call.

That became the goal: build an API that takes any URL, reads what's really on the page, and returns a content category — reliably enough to be worth paying for.

---

## 2. Phase 1 — The First Model: TF-IDF + SVM

**What was built:** A classical NLP pipeline — TF-IDF vectorization of page text, feeding a Support Vector Machine classifier across the category set.

**Why this approach first:** TF-IDF + SVM is the standard, fast-to-iterate baseline for text classification. It doesn't require GPU training, trains in minutes, and gives a quick read on whether the category boundaries in the data are even learnable before investing in something heavier like a transformer model.

**What went wrong:** Accuracy plateaued hard — around **59%**, no matter how the SVM's hyperparameters were tuned or how much data was thrown at it. A ceiling that stubborn is rarely a modeling problem; it's usually a data problem. That mismatch was the trigger for the next phase.

---

## 3. Phase 2 — Diagnosing the 59% Ceiling: Train-Inference Mismatch

**The investigation:** Instead of continuing to tune the model, the training data itself was audited. The finding: the training set contained mostly **domain-name tokens** (e.g. the words in a URL or site title) rather than **real page content** (body text, actual article/product copy). But at inference time, the model was being asked to classify based on real scraped content — a mismatch between what the model learned to recognize and what it was actually being shown in production.

**Why this mattered:** A model trained predominantly on domain-name patterns learns shortcuts tied to naming conventions, not genuine topical signal. It looked reasonable on paper (training accuracy wasn't catastrophic) but collapsed in real-world use because the *distribution* of information it saw during training didn't match the distribution it saw during inference.

**The decision:** Rebuild the dataset from real, scraped page content instead of relying on domain/title tokens as a shortcut.

---

## 4. Phase 3 — Rebuilding the Dataset

**What was done:**
- Scraped data from **DMOZ** (a historical open web directory) to get a broad, pre-categorized base of real websites across topics.
- Sourced and verified **Indian URL datasets** to ensure the classifier wasn't just trained on a Western-web-centric sample — important since the deployed product needed to generalize across a global range of live traffic.
- Specifically **augmented the Adult category** — sensitive-content categories are typically underrepresented in general-purpose scraped datasets, and a classifier that can't reliably catch this category is close to useless for its main real-world use case (brand safety, parental controls).

**Why this combination:** No single source gives balanced, representative, real-content coverage across 11 categories. Combining an established directory (DMOZ) with independently verified regional data and deliberate category augmentation was necessary to avoid a dataset that was accidentally biased toward whatever was easiest to scrape.

---

## 5. Phase 4 — Catching a Second, Subtler Leakage Bug

**The problem:** Even after the dataset rebuild, evaluation numbers looked *too* good in places — a classic sign of data leakage. On inspection, the issue was **seed-word repetition**: certain words used to *seed the search/scrape process* for a given category (e.g. searching "gaming" to find gaming sites) were ending up repeated in the scraped text itself, giving the model a shortcut — it could learn to detect the seed word rather than the actual topical content.

**Why this is dangerous:** A model exploiting seed-word leakage will show excellent validation accuracy but fail on real-world URLs that happen to be genuinely about a topic without using the "obvious" seed vocabulary. It's an invisible failure mode until you test on truly out-of-distribution URLs.

**The fix:** Identify and strip seed-word artifacts from the training text, then re-validate on a held-out set constructed to *not* share the seeding methodology — i.e., testing generalization, not memorization.

---

## 6. Phase 5 — Upgrading to DistilBERT

**Why move on from TF-IDF + SVM:** Once the dataset was clean, the ceiling shifted from "the data is wrong" to "the model architecture isn't expressive enough." TF-IDF treats words as independent, order-less features — it can't capture context (e.g. distinguishing "python" the programming language from "python" the animal based on surrounding words). A transformer-based model can.

**Why DistilBERT specifically, not full BERT:** DistilBERT retains roughly 97% of BERT's language understanding while being significantly smaller and faster — a meaningful factor for an API that needs to run inference affordably on free/low-tier cloud infrastructure, not a GPU cluster. For a classification task (not open-ended generation), the accuracy trade-off versus full BERT was worth the large gain in inference speed and hosting cost.

**Outcome:** After retraining on the cleaned, leakage-free dataset, the model reached **82.2% accuracy on a genuine held-out test set** — a number that, unlike the earlier TF-IDF results, actually reflects real-world generalization.

---

## 7. Phase 6 — Architecture Decisions: Weighted Text Extraction & Microservice Split

**Weighted text extraction:** Rather than dumping all scraped text into the model with equal importance, the pipeline was designed to **weight different parts of a page differently** — title, meta description, headings, and body text each contribute, but not equally. Titles and meta tags tend to be higher-signal, lower-noise summaries of a page's purpose; raw body text carries volume but more noise. Weighting reflects that reality instead of treating a page as an undifferentiated bag of words.

**Why split into two microservices (scraper + classifier) instead of one monolith:**
- **Separation of concerns:** Scraping (I/O-bound, browser-automation-heavy) and inference (CPU/memory-bound, model-heavy) have very different resource profiles. Bundling them means the heaviest resource requirement of *either* task dictates the hosting tier for *both*.
- **Independent scaling and failure isolation:** If the scraper hits a bot-challenge storm and slows down, it doesn't take the classifier down with it, and vice versa.
- **Independent redeploy:** A scraper bug fix doesn't require redeploying (and risking) the classifier, and a model update doesn't require touching scraping logic.

A lightweight **FastAPI router** sits in front of both, handling orchestration and fallback logic, with RapidAPI as the outward-facing gateway for auth, quotas, and billing.

---

## 8. Phase 7 — Explainability: LIME, Implemented Then Removed

**What was tried:** LIME (Local Interpretable Model-Agnostic Explanations) was integrated to show *why* the model classified a page a certain way — useful for debugging and for giving API consumers trust in a given classification.

**Why it was removed:** LIME's computation adds meaningful memory overhead at inference time, and the API was running on **Render's free tier**, which has a strict 512MB memory ceiling. Keeping LIME in the deployed path risked out-of-memory crashes under real traffic. The trade-off decision: explainability is valuable, but not at the cost of reliability on a hosting tier with no headroom. LIME was kept as a local/offline diagnostic tool rather than a production feature.

---

## 9. Phase 8 — Bug Fixes That Shaped the Architecture

A few production bugs were significant enough to actually change how the system was built, not just patched:

**Middleware auth header conflict.** The API needed to accept requests from two different callers with two different auth mechanisms — RapidAPI's proxy secret header (for marketplace traffic) and a direct API key (for direct/internal calls). Early middleware only checked for one, silently rejecting the other with a 403. The fix generalized the auth check to accept *either* valid header rather than assuming a single fixed auth path — a pattern that later proved directly relevant again when cross-listing on APILayer, which requires its own separate header check.

**Bot-challenge detection gaps.** Some sites return a "successful" HTTP response that's actually just a bot-challenge or CAPTCHA page, not real content. Early scraping logic didn't distinguish this from genuine content, so the classifier was sometimes classifying *challenge pages* instead of the actual site — silently corrupting results without throwing any error. This is what motivated explicit bot-challenge detection logic and, eventually, the Browserless.io fallback (see Phase 9).

**Silent scrape failure modes.** Related to the above — a scrape could "succeed" (no exception thrown) while returning empty or near-empty content. Without explicit checks, this fed garbage into the classifier, which would then confidently return a low-quality prediction with no visible sign anything had gone wrong. The fix added validation checks on scraped content before it's passed downstream, so failures fail loudly (fallback triggered) instead of silently (bad prediction returned).

---

## 10. Phase 9 — Browserless.io Fallback

**Why a fallback was needed at all:** Playwright alone, even with careful configuration, can't get past every site's bot protection — some sites specifically detect and block headless browser automation regardless of how it's configured.

**The decision:** Rather than accepting a hard failure rate on protected sites, a fallback path was added: when the primary Playwright scrape is detected as blocked or bot-challenged, the request is retried through **Browserless.io**, a managed headless-browser service with different fingerprinting characteristics that gets past some protections Playwright alone can't.

**Trade-off acknowledged:** This adds latency and, at scale, cost, for the subset of requests that need it — but it converts a hard failure into a successful (if slower) classification, which matters far more for API reliability than raw speed on the easy cases.

---

## 11. Phase 10 — The Chrome Extension

**Why:** Beyond the API itself, a Chrome extension was built as both a **testing tool** (quick manual validation of the classifier against whatever page you're currently browsing, without needing to hit the API through a separate client) and a **demonstration artifact** — a tangible, visual way to show the classifier working in real time, useful for demos, portfolio purposes, and later, the hackathon submission.

---

## 12. Phase 11 — Monetization: The RapidAPI Structure

**Why RapidAPI first:** It offers a self-serve marketplace with built-in auth, quota enforcement, and billing — meaning the project didn't need to build subscription/payment infrastructure from scratch to start generating revenue or getting real usage data.

**Tier structure:** Free / Basic / Pro tiers were set up to capture both the low-commitment evaluators (Free) and the higher-volume, higher-willingness-to-pay users (Basic/Pro) — a standard API-monetization pattern that lets the product self-segment its user base by usage need rather than guessing a single price point.

---

## 13. Phase 12 — The Hosting Journey: Three Platforms, One Lesson

This is one of the most consequential threads in the project's history — not because of any single dramatic failure, but because of what it taught about picking infrastructure for a product meant to run indefinitely, not just for a demo.

**Attempt 1 — Hugging Face Spaces.**
Chosen initially because it's a natural home for a Hugging-Face-hosted model, with easy integration and no separate infra to manage. **Suspended** because the deployment used Playwright for scraping, which HF Spaces' infrastructure doesn't support well for sustained production use (reinstatement was requested and, at time of most recent update, was still pending — no longer a blocker given the current setup).

*Lesson:* A hosting platform's suitability for a model doesn't guarantee its suitability for the surrounding infrastructure (in this case, browser automation) that the model depends on.

**Attempt 2 — Railway.**
Chosen as a quick, developer-friendly alternative that supports arbitrary containerized services (solving the Playwright problem). But Railway's free tier is a **30-day trial**, not a persistent free tier — meaning the "fix" was only ever temporary, with a hard countdown to migrate again.

*Lesson:* "It works and it's free" isn't the same question as "it works and it's free indefinitely." Trial credit disguised as a free tier creates a deadline that's easy to underestimate until it's close.

**Attempt 3 — Google Cloud Run (current, permanent).**
Chosen specifically to solve the sustainability problem the first two attempts didn't: Cloud Run's free tier is an **ongoing monthly allowance** (2 million requests, hundreds of thousands of GB-seconds and vCPU-seconds every month), not a countdown. It also supports arbitrary Docker containers, solving the Playwright constraint that broke HF Spaces.

*Migration process:* Both `wcc-scraper` and `wcc-classifier` were containerized (Dockerfiles listening on Cloud Run's injected `$PORT`), built via Cloud Build (`gcloud builds submit`), and deployed as independent Cloud Run services (`gcloud run deploy`) with explicit memory, CPU, timeout, and instance-scaling limits set deliberately — `min-instances 0` to stay within the free tier (accepting cold starts as a trade-off), `max-instances` capped to prevent a traffic spike or bug from silently exhausting the free quota.

*Why this was the last hosting migration (by design):* Unlike the prior two choices, Cloud Run was selected specifically to *not* need replacing again — evaluated against the actual failure modes of the previous two platforms (container support gap; trial-not-free-tier gap) rather than picked reactively.

---

## 14. Phase 13 — Monitoring, Verification, and Closing the Loop

Once both services were live on Cloud Run, the priority shifted from "does it work" to "will I know if it stops working":

- **UptimeRobot monitors** were repointed from the old Render/HF Spaces/Railway URLs to the new Cloud Run URLs, pointed at lightweight `/health` or `/stats` endpoints rather than the heavier classify endpoint, so monitoring pings wouldn't eat into request quota.
- **Full pipeline testing** was re-run end-to-end through RapidAPI's test console with varied real URLs, explicitly checking that the `method` field in responses showed genuine model-based classification (`combined_features` / `model`) rather than the `unable_to_extract_fallback` failure path — the same silent-failure risk identified back in Phase 8, now being checked as a release gate rather than discovered as a bug.

---

## 15. Phase 14 — Expanding Distribution: APILayer Cross-Listing

**Why a second marketplace:** RapidAPI captures one audience; APILayer captures a different, overlapping-but-distinct one. Since the underlying API doesn't need to change to be listed twice, and APILayer doesn't require exclusivity, this was a low-marginal-cost way to expand reach.

**Key difference from RapidAPI:** APILayer uses a manually reviewed provider application (via apilayer.com/provider) rather than instant self-serve listing — closer to a partnership relationship than a pure marketplace signup. This required adapting the project's existing documentation and pitch (reused from RapidAPI, reworded rather than duplicated) into their submission format, and being ready to demonstrate uptime/reliability practices (the UptimeRobot monitoring from Phase 13 became a relevant talking point here, not just an internal safeguard).

---

## 16. Phase 15 — Launch Assets: README, Demo GIF, and Diagrams

With the infrastructure and distribution work done, the remaining work shifted to **communicating** the project clearly to people who'd never seen it — recruiters, hiring managers, potential API customers:

- A rewritten **README** reflecting the *current* architecture (Cloud Run, not the now-retired HF Spaces/Railway setup), including a live sample request/response, category list, and architecture summary.
- A short **demo GIF** capturing a real RapidAPI test-console call, chosen over a static screenshot because it proves the API actually works live, not just that it has documentation describing that it should.
- A set of **diagrams** (system architecture, request workflow, model/system metrics, and a system-overview/pitch graphic) built to visually communicate the pipeline and the honest performance numbers without requiring a reader to parse code or prose to understand the system at a glance.

---

## 17. Phase 16 — Public Launch and Outreach (In Progress)

The final open track is direct distribution beyond marketplace listings: a **LinkedIn launch post** framing the project's technical journey (the pivots and fixes documented above are, deliberately, the actual substance of that post — not just a feature announcement), and **targeted outreach** to ad networks, brand-safety platforms, and EdTech companies who are plausible real customers, prioritized toward smaller/more accessible organizations first rather than large ad networks with in-house solutions and long sales cycles.

---

## 18. Retrospective — What This Project's History Actually Demonstrates

Looking back across all 16 phases, a few patterns repeat, and they're arguably the most transferable part of the project:

1. **Suspiciously good or suspiciously stuck numbers are both signals to investigate the data, not just the model.** Both the 59% ceiling (Phase 2) and the too-good-to-be-true post-rebuild numbers (Phase 4) were data problems wearing a model-performance costume.
2. **"Free" isn't a single category of hosting risk.** A trial-credit-based free tier (Railway) and a genuinely ongoing free tier (Cloud Run) look identical on day one and completely different on day thirty — this distinction wasn't obvious until it caused a second migration.
3. **Silent failures are worse than loud ones.** Several of the most consequential bugs (bot-challenge pages read as real content, empty scrapes treated as successful) didn't throw errors — they returned confidently wrong answers. The architectural response was to make failure *visible* (explicit validation, fallback triggers, a `method` field that reports which code path actually produced a result) rather than just trying to prevent failure entirely.
4. **Infrastructure decisions compound.** The microservice split (Phase 6) made the later Cloud Run migration (Phase 12) and the auth-header fix (Phase 8) both easier than they would have been in a monolith — an architectural choice made for one reason (resource isolation) paid off later for unrelated reasons (independent redeploy during migration, reusable auth pattern for a second marketplace).

---

## Appendix — Tech Stack Summary

| Layer | Technology |
|---|---|
| Model | DistilBERT (fine-tuned), via HuggingFace Transformers, PyTorch |
| Model hosting | Hugging Face Hub (`SanandaDutta/website-category-distilbert`) |
| Scraping | Playwright, with Browserless.io as bot-challenge fallback |
| API framework | FastAPI |
| Containerization | Docker |
| Hosting (current) | Google Cloud Run (2 independent microservices) |
| Hosting (retired) | Hugging Face Spaces → Railway → Google Cloud Run |
| Monetization | RapidAPI (Free/Basic/Pro tiers), APILayer (cross-listed) |
| Monitoring | UptimeRobot |
| Client tooling | Chrome extension (testing + demo) |
| Explainability (offline only) | LIME |

## Appendix — The 11 Categories

Adult · Arts · Business · Education · Gaming · Health · Kids · Lifestyle · News · Recreation · Technology
