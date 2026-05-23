import streamlit as st
import requests
import pandas as pd
import re
from urllib.parse import urlparse

# ─────────────────────────────────────────────
# CONFIG  🔴 CHANGE THIS LINE
# ─────────────────────────────────────────────
API_URL = "https://your-api.onrender.com"   # 🔴 your Render URL
HEADERS = {"Content-Type": "application/json"}

# ─────────────────────────────────────────────
# CATEGORY EMOJIS
# ─────────────────────────────────────────────
CATEGORY_EMOJI = {
    "News":          "📰",
    "E-commerce":    "🛒",
    "Technology":    "💻",
    "Sports":        "⚽",
    "Health":        "🏥",
    "Education":     "🎓",
    "Entertainment": "🎬",
    "Business":      "💼",
    "Adult":         "🔞",
    "Kids":          "👧",
    "Arts":          "🎨",
    "Recreation":    "🏕️",
    "Gaming":        "🎮",
    "Lifestyle":     "🏠",
    "Other":         "🌐",
}

RISK_COLOR = {
    "high":   "#ff4444",
    "medium": "#ffaa00",
    "low":    "#ffee44",
    "none":   "#44aa44",
}

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Website Category Classifier",
    page_icon="🌐",
    layout="centered"
)

st.markdown("""
<style>
.result-box {
    background: linear-gradient(135deg,#1e3a5f,#0f2540);
    border: 1px solid #2d6a9f;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
}
.result-label {
    font-size:.78rem;
    color:#7ab3d4;
    text-transform:uppercase;
    letter-spacing:1px;
    margin-bottom:4px;
}
.result-value {
    font-size:1.8rem;
    font-weight:700;
    color:#ffffff;
}
.badge {
    display:inline-block;
    padding:2px 10px;
    border-radius:20px;
    font-size:.75rem;
    margin-top:6px;
}
.badge-source { background:#1a4a6e; color:#7ab3d4; }
.badge-bert   { background:#1a3a2e; color:#5dba7d; }
.badge-unsafe { background:#4e1a1a; color:#ff6b6b; }
.badge-safe   { background:#1a4e2e; color:#5dba7d; }
.word-pos {
    background:#1a4e2e; color:#5dba7d;
    padding:2px 8px; border-radius:4px;
    margin:2px; display:inline-block; font-size:.82rem;
}
.word-neg {
    background:#4e1a1a; color:#ff6b6b;
    padding:2px 8px; border-radius:4px;
    margin:2px; display:inline-block; font-size:.82rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SHARED ERROR PARSER
# ─────────────────────────────────────────────
def parse_response(res):
    try:
        data = res.json()
    except Exception:
        return None, f"Server returned non-JSON response (status {res.status_code})"
    if res.status_code == 422:
        return None, f"❌ Validation error: {data.get('detail','Unprocessable input.')}"
    if res.status_code == 429:
        return None, "⏱️ Rate limit exceeded. Wait a minute and try again."
    if res.status_code >= 500:
        return None, f"🔴 Server error ({res.status_code}). API may be starting — wait 30s and retry."
    if res.status_code != 200:
        return None, f"Unexpected status {res.status_code}: {data}"
    if "error" in data:
        return None, data["error"]
    return data, None

# ─────────────────────────────────────────────
# API CALL FUNCTIONS
# ─────────────────────────────────────────────
def call_classify_url(url):
    try:
        res = requests.post(
            f"{API_URL}/classify/url",
            json={"url": url},
            headers=HEADERS, timeout=30
        )
        return parse_response(res)
    except requests.exceptions.ConnectionError:
        return None, "🔴 Cannot reach the API. Check API_URL or internet."
    except requests.exceptions.Timeout:
        return None, "⏱️ Timed out. Server may be waking up — try again in 30s."
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"


def call_classify_text(text):
    try:
        res = requests.post(
            f"{API_URL}/classify/text",
            json={"text": text},
            headers=HEADERS, timeout=20
        )
        return parse_response(res)
    except requests.exceptions.ConnectionError:
        return None, "🔴 Cannot reach the API."
    except requests.exceptions.Timeout:
        return None, "⏱️ Timed out. Try again."
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"


def call_classify_batch(urls):
    try:
        res = requests.post(
            f"{API_URL}/classify/batch",
            json={"urls": urls},
            headers=HEADERS, timeout=90
        )
        return parse_response(res)
    except requests.exceptions.ConnectionError:
        return None, "🔴 Cannot reach the API."
    except requests.exceptions.Timeout:
        return None, "⏱️ Batch timed out. Try fewer URLs."
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"


def call_safe_check(url):
    try:
        res = requests.post(
            f"{API_URL}/safe-check",
            json={"url": url},
            headers=HEADERS, timeout=30
        )
        return parse_response(res)
    except requests.exceptions.ConnectionError:
        return None, "🔴 Cannot reach the API."
    except requests.exceptions.Timeout:
        return None, "⏱️ Timed out. Try again."
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"


def call_explain(text):
    try:
        res = requests.post(
            f"{API_URL}/explain",
            json={"text": text},
            headers=HEADERS, timeout=60
        )
        return parse_response(res)
    except requests.exceptions.ConnectionError:
        return None, "🔴 Cannot reach the API."
    except requests.exceptions.Timeout:
        return None, "⏱️ Explanation timed out. Try shorter text."
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"

# ─────────────────────────────────────────────
# REUSABLE RESULT CARD
# ─────────────────────────────────────────────
def show_result_card(data, source="API"):
    category   = data.get("category", "Unknown")
    confidence = data.get("confidence", 0)
    method     = data.get("method", "ml_model")
    time_ms    = data.get("time_ms", "—")
    emoji      = CATEGORY_EMOJI.get(category, "🌐")
    badge_text = "DistilBERT" if "domain" not in str(method) else f"Domain shortcut"

    st.markdown(f"""
    <div class="result-box">
        <div class="result-label">Predicted Category</div>
        <div class="result-value">{emoji} {category}</div>
        <span class="badge badge-bert">{badge_text}</span>
        <span class="badge badge-source" style="margin-left:6px">
            {confidence}% confidence · {time_ms} ms
        </span>
    </div>
    """, unsafe_allow_html=True)

    top3 = data.get("top3", [])
    if top3:
        st.markdown("#### Top 3 Predictions")
        for item in top3:
            e = CATEGORY_EMOJI.get(item["category"], "🌐")
            col_a, col_b = st.columns([5, 1])
            with col_a:
                st.progress(
                    item["confidence"] / 100,
                    text=f"{e} {item['category']}"
                )
            with col_b:
                st.write(f"**{item['confidence']}%**")

# ─────────────────────────────────────────────
# HEALTH CHECK  (cached 60s)
# ─────────────────────────────────────────────
@st.cache_data(ttl=60)
def check_health():
    try:
        res = requests.get(f"{API_URL}/health", timeout=10)
        if res.status_code == 200:
            d = res.json()
            return True, d.get("classes", []), d.get("model_type", "DistilBERT")
        return False, [], ""
    except Exception:
        return False, [], ""

is_online, categories, model_type = check_health()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("🌐 Website Category Classifier")
st.caption("DistilBERT fine-tuned · 13 categories · FastAPI + Render")

if is_online:
    c1, c2, c3 = st.columns(3)
    c1.metric("Status",     "Online ✅")
    c2.metric("Categories", len(categories) if categories else 13)
    c3.metric("Model",      model_type or "DistilBERT")
else:
    st.warning(
        "⚠️ API offline or starting up — Render free tier sleeps after 15 min. "
        "Wait 30 seconds then refresh.",
        icon="🟡"
    )
    if st.button("🔄 Retry connection"):
        st.cache_data.clear()
        st.rerun()

st.divider()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔗 URL",
    "📝 Text",
    "📦 Batch",
    "🛡️ Safety Check",
    "🔎 Explain"
])

# ══════════════════════════════════════════════
# TAB 1 — CLASSIFY BY URL
# ══════════════════════════════════════════════
with tab1:
    st.markdown("Enter any URL — the API scrapes it live and classifies using DistilBERT.")
    url_input = st.text_input(
        "Website URL",
        placeholder="https://www.bbc.com/news",
        key="url_tab"
    )

    if st.button("🔍 Scrape & Classify", key="btn_url", use_container_width=True):
        if not url_input.strip():
            st.warning("Please enter a URL.")
        else:
            with st.spinner("Scraping and classifying..."):
                data, err = call_classify_url(url_input.strip())

            if err:
                st.error(err)
                st.info("Tip: Make sure API_URL at the top of app.py is your actual Render URL.")
            else:
                show_result_card(data, source="Live scrape")

                if data.get("scraped"):
                    with st.expander("📄 What was scraped from this page"):
                        s = data["scraped"]
                        st.write(f"**Title:** {s.get('title') or '—'}")
                        st.write(f"**Meta description:** {s.get('meta_description') or '—'}")
                        st.write(f"**H1:** {(s.get('h1') or '—')[:120]}")
                        st.write(f"**Body preview:** {(s.get('body') or '—')[:300]}...")

# ══════════════════════════════════════════════
# TAB 2 — CLASSIFY BY TEXT
# ══════════════════════════════════════════════
with tab2:
    st.markdown("Paste any website text — homepage copy, article, product description.")
    text_input = st.text_area(
        "Website text",
        height=200,
        placeholder="e.g. 'Buy the latest smartphones at the best prices. Free delivery above ₹499...'",
        key="text_tab"
    )

    if st.button("🔍 Classify Text", key="btn_text", use_container_width=True):
        if not text_input.strip():
            st.warning("Please enter some text.")
        elif len(text_input.strip()) < 10:
            st.warning("Text too short — enter at least 10 characters.")
        else:
            with st.spinner("Classifying..."):
                data, err = call_classify_text(text_input.strip())

            if err:
                st.error(err)
            else:
                show_result_card(data, source="Text input")

# ══════════════════════════════════════════════
# TAB 3 — BATCH MODE
# ══════════════════════════════════════════════
with tab3:
    st.markdown("Upload a CSV with a `url` column — classify up to 20 URLs at once.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="batch_tab")

    if uploaded:
        df_up = pd.read_csv(uploaded)
        df_up.columns = df_up.columns.str.strip().str.lower()

        if "url" not in df_up.columns:
            st.error("CSV must have a column named `url`.")
        else:
            total_urls = len(df_up["url"].dropna())
            st.write(f"Found **{total_urls}** URLs. Preview:")
            st.dataframe(df_up.head(5), use_container_width=True)

            if total_urls > 20:
                st.info("Only the first 20 URLs will be classified (API limit per request).")

            if st.button("🚀 Run Batch Classification", key="btn_batch", use_container_width=True):
                urls = df_up["url"].dropna().tolist()[:20]

                with st.spinner(f"Classifying {len(urls)} URLs..."):
                    data, err = call_classify_batch(urls)

                if err:
                    st.error(err)
                elif not data or not data.get("results"):
                    st.warning("API returned no results. Try again.")
                else:
                    results   = data["results"]
                    df_result = pd.DataFrame(results)

                    st.success(f"✅ Classified {data.get('total', len(df_result))} URLs!")
                    st.dataframe(df_result, use_container_width=True)

                    if "category" in df_result.columns:
                        st.markdown("#### Category Breakdown")
                        counts = df_result["category"].value_counts().reset_index()
                        counts.columns = ["Category", "Count"]
                        st.bar_chart(counts.set_index("Category"))

                    csv_out = df_result.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download Results CSV",
                        data=csv_out,
                        file_name="classification_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

# ══════════════════════════════════════════════
# TAB 4 — BRAND SAFETY DASHBOARD
# ══════════════════════════════════════════════
with tab4:
    st.markdown("### 🛡️ Brand Safety Dashboard")
    st.caption(
        "Check if URLs are safe for ads, kids, or workplace. "
        "Uses the /safe-check endpoint which flags Adult, Gaming, and other risky categories."
    )

    # Single URL check
    st.markdown("#### Single URL")
    safe_url = st.text_input(
        "URL to check",
        placeholder="https://www.example.com",
        key="safe_single"
    )

    if st.button("🔍 Check Safety", key="btn_safe", use_container_width=True):
        if not safe_url.strip():
            st.warning("Enter a URL.")
        else:
            with st.spinner("Checking safety..."):
                data, err = call_safe_check(safe_url.strip())

            if err:
                st.error(err)
                st.info(
                    "Make sure your api.py has the /safe-check endpoint. "
                    "If not, add it from the Phase 3 guide."
                )
            else:
                category   = data.get("category", "Unknown")
                is_safe    = data.get("is_safe", True)
                is_kids    = data.get("is_kids_safe", False)
                risk       = data.get("risk_level", "none")
                confidence = data.get("confidence", 0)
                emoji      = CATEGORY_EMOJI.get(category, "🌐")

                if is_safe:
                    st.success(f"✅ SAFE — {emoji} {category} ({confidence}% confidence)")
                else:
                    st.error(f"🚫 UNSAFE — {emoji} {category} ({confidence}% confidence)")

                col1, col2, col3 = st.columns(3)
                col1.metric("Category",  f"{emoji} {category}")
                col2.metric("Ad Safe",   "✅ Yes" if is_safe  else "🚫 No")
                col3.metric("Kids Safe", "✅ Yes" if is_kids  else "🚫 No")

                risk_color = RISK_COLOR.get(risk, "#888888")
                st.markdown(
                    f'<div style="background:{risk_color}22;border-left:4px solid {risk_color};'
                    f'padding:8px 14px;border-radius:0 8px 8px 0;margin-top:8px;font-size:.9rem;">'
                    f'Risk level: <strong>{risk.upper()}</strong></div>',
                    unsafe_allow_html=True
                )

    st.divider()

    # Bulk safety check
    st.markdown("#### Bulk Safety Report")
    st.caption("Upload a CSV of ad placement URLs and get a color-coded safety report.")

    bulk_file = st.file_uploader("Upload CSV (column: url)", type=["csv"], key="safe_bulk")

    if bulk_file:
        df_bulk = pd.read_csv(bulk_file)
        df_bulk.columns = df_bulk.columns.str.strip().str.lower()

        if "url" not in df_bulk.columns:
            st.error("CSV must have a `url` column.")
        else:
            total_bulk = len(df_bulk["url"].dropna())
            st.write(f"Found **{total_bulk}** URLs.")

            max_check = st.slider(
                "How many to check?",
                min_value=1,
                max_value=min(50, total_bulk),
                value=min(10, total_bulk),
                key="bulk_slider"
            )

            if st.button("🔍 Run Bulk Safety Check", key="btn_bulk", use_container_width=True):
                urls_to_check = df_bulk["url"].dropna().tolist()[:max_check]
                results_bulk  = []
                prog = st.progress(0, text="Checking URLs...")

                for i, url in enumerate(urls_to_check):
                    prog.progress(
                        (i + 1) / max_check,
                        text=f"Checking {i+1}/{max_check}..."
                    )
                    d, e = call_safe_check(str(url).strip())
                    if e:
                        results_bulk.append({
                            "URL":        url,
                            "Category":   "Error",
                            "Safe":       "—",
                            "Kids Safe":  "—",
                            "Risk Level": "UNKNOWN",
                            "Confidence": "—"
                        })
                    else:
                        results_bulk.append({
                            "URL":        url,
                            "Category":   d.get("category", "Unknown"),
                            "Safe":       "✅ Yes" if d.get("is_safe")      else "🚫 No",
                            "Kids Safe":  "✅ Yes" if d.get("is_kids_safe") else "🚫 No",
                            "Risk Level": d.get("risk_level", "none").upper(),
                            "Confidence": f"{d.get('confidence', 0)}%",
                        })

                prog.empty()
                df_safe_out = pd.DataFrame(results_bulk)

                safe_count  = (df_safe_out["Safe"] == "✅ Yes").sum()
                unsafe_count = len(df_safe_out) - safe_count
                kids_count  = (df_safe_out["Kids Safe"] == "✅ Yes").sum()

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Checked",     len(df_safe_out))
                m2.metric("Safe ✅",      safe_count)
                m3.metric("Unsafe 🚫",   unsafe_count)
                m4.metric("Kids Safe 👧", kids_count)

                st.success("Safety check complete!")
                st.dataframe(df_safe_out, use_container_width=True)

                st.markdown("#### Category Distribution")
                cat_counts = df_safe_out["Category"].value_counts().reset_index()
                cat_counts.columns = ["Category", "Count"]
                st.bar_chart(cat_counts.set_index("Category"))

                csv_safe = df_safe_out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Safety Report",
                    data=csv_safe,
                    file_name="safety_report.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# ══════════════════════════════════════════════
# TAB 5 — EXPLAIN  (LIME)
# ══════════════════════════════════════════════
with tab5:
    st.markdown("### 🔎 Why did the model predict that?")
    st.caption(
        "Paste any text and see exactly which words drove the prediction. "
        "Powered by LIME (Local Interpretable Model-agnostic Explanations)."
    )

    explain_text = st.text_area(
        "Text to explain",
        height=160,
        placeholder="Paste website text here — e.g. article body, product description...",
        key="explain_tab"
    )

    st.info(
        "⏱️ Explanation takes 15-30 seconds. LIME runs ~300 mini-predictions internally.",
        icon="ℹ️"
    )

    if st.button("🔎 Explain Prediction", key="btn_explain", use_container_width=True):
        if not explain_text.strip():
            st.warning("Please enter some text.")
        elif len(explain_text.strip()) < 20:
            st.warning("Text too short — enter at least 20 characters.")
        else:
            with st.spinner("Running LIME explanation (15-30 seconds)..."):
                data, err = call_explain(explain_text.strip())

            if err:
                st.error(err)
                st.info(
                    "The /explain endpoint must be added to your api.py first. "
                    "See the LIME integration guide in the roadmap."
                )
            else:
                prediction = data.get("prediction", "Unknown")
                key_words  = data.get("key_words", [])
                emoji      = CATEGORY_EMOJI.get(prediction, "🌐")

                st.success(f"Prediction: **{emoji} {prediction}**")

                if key_words:
                    st.markdown("#### Words that influenced this prediction")
                    st.caption(
                        "🟢 Green = pushed TOWARDS this category  |  "
                        "🔴 Red = pushed AWAY from this category"
                    )

                    pos_words = sorted(
                        [w for w in key_words if w["impact"] > 0],
                        key=lambda x: x["impact"], reverse=True
                    )
                    neg_words = sorted(
                        [w for w in key_words if w["impact"] <= 0],
                        key=lambda x: x["impact"]
                    )

                    if pos_words:
                        st.markdown("**Positive signals (pushing towards prediction):**")
                        chips = " ".join([
                            f'<span class="word-pos">'
                            f'{w["word"]} +{w["impact"]:.3f}'
                            f'</span>'
                            for w in pos_words
                        ])
                        st.markdown(chips, unsafe_allow_html=True)

                    if neg_words:
                        st.markdown("**Negative signals (pushing away):**")
                        chips = " ".join([
                            f'<span class="word-neg">'
                            f'{w["word"]} {w["impact"]:.3f}'
                            f'</span>'
                            for w in neg_words
                        ])
                        st.markdown(chips, unsafe_allow_html=True)

                    st.markdown("#### Impact Chart")
                    df_words = (
                        pd.DataFrame(key_words)
                        .sort_values("impact", ascending=True)
                    )
                    st.bar_chart(df_words.set_index("word")["impact"])

                else:
                    st.warning("No word-level explanation returned. Try longer text.")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.divider()
fc1, fc2, fc3 = st.columns(3)
fc1.caption("🤖 Model: DistilBERT fine-tuned")
fc2.caption("⚡ API: FastAPI + Render")
fc3.caption("🎨 UI: Streamlit")