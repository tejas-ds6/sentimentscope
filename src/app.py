"""
SentimentScope — Streamlit UI
Run: streamlit run src/app.py
"""

import streamlit as st
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))

from aspect_extractor import extract_aspects

st.set_page_config(page_title="SentimentScope", page_icon="🎯", layout="wide")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main { background: #0f1117; }
.stApp { background: #0f1117; }
.sentiment-card {
    padding: 16px; border-radius: 10px; margin: 8px 0;
    display: flex; justify-content: space-between; align-items: center;
}
.positive { background: #14532d33; border: 1px solid #166534; }
.negative { background: #7f1d1d33; border: 1px solid #991b1b; }
.neutral  { background: #1e293b;   border: 1px solid #334155; }
</style>
""", unsafe_allow_html=True)

st.title("🎯 SentimentScope")
st.caption("Aspect-Based Sentiment Analysis · Fine-tuned DistilBERT · by Tejas DS")
st.markdown("---")

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    model_path = st.text_input("Model path (leave blank for base)", value="")
    use_auto   = st.checkbox("Auto-extract aspects", value=True)
    manual_asp = st.text_input("Manual aspects (comma-separated)", "")
    st.markdown("---")
    st.markdown("**About**")
    st.markdown("Fine-tuned DistilBERT classifies sentiment per aspect.\n\n"
                "[GitHub](https://github.com/tejas-ds6/sentimentscope)")

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model(path):
    try:
        from model import SentimentScopeModel
        return SentimentScopeModel(model_path=path if path else None), None
    except Exception as e:
        return None, str(e)

model, err = load_model(model_path)
if err:
    st.warning(f"Could not load fine-tuned model: {err}\n\nRunning in demo mode.")

# ── Demo mode inference (no GPU / model required) ─────────────────────────────
DEMO_MAP = {
    "battery":    {"positive": 0.85, "neutral": 0.10, "negative": 0.05},
    "screen":     {"positive": 0.12, "neutral": 0.18, "negative": 0.70},
    "camera":     {"positive": 0.75, "neutral": 0.15, "negative": 0.10},
    "performance":{"positive": 0.60, "neutral": 0.25, "negative": 0.15},
    "price":      {"positive": 0.30, "neutral": 0.40, "negative": 0.30},
    "design":     {"positive": 0.80, "neutral": 0.15, "negative": 0.05},
}

def demo_predict(text: str, aspects: list):
    import random
    results = []
    tl = text.lower()
    for asp in aspects:
        if asp in DEMO_MAP:
            scores = DEMO_MAP[asp].copy()
        else:
            # Heuristic from text keywords
            if any(w in tl for w in ["great","love","amazing","excellent","best","perfect"]):
                scores = {"positive": 0.80, "neutral": 0.12, "negative": 0.08}
            elif any(w in tl for w in ["bad","terrible","awful","worst","hate","broken"]):
                scores = {"positive": 0.08, "neutral": 0.12, "negative": 0.80}
            else:
                scores = {"positive": 0.30, "neutral": 0.45, "negative": 0.25}
        best = max(scores, key=scores.get)
        results.append({"aspect": asp, "sentiment": best,
                         "confidence": scores[best], "scores": scores})
    return results

# ── Main input ────────────────────────────────────────────────────────────────
review_text = st.text_area(
    "📝 Paste your product review",
    value="The battery lasts all day which is fantastic, but the screen has terrible brightness "
          "outdoors. Camera quality is impressive for the price though.",
    height=120,
)

col1, col2 = st.columns([1, 1])
with col1:
    run_btn = st.button("🔍 Analyse Sentiment", type="primary", use_container_width=True)
with col2:
    clear_btn = st.button("🗑️ Clear", use_container_width=True)

if clear_btn:
    st.rerun()

if run_btn and review_text.strip():
    # Aspect collection
    if use_auto:
        aspects = extract_aspects(review_text)
        st.info(f"Auto-detected aspects: **{', '.join(aspects)}**")
    else:
        aspects = [a.strip() for a in manual_asp.split(",") if a.strip()]

    if not aspects:
        st.warning("No aspects found. Try entering them manually.")
        st.stop()

    # Run inference
    with st.spinner("Analysing…"):
        if model:
            results = model.predict(review_text, aspects)
        else:
            results = demo_predict(review_text, aspects)

    # ── Results ────────────────────────────────────────────────────────────────
    st.markdown("### 📊 Results")
    emoji_map = {"positive": "🟢 Positive", "neutral": "🟡 Neutral", "negative": "🔴 Negative"}

    col_a, col_b = st.columns(2)
    for i, r in enumerate(results):
        target_col = col_a if i % 2 == 0 else col_b
        with target_col:
            css_cls = r["sentiment"]
            pct     = int(r["confidence"] * 100)
            st.markdown(f"""
            <div class="sentiment-card {css_cls}">
                <div>
                    <strong style="font-size:15px;">{r['aspect'].title()}</strong><br>
                    <span style="font-size:22px;">{emoji_map[r['sentiment']]}</span>
                </div>
                <div style="text-align:right;">
                    <span style="font-size:28px;font-weight:700;">{pct}%</span><br>
                    <span style="font-size:11px;color:#94a3b8;">confidence</span>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("### 🔬 Score Breakdown")
    import pandas as pd
    df = pd.DataFrame([
        {"Aspect": r["aspect"].title(),
         "Positive": r["scores"]["positive"],
         "Neutral":  r["scores"]["neutral"],
         "Negative": r["scores"]["negative"],
         "Prediction": r["sentiment"].title()}
        for r in results
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.download_button("⬇️ Download JSON", json.dumps(results, indent=2),
                       file_name="sentimentscope_results.json", mime="application/json")
