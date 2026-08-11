
import re
import string
import joblib
import streamlit as st

st.set_page_config(
    page_title="EmotionLens | AI Text Emotion",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "best_model.pkl"
VECTORIZER_PATH = "TF_IDF.pkl"

# The notebook maps the dataset's emotion strings to integers in first-seen order.
# The common dataset used in the notebook is:
# 0=sadness, 1=anger, 2=love, 3=surprise, 4=fear, 5=joy.
# If your original train.txt used a different first-seen order, edit this dictionary
# to exactly match that order.
EMOTION_NAMES = {
    0: "Sadness",
    1: "Anger",
    2: "Love",
    3: "Surprise",
    4: "Fear",
    5: "Joy",
}

EMOTION_ICONS = {
    "Sadness": "🌧️",
    "Anger": "🔥",
    "Love": "💗",
    "Surprise": "✨",
    "Fear": "🌙",
    "Joy": "☀️",
}

@st.cache_resource
def load_artifacts():
    return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)

def clean_text(text: str) -> str:
    """Match the preprocessing used in the supplied training notebook."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = "".join(ch for ch in text if not ch.isdigit())
    text = "".join(ch for ch in text if ch.isascii())
    return text

def predict_emotion(text: str):
    model, vectorizer = load_artifacts()
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    pred = int(model.predict(features)[0])

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        confidence = float(max(probabilities))
    else:
        probabilities = None
        confidence = None

    return pred, cleaned, confidence, probabilities

# ---------- Styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.block-container {
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
.hero {
    padding: 2.2rem 2.4rem;
    border: 1px solid rgba(120,120,140,.18);
    border-radius: 28px;
    background: linear-gradient(135deg, rgba(91,72,255,.12), rgba(255,255,255,.65));
    margin-bottom: 1.5rem;
}
.hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(2.2rem, 5vw, 4.2rem);
    line-height: .98;
    margin: 0;
    letter-spacing: -2px;
}
.hero p {
    font-size: 1.08rem;
    max-width: 760px;
    opacity: .75;
    margin-top: 1rem;
}
.eyebrow {
    text-transform: uppercase;
    letter-spacing: 2px;
    font-size: .75rem;
    font-weight: 700;
    opacity: .6;
    margin-bottom: .8rem;
}
.result-card {
    padding: 2rem;
    border-radius: 24px;
    border: 1px solid rgba(120,120,140,.18);
    background: rgba(255,255,255,.72);
    box-shadow: 0 12px 40px rgba(20,20,40,.06);
}
.result-label {
    font-size: .8rem;
    text-transform: uppercase;
    letter-spacing: 1.6px;
    opacity: .55;
    font-weight: 700;
}
.result-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3rem;
    font-weight: 700;
    margin: .2rem 0 .8rem;
}
.pill {
    display: inline-block;
    padding: .35rem .75rem;
    border-radius: 999px;
    background: rgba(91,72,255,.1);
    font-size: .82rem;
    font-weight: 700;
}
.small-muted {
    opacity: .62;
    font-size: .9rem;
}
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.45rem;
    font-weight: 700;
    margin: 1.8rem 0 .8rem;
}
[data-testid="stSidebar"] {
    border-right: 1px solid rgba(120,120,140,.15);
}
.stButton > button {
    border-radius: 14px;
    font-weight: 700;
    min-height: 3rem;
}
textarea {
    border-radius: 18px !important;
}
.footer {
    text-align:center;
    opacity:.5;
    font-size:.8rem;
    margin-top:3rem;
}
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🧠 EmotionLens")
    st.caption("A TF-IDF + Logistic Regression text emotion classifier.")
    st.divider()
    st.markdown("### Model")
    st.write("**Algorithm:** Logistic Regression")
    st.write("**Features:** TF-IDF")
    st.write("**Classes:** 6")
    st.write("**Training CV:** 5-fold GridSearchCV")
    st.divider()
    st.markdown("### Pipeline")
    st.write("1. Lowercase")
    st.write("2. Remove punctuation")
    st.write("3. Remove numbers")
    st.write("4. Remove non-ASCII characters")
    st.write("5. TF-IDF vectorization")
    st.write("6. Logistic Regression")
    st.divider()
    st.caption("Built from the supplied notebook and trained model artifacts.")

# ---------- Header ----------
st.markdown("""
<div class="hero">
    <div class="eyebrow">AI • NLP • MACHINE LEARNING</div>
    <h1>Understand the emotion<br>behind your words.</h1>
    <p>Paste a sentence and let the trained NLP model estimate the emotion hidden in the text.</p>
</div>
""", unsafe_allow_html=True)

# ---------- Main interaction ----------
left, right = st.columns([1.25, .75], gap="large")

with left:
    st.markdown('<div class="section-title">✍️ Your text</div>', unsafe_allow_html=True)

    examples = [
        "I am so happy today, everything is going perfectly!",
        "I cannot believe you did that to me.",
        "I miss the person I used to love.",
        "I am scared about what will happen next.",
        "Wow, I never expected that surprise!",
    ]
    example = st.selectbox("Quick examples", ["Write my own"] + examples)

    default_text = "" if example == "Write my own" else example
    text = st.text_area(
        "Message",
        value=default_text,
        height=190,
        placeholder="Type something like: I finally got the job I wanted!",
        label_visibility="collapsed",
    )

    c1, c2 = st.columns([2, 1])
    with c1:
        analyze = st.button("✨ Analyze emotion", type="primary", use_container_width=True)
    with c2:
        clear = st.button("Clear", use_container_width=True)

    if clear:
        st.rerun()

with right:
    st.markdown('<div class="section-title">🎯 Prediction</div>', unsafe_allow_html=True)

    if analyze:
        if not text.strip():
            st.warning("Please enter some text first.")
        else:
            try:
                pred, cleaned, confidence, probabilities = predict_emotion(text)
                emotion = EMOTION_NAMES.get(pred, f"Class {pred}")
                icon = EMOTION_ICONS.get(emotion, "🧠")

                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">Detected emotion</div>
                    <div class="result-name">{icon} {emotion}</div>
                    <span class="pill">Class {pred}</span>
                </div>
                """, unsafe_allow_html=True)

                if confidence is not None:
                    st.metric("Model confidence", f"{confidence:.1%}")

                if probabilities is not None:
                    st.markdown("#### Confidence by class")
                    chart_data = {
                        EMOTION_NAMES.get(int(cls), f"Class {cls}"): float(prob)
                        for cls, prob in zip(load_artifacts()[0].classes_, probabilities)
                    }
                    st.bar_chart(chart_data)

                with st.expander("See preprocessing"):
                    st.code(cleaned or "[empty after preprocessing]")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    else:
        st.info("Your prediction will appear here.")
        st.caption("Tip: try one of the example sentences.")

# ---------- Project explanation ----------
st.markdown('<div class="section-title">🔬 How this project works</div>', unsafe_allow_html=True)

a, b, c = st.columns(3)
with a:
    st.markdown("**01 — Clean**")
    st.caption("The text is transformed using the same cleaning steps used during training.")
with b:
    st.markdown("**02 — Represent**")
    st.caption("TF-IDF converts the cleaned sentence into numerical features.")
with c:
    st.markdown("**03 — Predict**")
    st.caption("The trained Logistic Regression model predicts one of six emotion classes.")

st.markdown('<div class="section-title">📦 Project details</div>', unsafe_allow_html=True)
d1, d2, d3, d4 = st.columns(4)
d1.metric("Vectorizer", "TF-IDF")
d2.metric("Classifier", "Logistic Regression")
d3.metric("CV", "5-fold")
d4.metric("Features", "15,046")

st.markdown("""
<div class="footer">
EmotionLens • Streamlit NLP Project • Trained model preserved from the supplied files
</div>
""", unsafe_allow_html=True)
