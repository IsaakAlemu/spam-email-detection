"""
SpamShield AI — Streamlit frontend for the INSA Spam Detection project.

Run with:
    streamlit run streamlit_app.py
NOT:
    python streamlit_app.py
"""

import re
import string
import time

import joblib
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ----------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="SpamShield AI",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# One-time resource loading (cached so it only runs once per session)
# ----------------------------------------------------------------------
@st.cache_resource
def load_nltk_resources():
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    return set(stopwords.words("english")), WordNetLemmatizer()


@st.cache_resource
def load_model():
    model = joblib.load("model/spam_model.pkl")
    vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
    return model, vectorizer


stop_words, lemmatizer = load_nltk_resources()
model, vectorizer = load_model()

# ----------------------------------------------------------------------
# Preprocessing — must match the notebook pipeline exactly
# ----------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    words = [w for w in text.split() if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)


def predict(message: str):
    cleaned = clean_text(message)
    vector = vectorizer.transform([cleaned])
    pred = model.predict(vector)[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vector)[0]
        ham_conf, spam_conf = proba[0], proba[1]
    else:
        # Fallback if the model hasn't been recalibrated yet.
        # decision_function gives an unbounded margin, not a probability —
        # this is a rough approximation only, kept as a safety net.
        margin = model.decision_function(vector)[0]
        spam_conf = 1 / (1 + 2.71828 ** (-margin))
        ham_conf = 1 - spam_conf

    label = "Spam" if pred == 1 else "Ham"
    confidence = spam_conf if pred == 1 else ham_conf
    return label, confidence, ham_conf, spam_conf


# ----------------------------------------------------------------------
# Custom CSS — dark theme, glassmorphism cards
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, #1f2340 0%, #0d0e1a 60%);
        color: #eaeaf5;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 18px;
        padding: 1.6rem 1.8rem;
        backdrop-filter: blur(10px);
        margin-bottom: 1.2rem;
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #7c8cff, #ff7cd4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .hero-subtitle {
        color: #a0a3c2;
        font-size: 1.02rem;
        margin-top: 0.2rem;
    }
    .result-spam {
        background: rgba(255, 76, 96, 0.12);
        border: 1px solid rgba(255, 76, 96, 0.4);
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        text-align: center;
    }
    .result-ham {
        background: rgba(58, 220, 150, 0.12);
        border: 1px solid rgba(58, 220, 150, 0.4);
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        text-align: center;
    }
    .result-label-spam {
        font-size: 1.8rem;
        font-weight: 800;
        color: #ff7c8c;
    }
    .result-label-ham {
        font-size: 1.8rem;
        font-weight: 800;
        color: #4ee8a4;
    }
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.05) !important;
        color: #eaeaf5 !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
    }
    div.stButton > button {
        border-radius: 10px;
        border: none;
        background: linear-gradient(90deg, #7c8cff, #ff7cd4);
        color: white;
        font-weight: 700;
        padding: 0.55rem 1.4rem;
    }
    div.stButton > button:hover {
        opacity: 0.9;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Sidebar — model info
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛡️ SpamShield AI")
    st.markdown("Intelligent SMS / email spam detection.")
    st.markdown("---")
    st.markdown("#### Model")
    st.markdown("Linear SVM (`class_weight='balanced'`), calibrated for probability output")
    st.markdown("#### Features")
    st.markdown("TF-IDF, unigrams, lemmatized + stopword-filtered text")
    st.markdown("#### Test-set performance")
    st.markdown(
        """
        | Metric | Score |
        |---|---|
        | Accuracy | 97.8% |
        | Precision (spam) | 95.0% |
        | Recall (spam) | 87.0% |
        | F1 (spam) | 90.8% |
        """
    )
    st.markdown("---")
    st.caption("SMS Spam Collection Dataset · 5,169 messages (deduplicated)")

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown('<div class="hero-title">🛡️ SpamShield AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Paste a message below and get an instant spam/ham prediction with confidence.</div>',
    unsafe_allow_html=True,
)
st.write("")

# ----------------------------------------------------------------------
# Input area
# ----------------------------------------------------------------------
if "message_input" not in st.session_state:
    st.session_state.message_input = ""

st.markdown('<div class="glass-card">', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("📩 Try a spam example"):
        st.session_state.message_input = (
            "Congratulations! You've WON a $1000 Walmart gift card. "
            "Click here now to claim your free prize before it expires!"
        )
with col2:
    if st.button("💬 Try a ham example"):
        st.session_state.message_input = "Hey, are we still on for lunch tomorrow at 1?"
with col3:
    uploaded_file = st.file_uploader("Or upload a .txt file", type=["txt"], label_visibility="collapsed")
    if uploaded_file is not None:
        st.session_state.message_input = uploaded_file.read().decode("utf-8", errors="ignore")

message = st.text_area(
    "Message",
    key="message_input",
    height=140,
    placeholder="Paste an SMS or email message here...",
    label_visibility="collapsed",
)

analyze_clicked = st.button("🔍 Analyze Message", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Prediction + result display
# ----------------------------------------------------------------------
if analyze_clicked:
    if not message.strip():
        st.warning("Please enter a message or choose an example first.")
    else:
        with st.spinner("Analyzing message..."):
            time.sleep(0.3)  # brief pause so the spinner is visible; purely cosmetic
            label, confidence, ham_conf, spam_conf = predict(message)

        if label == "Spam":
            st.markdown(
                f"""
                <div class="result-spam">
                    <div style="font-size:2.2rem;">🚨</div>
                    <div class="result-label-spam">SPAM DETECTED</div>
                    <div style="color:#c9cbe0; margin-top:0.3rem;">
                        Confidence: {confidence*100:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption("⚠️ Avoid clicking links or sharing personal information from this message.")
        else:
            st.markdown(
                f"""
                <div class="result-ham">
                    <div style="font-size:2.2rem;">✅</div>
                    <div class="result-label-ham">LOOKS LEGITIMATE</div>
                    <div style="color:#c9cbe0; margin-top:0.3rem;">
                        Confidence: {confidence*100:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")
        st.markdown("**Probability breakdown**")
        st.progress(float(spam_conf), text=f"Spam: {spam_conf*100:.1f}%")
        st.progress(float(ham_conf), text=f"Ham: {ham_conf*100:.1f}%")

st.write("")
st.caption("Built with scikit-learn + Streamlit · SMS Spam Collection Dataset")