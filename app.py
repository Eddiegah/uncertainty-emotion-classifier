import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from predict import predict_with_uncertainty, load_model

# ── Emotion emojis ──────────────────────────────────────
EMOTION_EMOJI = {
    "joy":      "😊",
    "sadness":  "😢",
    "anger":    "😠",
    "fear":     "😨",
    "love":     "❤️",
    "surprise": "😲"
}

EMOTION_COLOR = {
    "joy":      "#FFD700",
    "sadness":  "#4A90D9",
    "anger":    "#E74C3C",
    "fear":     "#8E44AD",
    "love":     "#E91E8C",
    "surprise": "#2ECC71"
}

# ── Page config ─────────────────────────────────────────
st.set_page_config(
    page_title="Emotion Classifier",
    page_icon="🧠",
    layout="centered"
)

# ── Title ───────────────────────────────────────────────
st.title("🧠 Uncertainty-Aware Emotion Classifier")
st.markdown("Type any sentence and the model will predict its emotion — and tell you **how confident** and **how uncertain** it is.")
st.markdown("---")

# ── Load model (cached so it only loads once) ───────────
@st.cache_resource
def get_model():
    return load_model()

with st.spinner("Loading model... (first time only, ~10 seconds)"):
    model, tokenizer = get_model()

# ── Input ───────────────────────────────────────────────
st.subheader("Enter your text")
user_input = st.text_area(
    label="",
    placeholder="e.g. I am so happy today, everything is going great!",
    height=100
)

run_button = st.button("Analyse Emotion", type="primary")

# ── Prediction ──────────────────────────────────────────
if run_button:
    if not user_input.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Running prediction..."):
            result = predict_with_uncertainty(user_input, model, tokenizer)

        emotion    = result["predicted_emotion"]
        confidence = result["confidence_pct"]
        uncertainty = result["uncertainty_pct"]
        all_probs  = result["all_emotion_probs"]
        emoji      = EMOTION_EMOJI[emotion]
        color      = EMOTION_COLOR[emotion]

        st.markdown("---")

        # ── Main result ─────────────────────────────────
        st.markdown(f"## {emoji} {emotion.upper()}")

        # Confidence bar
        st.markdown("**Confidence**")
        st.progress(int(confidence))
        st.markdown(f"`{confidence}%`")

        # Uncertainty bar
        st.markdown("**Uncertainty**")
        st.progress(int(uncertainty))
        st.markdown(f"`{uncertainty}%` — {'⚠️ High uncertainty: model is unsure' if uncertainty > 40 else '✅ Low uncertainty: model is confident'}")

        st.markdown("---")

        # ── All emotion probabilities chart ─────────────
        st.subheader("All Emotion Probabilities")

        emotions = list(all_probs.keys())
        probs    = list(all_probs.values())
        colors   = [EMOTION_COLOR[e] for e in emotions]

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(emotions, probs, color=colors, edgecolor="white")
        ax.set_xlabel("Probability (%)", fontsize=12)
        ax.set_xlim(0, 100)
        ax.set_title("Emotion Probability Distribution", fontsize=13)

        for bar, prob in zip(bars, probs):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f"{prob}%", va="center", fontsize=10)

        fig.patch.set_facecolor("#0E1117")
        ax.set_facecolor("#0E1117")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.title.set_color("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#444")
        ax.spines["bottom"].set_color("#444")

        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("---")
        st.caption("Model: DistilBERT fine-tuned on dair-ai/emotion · Uncertainty via MC Dropout (20 forward passes)")