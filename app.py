import re
import pickle

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(
    page_title="Financial Phishing Email Detector",
    page_icon="📧",
    layout="centered",
)

# Must match the training notebook exactly
MAX_LEN = 200
THRESHOLD = 0.5


@st.cache_resource
def load_artifacts():
    model = load_model("gru_sentiment_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


model, tokenizer = load_artifacts()


# ---- Same cleaning function used during training ----
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)        # remove HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)     # keep only letters
    text = re.sub(r"\s+", " ", text).strip()  # collapse whitespace
    return text


def predict_phishing(subject, body):
    # The model was trained on: subject + " " + body, then cleaned.
    email_text = f"{subject} {body}"
    cleaned = clean_text(email_text)

    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")

    prob = float(model.predict(padded, verbose=0)[0][0])  # P(phishing)
    label = "Phishing" if prob > THRESHOLD else "Legitimate"
    return label, prob, cleaned


# ---------------- Streamlit UI ----------------
st.title("📧 Financial Phishing Email Detector")
st.write("Powered by a GRU neural network (99.47% test accuracy)")

subject = st.text_input(
    "Email subject:",
    placeholder="e.g. Urgent: verify your account details",
)
body = st.text_area(
    "Email body:",
    height=200,
    placeholder="Paste the full email body here...",
)

if st.button("Analyze Email"):
    if not subject.strip() and not body.strip():
        st.warning("Please enter an email subject or body first.")
    else:
        label, prob, cleaned = predict_phishing(subject, body)
        confidence = prob if label == "Phishing" else 1 - prob

        if label == "Phishing":
            st.error(f"⚠️ **{label}** (confidence: {confidence:.2%})")
        else:
            st.success(f"✅ **{label}** (confidence: {confidence:.2%})")

        st.progress(prob)
        st.caption(f"Raw model output (probability of phishing): {prob:.4f}")

        if not cleaned:
            st.info(
                "After cleaning, no usable text remained (the cleaner keeps letters only). "
                "The prediction above is not meaningful."
            )

        with st.expander("What the model actually saw (after cleaning)"):
            st.code(cleaned[:1000] or "(empty)", language=None)