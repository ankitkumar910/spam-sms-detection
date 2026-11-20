
import re
import joblib
import streamlit as st
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Load trained model and vectorizer
model_path = BASE_DIR / "spam_classifier_model.pkl"
vectorizer_path = BASE_DIR / "tfidf_vectorizer.pkl"

if not model_path.exists() or not vectorizer_path.exists():
    st.error("Model files not found. Please run main.py first.")
    st.stop()

model = joblib.load(model_path)
tfidf = joblib.load(vectorizer_path)


# Same preprocessing as main.py
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Prediction function
def predict_message(message):
    processed = preprocess_text(message)

    vectorized = tfidf.transform([processed])

    prediction = model.predict(vectorized)[0]

    probabilities = model.predict_proba(vectorized)[0]

    # Find probability of predicted class
    class_index = list(model.classes_).index(prediction)
    confidence = probabilities[class_index]

    prediction_text = str(prediction).strip().lower()
    result = "Spam" if prediction_text in {"1", "spam"} else "Not Spam"

    return result, confidence


# Streamlit UI
st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📩",
    layout="centered"
)

st.title("📩 SMS Spam Classifier")

st.write(
    "Enter an SMS message below to check whether "
    "it is Spam or Not Spam."
)

user_input = st.text_area(
    "Enter your SMS message:",
    height=150,
    placeholder="Example: Congratulations! You have won a free prize..."
)

if st.button("Predict", use_container_width=True):

    if not user_input.strip():
        st.warning("Please enter a message first.")

    else:
        result, confidence = predict_message(user_input)

        st.subheader("Prediction")

        if result == "Spam":
            st.error(f"🚫 This message is SPAM!")

        else:
            st.success(f"✅ This message is NOT SPAM!")

        st.metric(
            "Prediction Confidence",
            f"{confidence * 100:.2f}%"
        )

st.divider()

st.caption("Built using Python, Scikit-learn and Streamlit.")
