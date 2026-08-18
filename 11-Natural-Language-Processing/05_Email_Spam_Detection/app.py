import streamlit as st
import joblib
from pathlib import Path

# Get the folder containing app.py
BASE_DIR = Path(__file__).resolve().parent

# Load trained model and TF-IDF vectorizer
model = joblib.load(BASE_DIR / "spam_model.pkl")
vectorizer = joblib.load(BASE_DIR / "tfidf_vectorizer.pkl")

# Page configuration
st.set_page_config(
    page_title="Spam Email Detector",
    page_icon="📧",
    layout="centered"
)

# Title
st.title("📧 Spam Email Detector")

st.write(
    "Enter an email message below and the model will "
    "predict whether it is spam or not spam."
)

# Email input
email = st.text_area(
    "Enter your email:",
    height=250,
    placeholder="Paste your email message here..."
)

# Prediction
if st.button("🔍 Check Email"):

    if email.strip() == "":
        st.warning("Please enter an email message.")

    else:
        # Convert email into TF-IDF features
        email_vector = vectorizer.transform([email])

        # Make prediction
        prediction = model.predict(email_vector)[0]

        # Get prediction probabilities
        probability = model.predict_proba(email_vector)[0]

        # Calculate confidence
        confidence = max(probability) * 100

        # Convert prediction to string
        prediction = str(prediction).lower()

        # Display result
        if prediction == "spam":
            st.error("🚨 SPAM EMAIL")
        else:
            st.success("✅ NOT SPAM")

        st.write(f"Confidence: **{confidence:.2f}%**")