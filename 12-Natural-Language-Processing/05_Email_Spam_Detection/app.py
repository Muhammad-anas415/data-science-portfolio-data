import streamlit as st
import joblib


# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


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


# Text input
email = st.text_area(
    "Enter your email:",
    height=250,
    placeholder="Paste your email message here..."
)


# Prediction button
if st.button("🔍 Check Email"):

    if email.strip() == "":
        st.warning("Please enter an email message.")

    else:

        # Transform email using trained TF-IDF
        email_vector = vectorizer.transform([email])

        # Prediction
        prediction = model.predict(email_vector)[0]

        # Probability
        probability = model.predict_proba(email_vector)[0]

        confidence = max(probability) * 100


        # Display result
        if prediction.lower() == "spam":

            st.error("🚨 SPAM EMAIL")

            st.write(
                f"Confidence: **{confidence:.2f}%**"
            )

        else:

            st.success("✅ NOT SPAM")

            st.write(
                f"Confidence: **{confidence:.2f}%**"
            )