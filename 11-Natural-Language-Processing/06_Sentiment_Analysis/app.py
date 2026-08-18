import streamlit as st
import joblib
import re
import string
import nltk

from pathlib import Path
from bs4 import BeautifulSoup
from better_profanity import profanity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# =========================================================
# NLTK
# =========================================================

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


# =========================================================
# LOAD MODEL
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "sentiment_model.pkl")
vectorizer = joblib.load(BASE_DIR / "vectorizer.pkl")


# =========================================================
# PROFANITY
# =========================================================

profanity.load_censor_words()


# =========================================================
# STOPWORDS
# =========================================================

stop_words = set(stopwords.words("english"))

stop_words.discard("not")
stop_words.discard("no")
stop_words.discard("never")

lemmatizer = WordNetLemmatizer()


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="IBM Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    padding-top: 2rem;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #6b7280 !important;
    font-size: 17px;
    margin-bottom: 35px;
}

textarea {
    border-radius: 12px !important;
}

div.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 50px;
    font-size: 17px;
    font-weight: 600;
}

.results-title {
    font-size: 30px;
    font-weight: 700;
    margin-top: 35px;
    margin-bottom: 25px;
}

.result-card {
    border: 1px solid #e5e7eb;
    border-radius: 15px;
    padding: 25px;
    background-color: #ffffff !important;
    height: 150px;
    color: #111827 !important;
}

.result-label {
    color: #6b7280 !important;
    font-size: 15px;
    margin-bottom: 8px;
}

.result-value {
    font-size: 32px;
    font-weight: 700;
    color: #111827 !important;
}

.confidence-value {
    font-size: 36px;
    font-weight: 700;
    color: #111827 !important;
}

.score-header {
    display: grid;
    grid-template-columns: 1fr 2fr 1fr;
    background-color: #f8fafc !important;
    padding: 14px;
    border: 1px solid #e5e7eb;
    font-weight: 600;
    color: #374151 !important;
}

.score-row {
    display: grid;
    grid-template-columns: 1fr 2fr 1fr;
    padding: 15px;
    border-left: 1px solid #e5e7eb;
    border-right: 1px solid #e5e7eb;
    border-bottom: 1px solid #e5e7eb;
    color: #f9fafb !important;
}

.clear-button {
    text-align: center;
    padding: 15px;
    margin-top: 25px;
    border: 1px solid #d1d5db;
    border-radius: 12px;
    color: #f9fafb !important;
    font-weight: 500;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="title">🎬 IBM Movie Sentiment Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Analyze movie reviews and predict whether the sentiment is '
    'Positive or Negative using Machine Learning.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# MOVIE REVIEW INPUT
# =========================================================

review = st.text_area(
    "Enter your movie review",
    height=220,
    placeholder=(
        "Example: This movie was absolutely amazing! "
        "The acting was excellent and the story was fantastic."
    )
)


# =========================================================
# ANALYZE BUTTON
# =========================================================

if st.button("🔍 Analyze Review"):

    if not review.strip():

        st.warning("Please enter a movie review.")

    else:

        # =================================================
        # PROFANITY CHECK
        # =================================================

        if profanity.contains_profanity(review):

            st.error(
                "⚠️ Your review contains abusive or offensive "
                "language. Please use respectful language and "
                "try again."
            )

        else:

            # =================================================
            # PREPROCESSING
            # =================================================

            review = review.lower()

            review = BeautifulSoup(
                review,
                "html.parser"
            ).get_text()

            review = re.sub(
                r"http\S+|www\S+",
                "",
                review
            )

            review = review.translate(
                str.maketrans(
                    "",
                    "",
                    string.punctuation
                )
            )

            review = re.sub(
                r"\d+",
                "",
                review
            )

            review = " ".join(
                review.split()
            )

            words = review.split()

            words = [
                word
                for word in words
                if word not in stop_words
            ]

            words = [
                lemmatizer.lemmatize(word)
                for word in words
            ]

            review = " ".join(words)


            # =================================================
            # VECTORIZATION
            # =================================================

            review_vector = vectorizer.transform(
                [review]
            )


            # =================================================
            # PREDICTION
            # =================================================

            prediction = model.predict(
                review_vector
            )[0]


            # =================================================
            # PROBABILITIES
            # =================================================

            probabilities = model.predict_proba(
                review_vector
            )[0]

            classes = model.classes_

            probability_dict = {
                str(cls).lower(): probability
                for cls, probability in zip(
                    classes,
                    probabilities
                )
            }


            # =================================================
            # POSITIVE / NEGATIVE PROBABILITIES
            # =================================================

            positive_probability = 0
            negative_probability = 0

            for label, probability in probability_dict.items():

                if label in [
                    "positive",
                    "1",
                    "true"
                ]:

                    positive_probability = probability

                elif label in [
                    "negative",
                    "0",
                    "false"
                ]:

                    negative_probability = probability


            # =================================================
            # RESULT
            # =================================================

            prediction_text = str(
                prediction
            ).lower()

            if prediction_text in [
                "positive",
                "1",
                "true"
            ]:

                result = "POSITIVE"
                emoji = "😊"
                confidence = positive_probability * 100

            else:

                result = "NEGATIVE"
                emoji = "😞"
                confidence = negative_probability * 100


            # =================================================
            # RESULTS
            # =================================================

            st.markdown(
                '<div class="results-title">'
                'Analysis Results'
                '</div>',
                unsafe_allow_html=True
            )

            col1, col2 = st.columns(2)


            # =================================================
            # PREDICTION CARD
            # =================================================

            with col1:

                st.markdown(
                    '<div class="result-card">'
                    '<div class="result-label">'
                    'Predicted Sentiment'
                    '</div>'
                    f'<div class="result-value">'
                    f'{emoji} {result}'
                    '</div>'
                    '</div>',
                    unsafe_allow_html=True
                )


            # =================================================
            # CONFIDENCE CARD
            # =================================================

            with col2:

                st.markdown(
                    '<div class="result-card">'
                    '<div class="result-label">'
                    'Confidence Score'
                    '</div>'
                    f'<div class="confidence-value">'
                    f'{confidence:.2f}%'
                    '</div>'
                    '</div>',
                    unsafe_allow_html=True
                )


            # =================================================
            # DETAILED SCORES
            # =================================================

            st.markdown(
                '<div class="results-title" '
                'style="font-size:26px;">'
                'Detailed Scores'
                '</div>',
                unsafe_allow_html=True
            )


            # =================================================
            # HEADER
            # =================================================

            st.markdown(
                '<div class="score-header">'
                '<div>Type</div>'
                '<div>Sentiment</div>'
                '<div>Confidence</div>'
                '</div>',
                unsafe_allow_html=True
            )


            # =================================================
            # NEGATIVE
            # =================================================

            st.markdown(
                '<div class="score-row">'
                '<div>😞</div>'
                '<div>NEGATIVE</div>'
                f'<div>{negative_probability * 100:.2f}%</div>'
                '</div>',
                unsafe_allow_html=True
            )


            # =================================================
            # POSITIVE
            # =================================================

            st.markdown(
                '<div class="score-row">'
                '<div>😊</div>'
                '<div>POSITIVE</div>'
                f'<div>{positive_probability * 100:.2f}%</div>'
                '</div>',
                unsafe_allow_html=True
            )


            # =================================================
            # CLEAR MESSAGE
            # =================================================

            st.markdown(
                '<div class="clear-button">'
                '🔄 Enter another movie review '
                'to analyze its sentiment'
                '</div>',
                unsafe_allow_html=True
            )