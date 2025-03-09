import streamlit as st
import re
import nltk
import spacy
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download NLTK stopwords if not already downloaded
nltk.download("stopwords")
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

# Load spaCy model fresh (do not pickle the spaCy object)
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# Load preprocessing objects (tokenizer, max_length, etc.) from pickle file (saved during training)
with open("preprocessing.pkl", "rb") as f:
    preprocessing = pickle.load(f)

# Retrieve tokenizer and max_length from the pickle file
tokenizer = preprocessing['tokenizer']
max_length = preprocessing['max_length']

# Load the trained model
model = load_model("spam_mail_model.h5")

# Preprocessing function: lowercasing, regex cleaning, extra space removal, stopword removal
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s!?]", " ", text)  # Remove numbers & special characters (keeping letters, spaces, !, ?)
    text = re.sub(r"\s+", " ", text).strip()   # Remove extra spaces
    text = " ".join(word for word in text.split() if word not in stop_words)  # Remove stopwords
    return text

# Lemmatization function using spaCy
def lemmatize_text(text):
    doc = nlp(text)
    return " ".join(token.lemma_ for token in doc)

# Complete prediction function
def predict_spam(text):
    processed_text = preprocess_text(text)
    lemmatized_text = lemmatize_text(processed_text)
    seq = tokenizer.texts_to_sequences([lemmatized_text])
    padded_seq = pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')
    prediction_prob = model.predict(padded_seq)[0][0]
    return "Spam" if prediction_prob > 0.5 else "Ham"

# 1) Use layout="wide" to allow full-width content
st.set_page_config(page_title="Email Spam Detector", page_icon="📧", layout="wide")

# 2) Add CSS to stretch the main container and text area
st.markdown(
    """
    <style>
    /* Make the main container take full width */
    main .block-container {
        max-width: 100% !important;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    /* Full page background color */
    .stApp {
        background-color: #87CEEB; /* Sky blue background */
    }
    /* Title styling */
    .title {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        color: #FF7F50;
        margin-bottom: 20px;
    }
    /* Force the text area to take full width */
    .stTextArea textarea {
        width: 100% !important;
        min-height: 350px;
        background-color: #ffffff;
        color: black;
        border-radius: 10px;
        padding: 10px;
        font-size: 25px;
    }
    /* Button styling */
    .stButton>button {
        background-color: #9932CC;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        padding: 10px 20px;
        border: none;
        cursor: pointer;
    }
    /* Button hover effect */
    .stButton>button:hover {
        background-color: #9932CC;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App Title
st.markdown("<h1 class='title'>📧 Spam Email Detector 📧</h1>", unsafe_allow_html=True)

st.markdown("<h3 style='text-align: left; color: #000;'>📝 Enter your email content:</h3>", unsafe_allow_html=True)
email_text = st.text_area("", height=300)

if st.button("🔎 Check Spam"):
    if email_text.strip():
        result = predict_spam(email_text)
        if result == "Spam":
            st.markdown("<div style='background-color: #C00000; color: black; padding: 10px; border-radius: 10px; text-align: center; width: 50%; margin: auto;'>🚨 This email is classified as SPAM!</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='background-color: #4CAF50; color: black; padding: 10px; border-radius: 10px; text-align: center; width: 50%; margin: auto;'>✅ This email is classified as HAM (Not Spam)</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='background-color: #FF8C00; color: black; padding: 10px; border-radius: 10px; text-align: center; width: 50%; margin: auto;'>⚠️ Please enter an email to check.</div>", unsafe_allow_html=True)
