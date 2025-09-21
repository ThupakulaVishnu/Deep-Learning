# import streamlit as st
# import re
# import nltk
# import spacy
# import pickle
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# # Download NLTK stopwords if not already downloaded
# nltk.download("stopwords")
# from nltk.corpus import stopwords
# stop_words = set(stopwords.words("english"))

# # Load spaCy model fresh (do not pickle the spaCy object)
# nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# # Load preprocessing objects (tokenizer, max_length, etc.) from pickle file (saved during training)
# with open("preprocessing.pkl", "rb") as f:
#     preprocessing = pickle.load(f)

# # Retrieve tokenizer and max_length from the pickle file
# tokenizer = preprocessing['tokenizer']
# max_length = preprocessing['max_length']

# # Load the trained model
# model = load_model("spam_mail_model.h5")

# # Preprocessing function: lowercasing, regex cleaning, extra space removal, stopword removal
# def preprocess_text(text):
#     text = text.lower()
#     text = re.sub(r"[^a-z\s!?]", " ", text)  # Remove numbers & special characters (keeping letters, spaces, !, ?)
#     text = re.sub(r"\s+", " ", text).strip()   # Remove extra spaces
#     text = " ".join(word for word in text.split() if word not in stop_words)  # Remove stopwords
#     return text

# # Lemmatization function using spaCy
# def lemmatize_text(text):
#     doc = nlp(text)
#     return " ".join(token.lemma_ for token in doc)

# # Complete prediction function
# def predict_spam(text):
#     processed_text = preprocess_text(text)
#     lemmatized_text = lemmatize_text(processed_text)
#     seq = tokenizer.texts_to_sequences([lemmatized_text])
#     padded_seq = pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')
#     prediction_prob = model.predict(padded_seq)[0][0]
#     return "Spam" if prediction_prob > 0.5 else "Ham"

# # 1) Use layout="wide" to allow full-width content
# st.set_page_config(page_title="Spam Email Detection", page_icon="📧", layout="wide")

# # 2) Add CSS to stretch the main container and text area
# st.markdown(
#     """
#     <style>
#     /* Make the main container take full width */
#     main .block-container {
#         max-width: 100% !important;
#         padding-left: 1rem;
#         padding-right: 1rem;
#     }
#     /* Full page background color */
#     .stApp {
#         background-color: #87CEEB; /* Sky blue background */
#     }
#     /* Title styling */
#     .title {
#         font-size: 36px;
#         font-weight: bold;
#         text-align: center;
#         color: #FF7F50;
#         margin-bottom: 20px;
#     }
#     /* Force the text area to take full width */
#     .stTextArea textarea {
#         width: 100% !important;
#         min-height: 350px;
#         background-color: #ffffff;
#         color: black;
#         border-radius: 10px;
#         padding: 10px;
#         font-size: 25px;
#     }
#     /* Button styling */
#     .stButton>button {
#         background-color: #9932CC;
#         color: white;
#         font-size: 50px;
#         font-weight: bold;
#         border-radius: 15px;
#         padding: 15px 30px;
#         border: none;
#         cursor: pointer;
#         transform: scale(1.2);
#     }
#     /* Button hover effect */
#     .stButton>button:hover {
#         background-color: #8000A0;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # App Title
# st.markdown("<h1 class='title'><b>📧 Spam Email Detector 📧</b></h1>", unsafe_allow_html=True)

# st.markdown("<h3 style='text-align: left; color: #000;'>📝 Enter your email content:</h3>", unsafe_allow_html=True)
# email_text = st.text_area("", height=300)

# if st.button("🔎 Check Spam"):
#     if email_text.strip():
#         result = predict_spam(email_text)
#         if result == "Spam":
#             st.markdown("<div style='background-color: #f01000; color: black; padding: 25px; border-radius: 10px; text-align: center; width: 50%; margin: auto;'><h2>🚨 This email is classified as SPAM!</h2></div>", unsafe_allow_html=True)
#         else:
#             st.markdown("<div style='background-color: #28A745; color: black; padding: 25px; border-radius: 10px; text-align: center; width: 50%; margin: auto;'><h2>✅ This email is classified as HAM (Not Spam)</h2></div>", unsafe_allow_html=True)
#     else:
#         st.markdown("<div style='background-color: #FF8C00; color: black; padding: 25px; border-radius: 10px; text-align: center; width: 50%; margin: auto;'><h2>⚠️ Please enter an email to check.</h2></div>", unsafe_allow_html=True)


import streamlit as st
import re
import pickle
import numpy as np
import spacy
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load stopwords
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

# Load preprocessing objects
with open("preprocessing.pkl", "rb") as f:
    preprocessing = pickle.load(f)

word2vec_model = preprocessing['word2vec_model']
nlp = preprocessing['nlp']
max_length = preprocessing['max_length']

# Load trained model
model = load_model("spam_mail_model.h5")

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s!?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text

# Lemmatize text
def lemmatize_text(text):
    doc = nlp(text)
    return " ".join(token.lemma_ for token in doc)

# Convert text to Word2Vec vectors
def text_to_vectors(text):
    words = text.split()
    vecs = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
    if len(vecs) < max_length:
        vecs.extend([[0]*word2vec_model.vector_size]*(max_length - len(vecs)))  # padding
    else:
        vecs = vecs[:max_length]
    return np.array([vecs])  # add batch dimension

# Prediction function
def predict_spam(text):
    processed = preprocess_text(text)
    lemmatized = lemmatize_text(processed)
    vectors = text_to_vectors(lemmatized)
    pred_prob = model.predict(vectors)[0][0]
    return "Spam" if pred_prob > 0.5 else "Ham"

# Streamlit app UI
st.set_page_config(page_title="Spam Email Detection", page_icon="📧", layout="wide")
st.markdown("<h1 style='text-align:center; color:#FF7F50;'>📧 Spam Email Detector 📧</h1>", unsafe_allow_html=True)

email_text = st.text_area("📝 Enter your email content:", height=300)

if st.button("🔎 Check Spam"):
    if email_text.strip():
        result = predict_spam(email_text)
        if result == "Spam":
            st.markdown("<div style='background-color:#f01000; color:black; padding:25px; border-radius:10px; text-align:center;'><h2>🚨 This email is classified as SPAM!</h2></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='background-color:#28A745; color:black; padding:25px; border-radius:10px; text-align:center;'><h2>✅ This email is classified as HAM (Not Spam)</h2></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='background-color:#FF8C00; color:black; padding:25px; border-radius:10px; text-align:center;'><h2>⚠️ Please enter an email to check.</h2></div>", unsafe_allow_html=True)
