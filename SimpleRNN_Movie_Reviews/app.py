import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Mapping of word index to word
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with Relu activation
model = load_model('simple_rnn_imdb.h5')

# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3, '?') for i in encoded_review])

# Function to preprocess user input
def prediction_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]  # Default to index 2 if the word is not found
    encoded_review = [min(i, 9999) for i in encoded_review]  # Make sure the index doesn't exceed 9999
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Prediction function
def prediction(review):
    pre_input = prediction_text(review)
    predict = model.predict(pre_input)
    sentiment = 'Positive' if predict[0][0] > 0.6 else 'Negative'
    return sentiment, predict[0][0]

# Streamlit app
st.set_page_config(page_title='IMDB Movie Review Sentiment Analysis', page_icon='🎬', layout='wide')

st.title('🎬 IMDB Movie Review Sentiment Analysis')
st.markdown("""
    <style>
        .main { 
            background-color: #f0f8ff;
            color: #1a1a1a;
            padding: 20px;
            font-family: 'Arial', sans-serif;
        }
        .title {
            font-size: 36px;
            font-weight: bold;
            color: #2d87f0;
        }
        .positive {
            background-color: #28a745;
            color: white;
            padding: 10px;
            border-radius: 10px;
        }
        .negative {
            background-color: #dc3545;
            color: white;
            padding: 10px;
            border-radius: 10px;
        }
        .box {
            background-color: #e9ecef;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .emoji {
            font-size: 48px;
        }
    </style>
""", unsafe_allow_html=True)

# Layout with columns
col1, col2 = st.columns([2, 3])

with col1:
    st.markdown("<div class='emoji'>🎥</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<h2 class='title'>Classify Movie Reviews</h2>", unsafe_allow_html=True)
    st.write("Enter a movie review, and I will tell you if it's positive or negative! 🍿")

# User input
user_input = st.text_area("Movie Review", height=200)

# Display message for better user experience
if user_input:
    st.write("Review is being processed... Please wait!")
else:
    st.write("Enter a movie review to begin.")

# Button to classify the review
if st.button('Classify'):
    prediction_input = prediction_text(user_input)

    # Make Prediction
    predict = model.predict(prediction_input)
    sentiment = 'Positive' if predict[0][0] > 0.6 else 'Negative'
    
    # Display the result with dynamic colors and emoji
    if sentiment == 'Positive':
        st.markdown(f"<div class='positive'>🌟 Sentiment: {sentiment} 🎉</div>", unsafe_allow_html=True)
        st.write(f"Prediction Score: {predict[0][0]:.2f}")
    else:
        st.markdown(f"<div class='negative'>💔 Sentiment: {sentiment} 😞</div>", unsafe_allow_html=True)
        st.write(f"Prediction Score: {predict[0][0]:.2f}")

else:
    st.write('Please enter a movie review for classification.')

# Add a footer message with some information
st.markdown("""
    <footer style="text-align:center; padding:10px; font-size:14px;">
        <p>Powered by TensorFlow and Streamlit 🍿🎬</p>
    </footer>
""", unsafe_allow_html=True)
