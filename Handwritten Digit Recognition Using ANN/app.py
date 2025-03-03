import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Handwritten Digit Recognition", layout="wide")
st.markdown("""
<style>
html, body {
    overflow: hidden; 
    height: 100vh; 
    margin: 0; 
    padding: 0;
}
.stApp {
    height: 100vh; 
    display: flex; 
    flex-direction: column; 
    justify-content: center; 
    align-items: center; 
    background-color: #f0f2f6;
}
.header {
    background-color: #2980B9; 
    width: 100%; 
    padding: 10px; 
    text-align: center; 
    color: white; 
    font-size: 28px; 
    border-radius: 0 0 10px 10px;
}
.main {
    width: 100%; 
    display: flex; 
    justify-content: space-around; 
    align-items: center; 
    flex-wrap: wrap;
}
.upload-section {
    background-color: #3498DB; 
    padding: 10px; 
    border-radius: 10px; 
    box-shadow: 1px 1px 5px rgba(0,0,0,0.1); 
    margin: 10px; 
    text-align: center; 
    flex: 1; 
    max-width: 45%;
    color: white;
}
.prediction-section {
    background-color: #27AE60; 
    padding: 10px; 
    border-radius: 10px; 
    box-shadow: 1px 1px 5px rgba(0,0,0,0.1); 
    margin: 10px; 
    text-align: center; 
    flex: 1; 
    max-width: 45%;
    color: white;
}
.prediction-result {
    background-color: #D4EDDA; /* Light Green Background */
    padding: 15px; 
    border-radius: 10px; 
    color: #155724; /* Dark Green Text */
    font-size: 36px; 
    font-weight: bold; 
    margin-top: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


st.markdown("<div class='header'>Handwritten Digit Recognition (MNIST)</div>", unsafe_allow_html=True)
file = st.file_uploader("", type=["png", "jpg", "jpeg"])
st.markdown("<div class='main'>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='upload-section'>Upload Image</div>", unsafe_allow_html=True)
    if file is not None:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            if np.mean(img) > 128:
                img = 255 - img
            original = img.copy()
            st.image(original, caption="Uploaded Image", width=150)
            img = cv2.resize(img, (28, 28))
            img = img.astype("float32") / 255.0
            img = np.expand_dims(img, axis=0)
            model = load_model('model.h5')
            pred = model.predict(img)
            digit = np.argmax(pred, axis=1)[0]
        else:
            st.error("Error reading image.")
    else:
        st.markdown("<div class='prediction-result'>-</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='prediction-section'>Prediction</div>", unsafe_allow_html=True)
    if file is not None and 'img' in locals() and img is not None:
        st.markdown(f"<div class='prediction-result'>{digit}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='prediction-result'>-</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
