import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

from tensorflow.keras.models import load_model

# Loading an train model
model = load_model(r"F:\Krish naik\9-Deep Learning\1)ANN\ANN_Classification_Churn\model.h5")

# Load the encoder and scaler
u = r"F:\Krish naik\9-Deep Learning\1)ANN\ANN_Classification_Churn\FScaler.pkl"
with open(u, 'rb') as file:
    Fscaler = pickle.load(file)

u = r"F:\Krish naik\9-Deep Learning\1)ANN\ANN_Classification_Churn\label_enocoder_gender.pkl"
with open(u, 'rb') as file:
    label_encoder = pickle.load(file)

u = r"F:\Krish naik\9-Deep Learning\1)ANN\ANN_Classification_Churn\onehotEncoder.pkl"
with open(u, 'rb') as file:
    onehotencoder = pickle.load(file)

## Streamlit app
st.title("📊 Customer Churn Prediction")
st.markdown("##### 💡 Leveraging AI to keep your customers happy and engaged.")

# User input
geography = st.selectbox("🌍 Geography", onehotencoder.categories_[0])
gender = st.selectbox("👤 Gender", label_encoder.classes_)
age = st.slider("🎂 Age", 18, 92)
balance = st.number_input("💰 Balance")
credit_score = st.number_input("📈 Credit Score")
estimated_salary = st.number_input("💵 Estimated Salary")
tenure = st.slider("⏳ Tenure", 0, 10)
num_of_products = st.slider("🛍️ Number of Products", 1, 4)
has_cr_card = st.selectbox("💳 Has Credit Card", [0, 1])
is_active_member = st.selectbox("🟢 Is Active Member", [0, 1])

# Prepare the input data
input_data = {
    'CreditScore': credit_score,
    'Gender': label_encoder.transform([gender])[0],
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary
}

h = onehotencoder.transform([[geography]]).toarray()
da = pd.DataFrame(h, columns=onehotencoder.get_feature_names_out(['Geography']))

data = pd.DataFrame([input_data])

u = data.iloc[:, 0]
v = pd.concat([u, da], axis=1)
data = pd.concat([v, data.iloc[:, 1:]], axis=1)

data = Fscaler.transform(data)

# Prediction
predit = model.predict(data)
predict_prob = predit[0][0]

if predict_prob > 0.5:
    st.success(f"✅ Churn Probability: {predict_prob:.2f}")
else:
    st.error(f"❌ Churn Probability: {predict_prob:.2f}")

if predict_prob > 0.5:
    st.success(" 🟢 The customer is likely to churn.")
else:
    st.error("🔴 The customer is not likely to churn.")
