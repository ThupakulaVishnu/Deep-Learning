import pickle

u=r"F:\Krish naik\9-Deep Learning\1)ANN\ANN_Regresstion_churn\Label_encoder.pkl"
with open(u,'rb') as file:
    le=pickle.load(file)

v=r"F:\Krish naik\9-Deep Learning\1)ANN\ANN_Regresstion_churn\onehotencoder.pkl"
with open(v,'rb') as file:
    ohe=pickle.load(file)

w=r"F:\Krish naik\9-Deep Learning\1)ANN\ANN_Regresstion_churn\scaler.pkl"
with open(w,'rb') as file:
    scle=pickle.load(file)

import tensorflow as tf
model=tf.keras.models.load_model(r"F:\Krish naik\9-Deep Learning\1)ANN\ANN_Regresstion_churn\Regression.h5")

## Streamlit app
import streamlit as st
st.title("📊 Estimated Salary Prediction")

# User input
geography = st.selectbox("🌍 Geography", ohe.categories_[0])
gender = st.selectbox("👤 Gender", le.classes_)
age = st.slider("🎂 Age", 18, 92)
balance = st.number_input("💰 Balance")
credit_score = st.number_input("📈 Credit Score")
exited = st.selectbox("🚪 Exited", [0, 1])
tenure = st.slider("⏳ Tenure", 0, 10)
num_of_products = st.slider("🛍️ Number of Products", 1, 4)
has_cr_card = st.selectbox("💳 Has Credit Card", [0, 1])
is_active_member = st.selectbox("🟢 Is Active Member", [0, 1])

# Prepare the input data
input_data = {
    'CreditScore': credit_score,
    'Gender': le.transform([gender])[0],
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member,
    'Exited':exited
}

import pandas as pd
h = ohe.transform([[geography]]).toarray()
da = pd.DataFrame(h, columns=ohe.get_feature_names_out(['Geography']))

data = pd.DataFrame([input_data])

u = data.iloc[:, 0]
v = pd.concat([u, da], axis=1)
data = pd.concat([v, data.iloc[:, 1:]], axis=1)



data = scle.transform(data.iloc[:,:].values)
print(data[:2,:])
# Prediction
predict=model.predict(data)
predict_salary=predict[0][0]

st.markdown(f"Prediction Estimated Salary : Rs.{predict_salary:.2f}")