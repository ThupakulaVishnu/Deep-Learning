import streamlit as st
import pickle
import numpy as np

# Inject custom CSS for gradient background, styled inputs, and button using colors from your image
st.markdown(
    """
    <style>
    /* Overall app background and global text color */
    .stApp {
        background: linear-gradient(135deg, #36d1dc, #5b86e5);

        color: #333333;
    }
    
    /* Custom header styling */
    .custom-header {
        font-size: 56px;
        font-weight: bold;
        color: #D72638;
        text-align: center;
        margin-bottom: 10px;
    }
    
    /* Input fields styling */
    .stTextInput>div>div>input {
        border-radius: 8px;
        padding: 10px;
        font-size: 16px;
        border: 1px solid #FF6F61;
    }
    
    /* Button styling */
    div.stButton > button {
        background: linear-gradient(45deg, #FF6F61, #FF914D);
        color: #1E3A8A !important;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px;
        font-size: 16px;
        border: none;
        margin-top: 20px;
        cursor: pointer;
    }
    div.stButton > button:hover {
        background: linear-gradient(45deg, #FF914D, #FF6F61);
    }
    
    /* Prediction/result box styling */
    .prediction-box {
        background: #28A745;
        color: white;
        font-size: 20px;
        font-weight: bold;
        padding: 10px;
        text-align: center;
        border-radius: 8px;
        margin-top: 15px;
        box-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

def load_models():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('label_encoder.pkl', 'rb') as file:
        label_encoder = pickle.load(file)
    with open('stand_scler.pkl', 'rb') as file:
        standard_scaler = pickle.load(file)
    return model, label_encoder, standard_scaler

def predict_price(data, model, label_encoder, standard_scaler):
    try:
        # Handle unseen school names in label_encoder
        if data[6] not in label_encoder.classes_:
            label_encoder.classes_ = np.append(label_encoder.classes_, data[6])
        data[6] = int(label_encoder.transform([data[6]]))
        
        # Convert list to numpy array and reshape
        data = np.array(data).reshape(1, -1)
        
        # Scale relevant features
        data[:, [1, 2, 3, 4, 6, 7]] = standard_scaler.transform(data[:, [1, 2, 3, 4, 6, 7]])
        
        return model.predict(data)[0]
    except:
        return "error"

# Display custom header
st.markdown("<div class='custom-header'>🏡 House Price Prediction</div>", unsafe_allow_html=True)

# (Optional) Feature names list for clarity
feature_names = [
    "GARAGE", "LAND_AREA", "FLOOR_AREA", "CBD_DIST", "NEAREST_STN_DIST", "POSTCODE",
    "NEAREST_SCH", "NEAREST_SCH_DIST", "Age", "No_of_rooms", "Sold_Year", "Sold_Month"
]

# Create two columns for input fields
col1, col2 = st.columns(2)
inputs = []

with col1:
    inputs.append(st.number_input("GARAGE 🚗", placeholder="Enter garage count", value=None, key="garage", step=1, format="%d"))
    inputs.append(st.number_input("LAND_AREA 🌱", placeholder="Enter land area", value=None, key="land_area", step=1, format="%d"))
    inputs.append(st.number_input("FLOOR_AREA 📐", placeholder="Enter floor area", value=None, key="floor_area", step=1, format="%d"))
    inputs.append(st.number_input("CBD_DIST 🌆", placeholder="Enter CBD distance", value=None, key="cbd_dist", step=0.1, format="%.2f"))
    inputs.append(st.number_input("NEAREST_STN_DIST🚉", placeholder="Enter nearest station distance", value=None, key="nearest_stn_dist", step=0.1, format="%.2f"))
    inputs.append(st.number_input("POSTCODE📮", placeholder="Enter postcode", value=None, key="postcode", step=1, format="%d"))

with col2:
    inputs.append(st.text_input("NEAREST_SCH 🏫", placeholder="Enter nearest school", key="nearest_sch"))
    inputs.append(st.number_input("NEAREST_SCH_DIST ✏️", placeholder="Enter nearest school distance", value=None, key="nearest_sch_dist", step=0.1, format="%.2f"))
    inputs.append(st.number_input("Age ⏳", placeholder="Enter property age", value=None, key="age", step=1, format="%d"))
    inputs.append(st.number_input("No_of_rooms 🏠", placeholder="Enter number of rooms", value=None, key="no_of_rooms", step=1, format="%d"))
    inputs.append(st.number_input("Sold_Year 📅", placeholder="Enter sold year", value=None, key="sold_year", step=1, format="%d"))
    inputs.append(st.number_input("Sold_Month 📅", placeholder="Enter sold month", value=None, key="sold_month", step=1, format="%d"))

# Prediction button and logic
if st.button("Predict Price"):
    # Ensure no input is missing
    if any(value == "" or value is None for value in inputs):
        st.error("Please fill all input values before predicting.")
    else:
        model, label_encoder, standard_scaler = load_models()
        prediction = predict_price(inputs, model, label_encoder, standard_scaler)
        
        if prediction == "error":
            st.error("Please enter valid data or check your inputs.")
        else:
            # Convert prediction from rupees to dollars using a conversion rate
            conversion_rate = 75  # Adjust this as needed
            dollar_value = prediction / conversion_rate
            
            # Display in two separate green boxes side by side
            st.markdown(f"""
            <div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
                <div class='prediction-box'>Predicted Price (in ₹): {prediction:,.2f}</div>
                <div class='prediction-box'>Predicted Price (in $): {dollar_value:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
