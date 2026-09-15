import pickle
import numpy as np
import pandas as pd
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="Customer Predictor",
    page_icon="🎯",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS for Attractive UI with Shadow Effects
st.markdown("""
    <style>
    /* Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }

    /* Main Container Card with Drop Shadow */
    .main-card {
        background: #ffffff;
        padding: 30px;
        border-radius: 16px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.12);
        margin-bottom: 25px;
    }

    /* Title Styling */
    .app-title {
        color: #1e293b;
        text-align: center;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .app-subtitle {
        color: #64748b;
        text-align: center;
        font-size: 1rem;
        margin-bottom: 25px;
    }

    /* Custom Shadow for Predict Button */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #4f46e5 0%, #3b82f6 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 28px;
        font-size: 1.1rem;
        font-weight: 600;
        width: 100%;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.4);
        transition: all 0.3s ease;
    }

    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.6);
        background: linear-gradient(90deg, #4338ca 0%, #2563eb 100%);
    }

    /* Result Card Styling */
    .result-card-yes {
        background-color: #dcfce7;
        border-left: 6px solid #16a34a;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(22, 163, 74, 0.15);
        text-align: center;
        color: #14532d;
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: 20px;
    }

    .result-card-no {
        background-color: #fee2e2;
        border-left: 6px solid #dc2626;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.15);
        text-align: center;
        color: #7f1d1d;
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)


# 3. Model Loading Function
@st.cache_resource
def load_model():
    with open("random.pkl", "rb") as file:
        model = pickle.load(file)
    return model


model = load_model()

# Header Section
st.markdown('<div class="app-title">Customer Response Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Enter customer details below to predict likelihood</div>',
            unsafe_allow_html=True)

# Form Section
with st.container():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        gender = st.selectbox("Gender", options=["Male", "Female"])
        marital_status = st.selectbox("Marital Status", options=["Single", "Married", "Prefer not to say"])
        occupation = st.selectbox("Occupation", options=["Student", "Employee", "Self Employed", "Housewife"])

    with col2:
        monthly_income = st.selectbox(
            "Monthly Income",
            options=["No Income", "Below Rs.10000", "10001 to 25000", "25001 to 50000", "More than 50000"]
        )
        qualification = st.selectbox(
            "Educational Qualifications",
            options=["Uneducated", "School", "Graduate", "Post Graduate", "Ph.D"]
        )
        family_size = st.number_input("Family Size", min_value=1, max_value=20, value=3, step=1)
        customer_type = st.selectbox("Customer Type", options=["New", "Existing"])

    st.markdown('</div>', unsafe_allow_html=True)

# Map string categorical inputs to standard numerical representations for model inference
gender_map = {"Male": 0, "Female": 1}
marital_map = {"Single": 0, "Married": 1, "Prefer not to say": 2}
occupation_map = {"Student": 0, "Employee": 1, "Self Employed": 2, "Housewife": 3}
income_map = {"No Income": 0, "Below Rs.10000": 1, "10001 to 25000": 2, "25001 to 50000": 3, "More than 50000": 4}
qual_map = {"Uneducated": 0, "School": 1, "Graduate": 2, "Post Graduate": 3, "Ph.D": 4}
cust_map = {"New": 0, "Existing": 1}

# Prediction Trigger
if st.button("Predict Target Class"):
    try:
        # Prepare Feature Vector matching `feature_names_in_`:
        # ['Age', 'Gender', 'Marital Status', 'Occupation', 'Monthly Income', 'Educational Qualifications', 'Family size', 'Customer Type']
        features = np.array([[
            age,
            gender_map[gender],
            marital_map[marital_status],
            occupation_map[occupation],
            income_map[monthly_income],
            qual_map[qualification],
            family_size,
            cust_map[customer_type]
        ]])

        prediction = model.predict(features)[0]

        if prediction == "Yes" or prediction == 1:
            st.markdown(
                '<div class="result-card-yes">🎉 Prediction: Positive Response (Yes)</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-card-no">⚠️ Prediction: Negative Response (No)</div>',
                unsafe_allow_html=True
            )

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
