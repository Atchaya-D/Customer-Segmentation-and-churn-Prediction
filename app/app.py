import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_prediction_model.pkl"

model = joblib.load(MODEL_PATH)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🏦 Bank Customer Analytics")
st.sidebar.markdown("Machine Learning Project")
st.sidebar.markdown("---")

# -----------------------------
# Title
# -----------------------------
st.title("🏦 Customer Churn Prediction Dashboard")
st.markdown("Predict whether a customer is likely to leave the bank using a trained Random Forest model.")

st.markdown("---")

# -----------------------------
# Customer Information
# -----------------------------
st.subheader("Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    customer_id = st.number_input("Customer ID", value=15634602)
    credit_score = st.number_input("Credit Score", 300, 900, 650)
    country = st.selectbox("Country", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.slider("Age", 18, 90, 35)
    tenure = st.slider("Tenure", 0, 10, 5)

with col2:
    balance = st.number_input("Balance", value=50000.0)
    products_number = st.slider("Number of Products", 1, 4, 2)
    credit_card = st.selectbox("Has Credit Card", [0, 1])
    active_member = st.selectbox("Is Active Member", [0, 1])
    estimated_salary = st.number_input("Estimated Salary", value=80000.0)

# -----------------------------
# Encoding
# -----------------------------
country_map = {
    "France": 0,
    "Germany": 1,
    "Spain": 2
}

gender_map = {
    "Female": 0,
    "Male": 1
}

# Default values
cluster = 0
customer_segment = 0

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Churn"):

    input_df = pd.DataFrame({
        "customer_id": [customer_id],
        "credit_score": [credit_score],
        "country": [country_map[country]],
        "gender": [gender_map[gender]],
        "age": [age],
        "tenure": [tenure],
        "balance": [balance],
        "products_number": [products_number],
        "credit_card": [credit_card],
        "active_member": [active_member],
        "estimated_salary": [estimated_salary],
        "Cluster": [cluster],
        "Customer Segment": [customer_segment]
    })

    prediction = model.predict(input_df)[0]

    st.markdown("---")

    if prediction == 1:
        st.error("⚠️ Prediction: Customer is likely to Churn")
    else:
        st.success("✅ Prediction: Customer is likely to Stay")

    st.subheader("Customer Details")

    st.dataframe(input_df)

st.markdown("---")

st.markdown(
"""
### 📌 About this Project

This application predicts whether a bank customer is likely to leave the bank based on customer demographics and banking information.

**Machine Learning Model:** Random Forest Classifier

**Accuracy:** **86.35%**

Developed using:
- Python
- Streamlit
- Pandas
- Scikit-Learn
"""
)