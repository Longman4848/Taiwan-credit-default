import streamlit as st
import requests
import pandas as pd
import os

# --- CONFIGURATION ---
st.set_page_config(
    page_title="CreditGuard | Risk Analysis",
    page_icon="🛡️",
    layout="wide"
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
API_ENDPOINT = f"{BACKEND_URL}/predict"

def check_api_health(url):
    try:
        response = requests.get(f"{url}/health", timeout=3)
        return response.json()
    except:
        return None

# --- STYLING ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; font-weight: bold; }
    .status-box { padding: 20px; border-radius: 10px; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- HELPER DATA ---
FEATURE_LABELS = {
    "X1": "Credit Limit (NT$)", "X2": "Gender", "X3": "Education", 
    "X4": "Marital Status", "X5": "Age", "X6": "Sept Status",
    "X7": "Aug Status", "X8": "July Status", "X9": "June Status",
    "X10": "May Status", "X11": "April Status"
}

# --- SIDEBAR: HEALTH CHECK ---
with st.sidebar:
    st.title("System Status")
    health = check_api_health(BACKEND_URL)
    
    if health and health.get("status") == "healthy":
        st.success("API: Connected")
        st.caption(f"Model: {health.get('model_loaded', 'XGBoost')}")
    else:
        st.error(" API: Offline")
        st.info(f"Connecting to: {BACKEND_URL}") # Helps you debug in AWS
    st.divider()
    st.info("Taiwan Credit Default Analysis (Python 3.12.7)")


# --- MAIN UI ---
st.title("Taiwan Credit Card Default Prediction")
st.subheader("Enter client details to generate a risk report")

# We use a form to ensure all inputs are collected at once before sending to API
with st.form("risk_assessment_form"):
    
    # SECTION 1: Demographics
    st.markdown("### 1. Demographic & Credit Profile")
    col1, col2, col3 = st.columns(3)
    with col1:
        x1 = st.number_input(FEATURE_LABELS["X1"], min_value=1000, value=20000, step=1000)
        x2 = st.selectbox("Gender", [1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
    with col2:
        x3 = st.selectbox("Education", [1, 2, 3, 4], format_func=lambda x: {1:"Grad School", 2:"University", 3:"High School", 4:"Others"}[x])
        x4 = st.selectbox("Marital Status", [1, 2, 3], format_func=lambda x: {1:"Married", 2:"Single", 3:"Others"}[x])
    with col3:
        x5 = st.number_input("Age", 18, 100, 30)

    st.divider()

    # SECTION 2: Payment History (X6 - X11)
    st.markdown("### 2. Repayment History")
    st.caption("Status Scale: -1 = Pay Duly, 1 = 1 Month Delay, 2 = 2 Month Delay, etc.")
    h_cols = st.columns(6)
    history_inputs = []
    months = ["Sept", "Aug", "July", "June", "May", "April"]
    for i, month in enumerate(months):
        with h_cols[i]:
            val = st.number_input(f"{month}", -2, 8, 0, key=f"hist_{i}")
            history_inputs.append(val)

    st.divider()

    # SECTION 3: Financials (X12 - X23)
    st.markdown("3. Bill Amounts & Previous Payments")
    f_cols = st.columns(2)
    bill_amounts = []
    prev_payments = []
    
    with f_cols[0]:
        st.caption("Bill Statement (NT$)")
        for i, month in enumerate(months):
            val = st.number_input(f"Bill {month}", value=0.0, key=f"bill_{i}")
            bill_amounts.append(val)
            
    with f_cols[1]:
        st.caption("Previous Payment (NT$)")
        for i, month in enumerate(months):
            val = st.number_input(f"Paid {month}", value=0.0, key=f"pay_{i}")
            prev_payments.append(val)

    # SUBMIT BUTTON
    submit_btn = st.form_submit_button("GENERATE RISK REPORT", type="primary")

# --- LOGIC & RESULTS ---
if submit_btn:
    # Construct Payload matching FastAPI 'CreditRiskFeatures' schema
    payload = {
        "X1": float(x1), "X2": int(x2), "X3": int(x3), "X4": int(x4), "X5": int(x5),
        "X6": history_inputs[0], "X7": history_inputs[1], "X8": history_inputs[2],
        "X9": history_inputs[3], "X10": history_inputs[4], "X11": history_inputs[5],
        "X12": bill_amounts[0], "X13": bill_amounts[1], "X14": bill_amounts[2],
        "X15": bill_amounts[3], "X16": bill_amounts[4], "X17": bill_amounts[5],
        "X18": prev_payments[0], "X19": prev_payments[1], "X20": prev_payments[2],
        "X21": prev_payments[3], "X22": prev_payments[4], "X23": prev_payments[5]
    }

    with st.spinner("Analyzing credit risk..."):
        try:
            response = requests.post(API_ENDPOINT, json=payload)
            response.raise_for_status()
            data = response.json()

            # UI Display for Results
            st.divider()
            res_col1, res_col2 = st.columns([1, 1])
            
            with res_col1:
                prob = data["probability_default"]
                st.metric("Probability of Default", f"{prob*100:.2f}%")
                
                if data["prediction"] == 1:
                    st.error("HIGH RISK: REJECT")
                else:
                    st.success("LOW RISK: APPROVE")
            
            with res_col2:
                # Simple gauge/visual
                st.write("#### Risk Level")
                risk_color = "red" if prob > 0.5 else "orange" if prob > 0.3 else "green"
                st.markdown(f"""
                    <div style="width: 100%; background-color: #ddd; border-radius: 10px;">
                        <div style="width: {prob*100}%; background-color: {risk_color}; 
                        padding: 10px; color: white; border-radius: 10px; text-align: center;">
                        {prob*100:.1f}%
                        </div>
                    </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction Failed: {e}")
            st.info("Ensure the FastAPI server is running at http://127.0.0.1:8000")
        #python -m streamlit run frontend/app.py        
        #streamlit run frontend/app.py