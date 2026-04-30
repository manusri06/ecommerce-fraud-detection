import streamlit as st
import pandas as pd
import pickle

# -------- PAGE CONFIG --------
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="centered"
)

# -------- LOAD MODEL (CACHED) --------
@st.cache_resource
def load_model():
    return pickle.load(open("fraud_pipeline.pkl", "rb"))

model = load_model()

# -------- CUSTOM CSS --------
st.markdown("""
    <style>
        .stButton>button {
            width: 100%;
            border-radius: 10px;
            height: 3em;
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

# -------- TITLE --------
st.title("💳 E-Commerce Fraud Detection")
st.write("Enter transaction details to check if it's fraudulent.")

# -------- FORM --------
with st.form("fraud_form"):

    col1, col2 = st.columns(2)

    with col1:
        amount = st.number_input("Transaction Amount", min_value=0.0, value=500.0)
        quantity = st.number_input("Quantity", min_value=1, value=1)
        age = st.number_input("Customer Age", min_value=0, value=25)
        account_age = st.number_input("Account Age (Days)", min_value=0, value=100)

    with col2:
        hour = st.slider("Transaction Hour", 0, 23, 12)
        payment = st.selectbox("Payment Method", ["UPI", "Card", "NetBanking"])
        category = st.selectbox("Product Category", ["Electronics", "Clothing", "Home"])
        device = st.selectbox("Device Used", ["Mobile", "Laptop", "Tablet"])

    submit = st.form_submit_button("🔍 Predict Fraud")

# -------- PREDICTION --------
if submit:

    # -------- FEATURE ENGINEERING --------
    is_night = 1 if hour >= 22 or hour <= 5 else 0
    amount_per_item = amount / (quantity + 1)
    is_new = 1 if account_age < 30 else 0

    input_data = pd.DataFrame([{
        "Transaction Amount": amount,
        "Quantity": quantity,
        "Customer Age": age,
        "Account Age Days": account_age,
        "Transaction Hour": hour,
        "Is_Night": is_night,
        "Amount_per_Item": amount_per_item,
        "Is_New_Account": is_new,
        "Payment Method": payment,
        "Product Category": category,
        "Device Used": device
    }])

    # -------- PREDICTION --------
    with st.spinner("Analyzing transaction..."):
        prob = model.predict_proba(input_data)[0][1]

    # -------- RESULT --------
    st.subheader("📊 Prediction Result")
    st.metric("Fraud Probability", f"{prob*100:.1f}%")

    # -------- RISK LEVEL --------
    if prob > 0.8:
        st.markdown("### 🔴 Risk Level: HIGH RISK")
    elif prob > 0.5:
        st.markdown("### 🟠 Risk Level: MEDIUM RISK")
    else:
        st.markdown("### 🟢 Risk Level: LOW RISK")

    # -------- FINAL DECISION --------
    threshold = 0.7
    if prob > threshold:
        st.error("⚠️ Fraudulent Transaction Detected")
        st.warning("This transaction has a high likelihood of being fraudulent. Immediate verification is recommended.")
    else:
        st.success("✅ Legitimate Transaction")

    st.progress(min(int(prob * 100), 100))

    # -------- KEY FACTORS --------
    st.subheader("🧠 Why this transaction is flagged")

    factors = []
    if amount > 10000:
        factors.append("High transaction amount")
    if is_new:
        factors.append("New account (less than 30 days)")
    if is_night:
        factors.append("Night-time transaction")
    if quantity > 5:
        factors.append("Bulk purchase behavior")

    if factors:
        for f in factors:
            st.write(f"- {f}")
    else:
        st.write("No strong risk factors detected.")

    # -------- DOWNLOAD --------
    st.download_button(
        label="📥 Download Prediction Data",
        data=input_data.to_csv(index=False),
        file_name="fraud_prediction.csv",
        mime="text/csv"
    )

    # -------- INFO --------
    st.info("Prediction is based on transaction patterns, user behavior, and historical fraud trends.")

# -------- FOOTER --------
st.markdown("---")
st.caption("Built by Manusri | ML-Based E-Commerce Fraud Detection System 🚀")
