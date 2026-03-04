import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load trained model
model = pickle.load(open("models/fraud_model.pkl","rb"))
# Page title
st.title("Credit Card Fraud Detection System")
st.subheader("Machine Learning Powered Fraud Monitoring")

# ---------------------------
# Single Transaction Check
# ---------------------------

st.header("Check Single Transaction")

amount = st.number_input("Transaction Amount", min_value=0.0)

if st.button("Check Transaction"):

    features = np.zeros((1, 30))
    features[0][-1] = amount

    probability = model.predict_proba(features)[0][1]

    st.write("Fraud Probability:", round(probability * 100, 2), "%")

    if probability > 0.5:
        st.error("Fraudulent Transaction Detected")
    else:
        st.success("Transaction is Legitimate")

# ---------------------------
# Transaction Monitoring
# ---------------------------

st.header("Live Transaction Monitoring")

if st.button("Generate Transactions"):

    transactions = np.random.uniform(1, 10000, 100)

    results = []

    for amt in transactions:

        features = np.random.normal(0,1,(1,30))
        features[0][-1] = amt

        prob = model.predict_proba(features)[0][1]

        results.append({
            "Transaction Amount": round(amt, 2),
            "Fraud Probability": round(prob, 4),
            "Fraud": prob > 0.5
        })

    df = pd.DataFrame(results)

    st.subheader("Transaction Table")
    st.dataframe(df)

    fraud_rate = df["Fraud"].mean()

    col1, col2 = st.columns(2)

    col1.metric("Total Transactions", len(df))
    col2.metric("Fraud Rate", str(round(fraud_rate * 100, 2)) + "%")

    st.subheader("Fraud vs Normal Transactions")
    st.bar_chart(df["Fraud"].value_counts())

    st.subheader("Fraud Probability Distribution")
    st.bar_chart(df["Fraud Probability"])

    st.subheader("High Risk Transactions")

    high_risk = df[df["Fraud Probability"] > 0.5]

    if len(high_risk) > 0:
        st.dataframe(high_risk)
    else:
        st.write("No high risk transactions detected.")
