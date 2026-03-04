import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, roc_curve, auc

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("fraud_model.pkl")

# -----------------------------
# Page Title
# -----------------------------
st.title("Credit Card Fraud Detection System")
st.subheader("Machine Learning Powered Fraud Monitoring")

# -----------------------------
# Fraud Threshold Control
# -----------------------------
threshold = st.slider(
    "Fraud Detection Threshold",
    min_value=0.01,
    max_value=0.50,
    value=0.10
)

# -----------------------------
# Single Transaction Check
# -----------------------------
st.header("Check Single Transaction")

amount = st.number_input("Transaction Amount", min_value=0.0)

if st.button("Check Transaction"):

    features = np.zeros((1, 30))
    features[0][-1] = amount

    probability = model.predict_proba(features)[0][1]

    risk_score = int(probability * 100)

    if risk_score < 30:
        risk_level = "Low"
    elif risk_score < 70:
        risk_level = "Medium"
    else:
        risk_level = "High"

    st.write("Fraud Probability:", round(probability * 100, 2), "%")
    st.write("Risk Score:", risk_score)
    st.write("Risk Level:", risk_level)

    if probability > threshold:
        st.error("Fraudulent Transaction Detected")
    else:
        st.success("Transaction is Legitimate")

# -----------------------------
# Transaction Monitoring
# -----------------------------
st.header("Live Transaction Monitoring")

if st.button("Generate Transactions"):

    transactions = np.random.uniform(1, 10000, 100)

    results = []

    for amt in transactions:

        features = np.random.normal(0, 1, (1, 30))
        features[0][-1] = amt

        prob = model.predict_proba(features)[0][1]

        risk_score = int(prob * 100)

        if risk_score < 30:
            risk_level = "Low"
        elif risk_score < 70:
            risk_level = "Medium"
        else:
            risk_level = "High"

        results.append({
            "Transaction Amount": round(amt, 2),
            "Fraud Probability": round(prob, 4),
            "Risk Score": risk_score,
            "Risk Level": risk_level,
            "Fraud": prob > threshold
        })

    df = pd.DataFrame(results)

    # -----------------------------
    # Table
    # -----------------------------
    st.subheader("Transaction Table")
    st.dataframe(df)

    fraud_rate = df["Fraud"].mean()

    col1, col2 = st.columns(2)

    col1.metric("Total Transactions", len(df))
    col2.metric("Fraud Rate", str(round(fraud_rate * 100, 2)) + "%")

    # -----------------------------
    # Fraud vs Normal
    # -----------------------------
    st.subheader("Fraud vs Normal Transactions")
    st.bar_chart(df["Fraud"].value_counts())

    # -----------------------------
    # Probability Distribution
    # -----------------------------
    st.subheader("Fraud Probability Distribution")
    st.bar_chart(df["Fraud Probability"])

    # -----------------------------
    # Risk Level Distribution
    # -----------------------------
    st.subheader("Risk Level Distribution")
    st.bar_chart(df["Risk Level"].value_counts())

    # -----------------------------
    # High Risk Transactions
    # -----------------------------
    st.subheader("High Risk Transactions")

    high_risk = df[df["Risk Level"] == "High"]

    if len(high_risk) > 0:
        st.dataframe(high_risk)
    else:
        st.write("No high risk transactions detected.")

    # -----------------------------
    # Fraud Alert System
    # -----------------------------
    st.subheader("Real-Time Fraud Alerts")

    fraud_count = df["Fraud"].sum()
    high_risk_count = len(high_risk)

    if high_risk_count > 5:
        st.error("⚠️ ALERT: Multiple high risk transactions detected!")
    elif fraud_count > 0:
        st.warning("⚠️ Some suspicious transactions detected")
    else:
        st.success("System Status: Normal")

    # -----------------------------
    # Confusion Matrix
    # -----------------------------
    st.subheader("Model Confusion Matrix")

    y_true = np.random.choice([0,1], size=100, p=[0.95,0.05])
    y_pred = df["Fraud"].astype(int)

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

    # -----------------------------
    # ROC Curve
    # -----------------------------
    st.subheader("ROC Curve")

    y_scores = df["Fraud Probability"]

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig2, ax2 = plt.subplots()

    ax2.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    ax2.plot([0,1],[0,1],'--')

    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")

    ax2.legend()

    st.pyplot(fig2)
