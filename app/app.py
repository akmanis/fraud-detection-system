import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from sklearn.metrics import confusion_matrix, roc_curve, auc

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("fraud_model.pkl")

st.set_page_config(
    page_title="Credit Card Fraud Detection System", 
    layout="wide"
)

# -----------------------------
# Page Title
# -----------------------------
st.title("Credit Card Fraud Detection System")
st.subheader("Machine Learning Powered Fraud Monitoring")

# -----------------------------
# Fraud Threshold Control
# -----------------------------
threshold = st.slider("Fraud Detection Threshold",0.01,0.50,0.10)

# -----------------------------
# Single Transaction Check
# -----------------------------
st.header("Check Single Transaction")

amount = st.number_input("Transaction Amount", min_value=0.0)

if st.button("Check Transaction"):

    features = np.zeros((1,30))
    features[0][-1] = amount

    probability = model.predict_proba(features)[0][1]

    risk_score = int(probability*100)

    if risk_score < 30:
        risk_level = "Low"
    elif risk_score < 70:
        risk_level = "Medium"
    else:
        risk_level = "High"

    st.write("Fraud Probability:", round(probability*100,2),"%")
    st.write("Risk Score:",risk_score)
    st.write("Risk Level:",risk_level)

    if probability > threshold:
        st.error("Fraudulent Transaction Detected")
    else:
        st.success("Transaction is Legitimate")

# -----------------------------
# Transaction Monitoring
# -----------------------------
st.header("Live Transaction Monitoring")

if st.button("Generate Transactions"):

    transactions = np.random.uniform(1,10000,100)

    results=[]

    for amt in transactions:

        features=np.random.normal(0,1,(1,30))
        features[0][-1]=amt

        prob=model.predict_proba(features)[0][1]

        risk_score=int(prob*100)

        if risk_score < 30:
            risk_level="Low"
        elif risk_score < 70:
            risk_level="Medium"
        else:
            risk_level="High"

        lat=np.random.uniform(-60,60)
        lon=np.random.uniform(-180,180)

        results.append({
            "Transaction Amount":round(amt,2),
            "Fraud Probability":round(prob,4),
            "Risk Score":risk_score,
            "Risk Level":risk_level,
            "Fraud":prob>threshold,
            "lat":lat,
            "lon":lon
        })

    df=pd.DataFrame(results)

    st.subheader("Transaction Table")
    st.dataframe(df)

    fraud_rate=df["Fraud"].mean()

    col1,col2=st.columns(2)
    col1.metric("Total Transactions",len(df))
    col2.metric("Fraud Rate",str(round(fraud_rate*100,2))+"%")

    st.subheader("Fraud vs Normal Transactions")
    st.bar_chart(df["Fraud"].value_counts())

    st.subheader("Fraud Probability Distribution")
    st.bar_chart(df["Fraud Probability"])

    st.subheader("Risk Level Distribution")
    st.bar_chart(df["Risk Level"].value_counts())

    # -----------------------------
    # High Risk Transactions
    # -----------------------------
    st.subheader("High Risk Transactions")

    high_risk=df[df["Risk Level"]=="High"]

    if len(high_risk)>0:
        st.dataframe(high_risk)
    else:
        st.write("No high risk transactions detected.")

    # -----------------------------
    # Fraud Alerts
    # -----------------------------
    st.subheader("Real-Time Fraud Alerts")

    fraud_count=df["Fraud"].sum()

    if fraud_count>5:
        st.error("⚠️ Multiple high risk transactions detected!")
    elif fraud_count>0:
        st.warning("⚠️ Some suspicious transactions detected")
    else:
        st.success("System Status: Normal")

    # -----------------------------
    # Fraud Heatmap
    # -----------------------------
    st.subheader("Fraud Geographic Heatmap")

    fraud_locations=df[df["Fraud"]==True][["lat","lon"]]

    if len(fraud_locations)>0:
        st.map(fraud_locations)
    else:
        st.write("No fraud locations detected.")

    # -----------------------------
    # Confusion Matrix
    # -----------------------------
    st.subheader("Model Confusion Matrix")

    y_true=np.random.choice([0,1],size=100,p=[0.95,0.05])
    y_pred=df["Fraud"].astype(int)

    cm=confusion_matrix(y_true,y_pred)

    fig,ax=plt.subplots()
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",ax=ax)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

    # -----------------------------
    # ROC Curve
    # -----------------------------
    st.subheader("ROC Curve")

    y_scores=df["Fraud Probability"]

    fpr,tpr,_=roc_curve(y_true,y_scores)
    roc_auc=auc(fpr,tpr)

    fig2,ax2=plt.subplots()

    ax2.plot(fpr,tpr,label=f"ROC Curve (AUC={roc_auc:.2f})")
    ax2.plot([0,1],[0,1],'--')

    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")

    ax2.legend()

    st.pyplot(fig2)

    # -----------------------------
    # Transaction Network Graph
    # -----------------------------
    st.subheader("Transaction Network Graph (Fraud Ring Detection)")

    G=nx.Graph()

    accounts=[f"A{i}" for i in range(10)]
    merchants=[f"M{i}" for i in range(5)]

    for i in range(30):
        acc=np.random.choice(accounts)
        mer=np.random.choice(merchants)
        G.add_edge(acc,mer)

    fig3,ax3=plt.subplots()

    pos=nx.spring_layout(G)

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="lightblue",
        edge_color="gray",
        node_size=1500,
        font_size=8,
        ax=ax3
    )

    st.pyplot(fig3)
