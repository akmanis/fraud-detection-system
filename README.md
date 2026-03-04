# AI-Powered Credit Card Fraud Detection & Monitoring Dashboard

## Overview

This project is an **AI-powered credit card fraud detection system** built using **Machine Learning and Streamlit**.
It simulates a **real-time fraud monitoring dashboard** similar to the systems used by banks and fintech companies.

The system predicts the probability of fraudulent transactions, assigns risk scores, and visualizes fraud activity using analytics dashboards, geographic heatmaps, and transaction network graphs.

This project demonstrates how **machine learning can be used for financial risk management and fraud detection**.

---

# Live Demo

You can run the deployed application here:

**Streamlit App:**
`https://fraud-detection-system-ak.streamlit.app`

---

# Dashboard Preview

## Main Fraud Detection Dashboard

<img width="1465" height="667" alt="image" src="https://github.com/user-attachments/assets/d2098c5d-ec03-4a9d-9014-1496d0959c6b" />


The main interface allows users to check individual transactions and monitor fraud activity.

Features:

* Fraud probability prediction
* Risk score calculation
* Risk classification
* Fraud detection alerts

---

# Transaction Monitoring System

<img width="1470" height="751" alt="image" src="https://github.com/user-attachments/assets/a49abee9-12c5-4f78-b0d2-c013098a31b7" />


The system simulates real-time transaction monitoring.

Each transaction includes:

* Transaction amount
* Fraud probability
* Risk score
* Risk level
* Fraud classification

Key metrics shown:

* Total transactions
* Fraud rate
* Risk distribution

---

# Fraud Geographic Heatmap

<img width="1470" height="600" alt="image" src="https://github.com/user-attachments/assets/7a107405-8bf7-48b9-bd00-dc3deaf406b5" />


The dashboard visualizes fraud locations on a geographic map.

This helps identify:

* Suspicious regions
* Fraud clusters
* Transaction patterns

Banks commonly use geographic monitoring to detect unusual spending patterns.

---

# Model Evaluation Metrics

<img width="1470" height="622" alt="image" src="https://github.com/user-attachments/assets/ca154a8a-07fe-459d-bde8-e623233d3853" />


The system includes model performance evaluation tools:

### Confusion Matrix

Shows classification accuracy between fraudulent and legitimate transactions.

### ROC Curve

Measures the model's ability to distinguish between fraud and legitimate transactions.

These metrics are commonly used in machine learning model validation.

---

# Fraud Ring Detection (Network Graph)

<img width="1470" height="654" alt="image" src="https://github.com/user-attachments/assets/ab8fdb2c-4b66-40d9-8cc2-d097616587bf" />


Fraud rings often involve multiple accounts interacting with the same merchants.

The system simulates transaction networks using graph analysis to visualize:

* Account relationships
* Merchant interactions
* Potential fraud clusters

This technique is widely used in **financial crime detection systems**.

---

# System Architecture

Machine Learning Model

Input Transaction → Feature Processing → Fraud Probability Prediction → Risk Scoring → Monitoring Dashboard

---

# Features

### Fraud Prediction

Predicts the probability of fraud using a trained machine learning model.

### Risk Scoring System

Each transaction is assigned a risk score from **0 to 100**.

### Risk Classification

Transactions are categorized as:

* Low Risk
* Medium Risk
* High Risk

### Real-Time Fraud Alerts

Alerts trigger when suspicious activity is detected.

### Transaction Monitoring

Simulated real-time transaction feed.

### Fraud Heatmap

Geographic visualization of suspicious activity.

### Network Graph Analysis

Detects possible fraud rings using transaction relationships.

### Model Evaluation Tools

Includes ROC Curve and Confusion Matrix.

---

# Tech Stack

Programming Language
Python

Libraries

* Streamlit
* Scikit-learn
* Pandas
* NumPy
* Matplotlib
* Seaborn
* NetworkX

Machine Learning
Fraud classification model using supervised learning.

Deployment
Streamlit Cloud

---

# Project Structure

```
fraud-detection-system
│
├── app
│    └── app.py
├── fraud_model.pkl
├── requirements.txt
└── README.md

```

---

# How to Run Locally

Clone the repository

```
git clone https://github.com/akmanis/fraud-detection-system.git
```

Install dependencies

```
pip install -r requirements.txt
```

Run the application

```
streamlit run app.py
```

---

# Future Improvements

Possible enhancements for production systems:

* Real credit card dataset integration
* Graph neural networks for fraud ring detection
* Real-time streaming data
* User authentication
* Fraud investigation tools

---

# Author

Manish,
Economic Sciences,
Indian Institute of Science Education and Research (IISER) Bhopal

---
