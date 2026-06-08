# 🛡️ BankShield — Intelligent Fraud Detection System

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bankshield-kpfzrzjrnr55b8otwvcoay.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![ML](https://img.shields.io/badge/ML-Random%20Forest-green?logo=scikit-learn)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-orange)
![Status](https://img.shields.io/badge/Status-Live-brightgreen)

> A production-ready machine learning application that detects fraudulent mobile financial transactions in real-time — with full AI explainability powered by SHAP.

---

## 🔴 Live Demo

🚀 **[Launch BankShield App](https://bankshield-kpfzrzjrnr55b8otwvcoay.streamlit.app/)**

---

## 📌 Problem Statement

Financial fraud costs the global economy **billions of dollars annually**. Traditional rule-based systems fail to detect sophisticated fraud patterns in real-time. BankShield solves this by using machine learning to analyze transaction behavior and flag suspicious activity — with explainable AI so every decision can be trusted and audited.

---

## 🎯 Key Results

| Metric | Score |
|--------|-------|
| ✅ Precision | **0.94** |
| ✅ Recall | **0.89** |
| ✅ F1 Score | **0.92** |
| ✅ ROC-AUC | **0.97** |

---

## 🧠 How It Works

```
Raw Transaction Data
        ↓
Feature Engineering (17 features)
        ↓
Random Forest Classifier
        ↓
Fraud Probability Score (0–100%)
        ↓
SHAP Explainability → Why this decision?
        ↓
FRAUD ⚠️  or  LEGITIMATE ✓
```

---

## 🔧 Tech Stack

| Layer | Technology |
|-------|-----------|
| **ML Model** | Random Forest, scikit-learn |
| **Class Balancing** | `class_weight='balanced'` |
| **Explainability** | SHAP |
| **Feature Engineering** | pandas, numpy |
| **Web App** | Streamlit |
| **Deployment** | Streamlit Cloud |
| **Version Control** | GitHub |

---

## 📋 Features Used

The model uses **17 engineered features** derived from raw transaction data:

| Feature | Description |
|---------|-------------|
| `amount` | Transaction amount |
| `type_encoded` | Transaction type (encoded) |
| `oldbalanceOrg` | Origin account opening balance |
| `newbalanceOrig` | Origin account closing balance |
| `oldbalanceDest` | Destination account opening balance |
| `newbalanceDest` | Destination account closing balance |
| `balanceOrigDiff` | Change in origin balance |
| `balanceDestDiff` | Change in destination balance |
| `errorBalanceOrig` | Balance error at origin |
| `errorBalanceDest` | Balance error at destination |
| `amountToBalance` | Amount to origin balance ratio |
| `amountToDestBalance` | Amount to destination balance ratio |
| `isHighAmount` | High amount flag |
| `zeroBalanceOrig` | Origin balance was zero flag |
| `zeroBalanceDest` | Destination balance was zero flag |
| `accountDrained` | Account drained to zero flag |
| `step` | Transaction time step (hour) |

---

## 🖥️ App Features

- 🔍 **Real-time fraud prediction** with confidence score
- 📊 **SHAP explainability chart** — see exactly which features drove the decision
- 🎨 **Dark-themed professional UI**
- 📋 **Transaction summary table**
- ⚡ **Instant results** — no retraining needed

---

## 📁 Project Structure

```
BankShield/
├── app.py                  ← Streamlit web application
├── generate_data.py        ← Synthetic data generator
├── train.py                ← Model training pipeline
├── models/
│   ├── model.pkl           ← Trained Random Forest model
│   ├── feature_names.pkl   ← Feature names
│   └── label_encoder.pkl   ← Label encoder for transaction type
├── requirements.txt        ← Python dependencies
└── README.md               ← You are here
```

---

## ⚙️ Run Locally

```bash
# Clone the repo
git clone https://github.com/venkatraman0400-blip/BankShield.git
cd BankShield

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

To retrain the model from scratch:

```bash
python generate_data.py   # generate synthetic training data
python train.py           # train and save the model
streamlit run app.py      # launch the app
```

---

## 👨‍💻 About

Built by **Venkatraman R** — Data Science & AI Engineer  
📍 Chennai, India  
🏛 Boston Institute of Analytics — Professional Certification in Data Science with AI

[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-black?logo=github)](https://venkatraman0400-blip.github.io/venkatraman-portfolio)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/venkatraman)

---

> *"In financial services, a model that cannot explain its decisions cannot be trusted in production."*
