# 🏦 AI Credit Risk Prediction Platform

> An end-to-end machine learning platform to predict financial delinquency risk — built with Python, Scikit-Learn, and Streamlit.

🔴 **[Live Demo](https://credit-risk-platform-mkzbzjv7stanrcpx2dg5mh.streamlit.app)** &nbsp;|&nbsp; ⭐ Star this repo if you find it useful!

---

## 📌 Overview

Banks and financial institutions use credit risk models to decide whether to approve loans. This platform replicates that process using real-world data — from raw CSV to a fully deployed interactive dashboard.

| Metric | Value |
|--------|-------|
| 📊 Dataset | 149,399 records (Kaggle — Give Me Some Credit) |
| 🎯 Best AUC | 0.859 (Gradient Boosting) |
| 🏆 Best Accuracy | 93.8% |
| 🚀 Deployment | Streamlit Cloud |

---

## ✨ Features

- 📊 **Overview Dashboard** — Default rates, income distribution, correlation heatmaps
- 🔍 **Data Explorer** — Interactive feature analysis with filters
- 🤖 **Model Performance** — ROC curves, confusion matrix, feature importance
- 🎯 **Risk Predictor** — Real-time prediction with SHAP-style explanations
- 📦 **Batch Prediction** — Upload CSV → predict 1000+ applicants → download Excel/CSV results

---

## 🧠 How It Works

```
Raw CSV Data (149,399 records)
        ↓
Preprocessing & Feature Engineering
        ↓
Train 3 ML Models
        ↓
Interactive Streamlit Dashboard
        ↓
Real-time Risk Prediction + SHAP Explanation
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas | Data manipulation |
| Scikit-Learn | ML models |
| Matplotlib / Seaborn | Visualizations |
| Streamlit | Interactive dashboard |
| OpenPyXL | Excel report generation |

---

## 🤖 ML Models

| Model | Accuracy | AUC |
|-------|----------|-----|
| Random Forest | 93.7% | 0.834 |
| **Gradient Boosting** | **93.8%** | **0.859** |
| Logistic Regression | 93.5% | 0.794 |

---

## 📁 Project Structure

```
credit_risk_platform/
├── app.py                  ← Streamlit dashboard (5 pages)
├── requirements.txt
├── src/
│   ├── preprocess.py       ← Data cleaning & encoding
│   ├── features.py         ← Feature engineering
│   └── model.py            ← ML training + SHAP + batch prediction
└── data/
    └── cs-training.csv     ← Kaggle dataset
```

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/Varshithaduppelli/credit-risk-platform.git
cd credit-risk-platform

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

---

## 📊 Dataset

**Give Me Some Credit** — Kaggle Competition Dataset

- 150,000 borrower records
- Target: `SeriousDlqin2yrs` (defaulted within 2 years)
- Features: credit utilization, income, missed payments, debt ratio, etc.

---

## 👩‍💻 Author

**Varshitha Duppelli**

[![GitHub](https://img.shields.io/badge/GitHub-Varshithaduppelli-black?logo=github)](https://github.com/Varshithaduppelli)
