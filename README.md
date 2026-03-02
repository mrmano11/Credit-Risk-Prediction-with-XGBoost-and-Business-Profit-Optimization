# Credit Risk Prediction with XGBoost and Business Profit Optimization

## 📌 Project Overview

This project builds a machine learning system to predict customer credit default risk and optimize loan approval decisions based on financial profit and loss simulation.

Instead of focusing only on accuracy, this project integrates business-driven threshold optimization to maximize profit and minimize financial risk.

---

## 🎯 Problem Statement

Financial institutions must decide whether to approve or reject loan applications.

Approving high-risk customers may lead to large financial losses, while rejecting safe customers results in lost profit.

The objective of this project is to:

- Predict probability of default
- Compare multiple machine learning models
- Optimize approval threshold using profit simulation
- Provide an interactive dashboard for decision-making

---

## 📊 Dataset

Default of Credit Card Clients Dataset (UCI / Kaggle)

Dataset link:  
https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset

### Description
This dataset contains information on credit card customers including demographic details, credit limits, payment history, bill amounts, and repayment behavior.

- Total records: 30,000 customers  
- Features: 23 input features  
- Target variable: `default.payment.next.month`  
  - 1 → Customer will default  
  - 0 → Customer will not default  

### Objective
Predict whether a customer will default on the next month’s payment using historical financial behavior. 

---

## 🤖 Models Used

- Logistic Regression
- Random Forest
- XGBoost (Best Performing Model)

### Model Performance (ROC-AUC)

- Logistic Regression: ~0.70
- Random Forest: ~0.75
- XGBoost: ~0.77

XGBoost achieved the highest discrimination ability.

---

## 💰 Business Profit Optimization

Business Assumptions:

- Profit per good approved customer = ₹10,000
- Loss per approved defaulter = ₹50,000

Instead of using default threshold (0.5), multiple thresholds were evaluated.

### Key Result:

- Default threshold (0.5) → ~₹23 lakh profit
- Optimized threshold (0.2) → ~₹1.22 crore profit

This demonstrates that threshold tuning significantly improves financial performance.

---

## 📈 Dashboard Features (Streamlit)

- Adjustable approval threshold
- Dynamic profit simulation
- Approval rate & default rate metrics
- Confusion matrix visualization
- Top feature importance visualization
- Download predictions option
- Manual customer scoring

---

## 🧠 Key Concepts Demonstrated

- ROC-AUC evaluation
- Model comparison
- Cost-sensitive decision making
- Threshold optimization
- Business risk management
- Expected value thinking
- Interactive ML deployment

---

## 👨‍💻 Auor

Mohanarengan S
Final-year Student
Paavai College of Engineering

---

## 🚀 How to Run

### 1. Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
