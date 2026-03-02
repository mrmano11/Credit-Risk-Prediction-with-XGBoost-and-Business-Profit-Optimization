# Credit Risk Prediction with XGBoost and Business Profit Optimization

This project predicts credit default risk using multiple ML models and optimizes the loan approval threshold based on simulated business profit/loss.

## Dataset
UCI Credit Card Default dataset (30,000 rows).

## Models
- Logistic Regression
- Random Forest
- XGBoost (best ROC-AUC in my run)

## Key Result (Business Optimization)
Using profit simulation (Profit = ₹10,000 for good approvals, Loss = ₹50,000 for defaults), the best threshold found was around **0.20**, improving simulated profit substantially compared to a default 0.50 threshold.

## How to Run
### Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
