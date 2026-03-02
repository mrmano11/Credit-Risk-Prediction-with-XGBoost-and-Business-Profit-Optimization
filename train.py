import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import xgboost as xgb
import joblib

# -------------------------------
# 1. Load data
# -------------------------------
df = pd.read_csv("data.csv")

print("Dataset shape:", df.shape)

# Drop ID column (not useful for prediction)
df = df.drop("ID", axis=1)

# Target column
target = "default.payment.next.month"

X = df.drop(target, axis=1)
y = df[target]

print("Features shape:", X.shape)

# -------------------------------
# 2. Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

# -------------------------------
# 3. Feature scaling
# -------------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# 4. Logistic Regression
# -------------------------------
print("\nTraining Logistic Regression...")
lr = LogisticRegression(max_iter=2000)
lr.fit(X_train_scaled, y_train)

lr_pred = lr.predict(X_test_scaled)
lr_prob = lr.predict_proba(X_test_scaled)[:,1]

print("Logistic Accuracy:", accuracy_score(y_test, lr_pred))
print("Logistic ROC-AUC:", roc_auc_score(y_test, lr_prob))
print("Confusion Matrix:\n", confusion_matrix(y_test, lr_pred))

# -------------------------------
# 5. Random Forest
# -------------------------------
print("\nTraining Random Forest...")
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:,1]

print("RF Accuracy:", accuracy_score(y_test, rf_pred))
print("RF ROC-AUC:", roc_auc_score(y_test, rf_prob))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))

# -------------------------------
# 6. XGBoost
# -------------------------------
print("\nTraining XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)
xgb_prob = xgb_model.predict_proba(X_test)[:,1]

print("XGB Accuracy:", accuracy_score(y_test, xgb_pred))
print("XGB ROC-AUC:", roc_auc_score(y_test, xgb_prob))
print("Confusion Matrix:\n", confusion_matrix(y_test, xgb_pred))

# -------------------------------
# 7. Save best model (XGBoost)
# -------------------------------
joblib.dump(xgb_model, "credit_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel saved as credit_model.pkl")
