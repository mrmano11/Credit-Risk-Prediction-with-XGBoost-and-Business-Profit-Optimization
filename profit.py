import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("data.csv")
df = df.drop("ID", axis=1)

target = "default.payment.next.month"
X = df.drop(target, axis=1)
y = df[target]

# Split same as before
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Load trained model
model = joblib.load("credit_model.pkl")

# Predict probabilities
probs = model.predict_proba(X_test)[:,1]

# Business assumptions
profit_good = 10000   # profit if good customer pays
loss_default = 50000  # loss if customer defaults

print("\nThreshold vs Profit Analysis\n")

best_profit = -999999
best_threshold = 0

for t in np.arange(0.1, 0.91, 0.1):
    approve = probs < t

    total_profit = 0

    for i in range(len(approve)):
        if approve[i]:  # loan approved
            if y_test.iloc[i] == 0:
                total_profit += profit_good
            else:
                total_profit -= loss_default

    print(f"Threshold {t:.1f} -> Profit: ₹{total_profit}")

    if total_profit > best_profit:
        best_profit = total_profit
        best_threshold = t

print("\nBest Threshold:", best_threshold)
print("Maximum Profit: ₹", best_profit)
