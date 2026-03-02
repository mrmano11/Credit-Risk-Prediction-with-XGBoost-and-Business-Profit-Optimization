import pandas as pd
import numpy as np
import streamlit as st
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

st.set_page_config(page_title="Credit Risk + Profit Optimization", layout="wide")

# -----------------------------
# Load artifacts
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("credit_model.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    df = df.drop("ID", axis=1)
    target = "default.payment.next.month"
    X = df.drop(target, axis=1)
    y = df[target]
    return df, X, y, target

model = load_model()
df, X, y, target = load_data()

st.title("Credit Risk Prediction with Profit Optimization")

st.markdown(
    """
This app predicts **default probability** and supports **loan approval decisions** using **profit/loss simulation**.

**Decision rule**
- If `default_probability ≥ threshold` → **Reject**
- If `default_probability < threshold` → **Approve**
"""
)

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Business Assumptions")
profit_good = st.sidebar.number_input("Profit per good approved customer (₹)", min_value=0, value=10000, step=1000)
loss_default = st.sidebar.number_input("Loss per approved defaulter (₹)", min_value=0, value=50000, step=5000)
threshold = st.sidebar.slider("Approval Threshold", min_value=0.05, max_value=0.95, value=0.20, step=0.05)

st.sidebar.header("Data Split")
test_size = st.sidebar.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
random_state = st.sidebar.number_input("Random state", min_value=0, value=42, step=1)

# -----------------------------
# Train-test split for simulation
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=float(test_size), random_state=int(random_state), stratify=y
)

probs = model.predict_proba(X_test)[:, 1]

# -----------------------------
# Helper: profit at a threshold
# -----------------------------
def profit_at_threshold(t: float) -> int:
    approve = probs < t
    total_profit = 0
    for i, ok in enumerate(approve):
        if ok:  # approved
            if int(y_test.iloc[i]) == 0:
                total_profit += int(profit_good)
            else:
                total_profit -= int(loss_default)
    return total_profit

# -----------------------------
# Layout
# -----------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Single Customer Scoring (Manual Input)")
    st.caption("Enter customer values to get default probability + approve/reject decision.")

    with st.form("input_form"):
        user_input = {}
        for c in X.columns:
            # Dataset is numeric, keep number inputs
            default_val = float(X[c].median())
            user_input[c] = st.number_input(c, value=default_val)

        submitted = st.form_submit_button("Predict")

    if submitted:
        row = pd.DataFrame([user_input])[X.columns]
        p = float(model.predict_proba(row)[:, 1][0])
        decision = "REJECT" if p >= float(threshold) else "APPROVE"
        st.metric("Default Probability", f"{p:.3f}")
        st.metric("Decision", decision)
        st.write(f"Rule: if default_probability ≥ {float(threshold):.2f} → Reject else Approve")

with col2:
    st.subheader("Portfolio Profit Optimization")

    # KPIs at selected threshold
    approve = probs < float(threshold)
    approved_count = int(approve.sum())
    total_count = len(approve)

    approved_defaults = int((y_test[approve] == 1).sum())
    approved_goods = int((y_test[approve] == 0).sum())

    approval_rate = approved_count / total_count if total_count > 0 else 0.0
    default_rate_approved = (approved_defaults / approved_count) if approved_count > 0 else 0.0

    current_profit = profit_at_threshold(float(threshold))

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Profit at Threshold", f"₹{current_profit:,}")
    k2.metric("Approval Rate", f"{approval_rate*100:.1f}%")
    k3.metric("Approved Defaulters", f"{approved_defaults} / {approved_count}")
    k4.metric("Default Rate (Approved)", f"{default_rate_approved*100:.1f}%")

    # Profit curve
    thresholds = np.arange(0.05, 0.96, 0.05)
    profits = [profit_at_threshold(float(t)) for t in thresholds]

    best_idx = int(np.argmax(profits))
    best_t = float(thresholds[best_idx])
    best_profit = int(profits[best_idx])

    st.info(f"Recommended threshold (max profit for these assumptions): **{best_t:.2f}** | Max profit: **₹{best_profit:,}**")

    fig = plt.figure()
    plt.plot(thresholds, profits)
    plt.xlabel("Threshold")
    plt.ylabel("Profit (₹)")
    st.pyplot(fig)

    # Confusion matrix at selected threshold
    # Predict default if prob >= threshold (1=default)
    y_pred_thr = (probs >= float(threshold)).astype(int)
    cm = confusion_matrix(y_test, y_pred_thr)

    st.subheader("Confusion Matrix at Selected Threshold")
    st.caption("Rows = Actual, Columns = Predicted (0=No Default, 1=Default)")
    st.write(cm)

st.divider()

# -----------------------------
# Feature importance
# -----------------------------
st.subheader("Top Feature Importances (XGBoost)")
importances = getattr(model, "feature_importances_", None)
if importances is not None:
    fi = (
        pd.DataFrame({"feature": X.columns, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(10)
    )

    fig2 = plt.figure()
    plt.barh(fi["feature"][::-1], fi["importance"][::-1])
    plt.xlabel("Importance")
    st.pyplot(fig2)
else:
    st.warning("Feature importances not available for this model.")

# -----------------------------
# Download predictions
# -----------------------------
st.subheader("Download Predictions")
pred_df = X_test.copy()
pred_df["default_probability"] = probs
pred_df["approved"] = (probs < float(threshold))
pred_df[target] = y_test.values

csv = pred_df.to_csv(index=False).encode("utf-8")
st.download_button("Download Predictions CSV", csv, "predictions.csv", "text/csv")

st.divider()
st.subheader("Dataset Preview")
st.dataframe(df.head(10))
