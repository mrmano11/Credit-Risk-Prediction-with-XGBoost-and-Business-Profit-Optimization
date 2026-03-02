import pandas as pd
import numpy as np
import streamlit as st
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Credit Risk + Profit Optimization", layout="wide")

# -----------------------------
# Load artifacts
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("credit_model.pkl")
    return model

@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    df = df.drop("ID", axis=1)
    target = "default.payment.next.month"
    X = df.drop(target, axis=1)
    y = df[target]
    return df, X, y, target

model = load_artifacts()
df, X, y, target = load_data()

st.title("Credit Risk Prediction with Profit Optimization")

st.markdown(
    """
This app predicts **default probability** and helps choose an **approval threshold** based on **profit/loss simulation**.
- If `default_probability >= threshold` → **Reject**
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
# Train-test split for simulation (same dataset)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=int(random_state), stratify=y
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
    st.caption("Enter values for one customer and get default probability + approve/reject decision.")

    # Build input form from feature columns
    with st.form("input_form"):
        user_input = {}
        for c in X.columns:
            # Use reasonable defaults from median/mode
            if pd.api.types.is_numeric_dtype(X[c]):
                default_val = float(X[c].median())
                user_input[c] = st.number_input(c, value=default_val)
            else:
                # For safety (though this dataset is numeric), fallback to most common
                options = sorted(df[c].astype(str).unique().tolist())
                pick = st.selectbox(c, options, index=0)
                user_input[c] = pick

        submitted = st.form_submit_button("Predict")

    if submitted:
        row = pd.DataFrame([user_input])
        # Ensure column order
        row = row[X.columns]
        p = float(model.predict_proba(row)[:, 1][0])
        decision = "REJECT" if p >= threshold else "APPROVE"
        st.metric("Default Probability", f"{p:.3f}")
        st.metric("Decision", decision)
        st.write(
            f"Rule used: if default_probability ≥ {threshold:.2f} → Reject else Approve"
        )

with col2:
    st.subheader("Profit Optimization (Portfolio Simulation)")
    current_profit = profit_at_threshold(float(threshold))
    st.metric("Profit at selected threshold", f"₹{current_profit:,}")

    # Compute profit curve quickly
    thresholds = np.arange(0.05, 0.96, 0.05)
    profits = [profit_at_threshold(float(t)) for t in thresholds]

    best_idx = int(np.argmax(profits))
    best_t = float(thresholds[best_idx])
    best_profit = int(profits[best_idx])

    st.metric("Best Threshold (from simulation)", f"{best_t:.2f}")
    st.metric("Max Profit (from simulation)", f"₹{best_profit:,}")

    fig = plt.figure()
    plt.plot(thresholds, profits)
    plt.xlabel("Threshold")
    plt.ylabel("Profit (₹)")
    st.pyplot(fig)

st.divider()
st.subheader("Dataset Preview")
st.dataframe(df.head(10))
