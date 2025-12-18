import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Customer Pattern Explorer",
    layout="centered"
)

# ---------------- Styling ----------------
st.markdown("""
<style>
body {
    background-color: #020617;
}
.card {
    background: rgba(30,41,59,0.75);
    padding: 2rem;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 1.5rem;
}
.result {
    background: linear-gradient(135deg,#2563eb,#1e40af);
    padding: 1.5rem;
    border-radius: 16px;
    text-align: center;
    color: white;
}
h1, h2, h3 {
    color: #e5e7eb;
}
p {
    color: #cbd5e1;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Load Assets ----------------
@st.cache_resource
def load_assets():
    model = joblib.load("dbscan_model.pkl")
    scaler = joblib.load("scaler.pkl")
    data = pd.read_csv("customer_data.csv")
    return model, scaler, data

dbscan, scaler, df = load_assets()

X = df[["AnnualIncome", "SpendingScore"]]
X_scaled = scaler.transform(X)

# ---------------- Header ----------------
st.markdown("""
<div class="card">
    <h1>Customer Pattern Explorer</h1>
    <p>
        This system uses <b>DBSCAN</b> to discover natural customer behavior patterns
        without forcing them into predefined categories.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- Input ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.header("Simulate a Customer")

income = st.slider(
    "Annual Income (k$)",
    int(X["AnnualIncome"].min()),
    int(X["AnnualIncome"].max()),
    50
)

spending = st.slider(
    "Spending Score",
    int(X["SpendingScore"].min()),
    int(X["SpendingScore"].max()),
    50
)

analyze = st.button("Analyze Pattern", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Analysis ----------------
if analyze:
    labels = dbscan.fit_predict(X_scaled)
    df["Cluster"] = labels

    user_point = np.array([[income, spending]])
    user_scaled = scaler.transform(user_point)

    combined = np.vstack([X_scaled, user_scaled])
    combined_labels = dbscan.fit_predict(combined)
    user_cluster = combined_labels[-1]

    if user_cluster == -1:
        title = "Unusual Customer Pattern"
        desc = "This behavior does not align with any dense customer group."
    else:
        title = "Recognized Customer Pattern"
        desc = "This customer fits an existing behavioral group."

    st.markdown(
        f"""
        <div class="result">
            <h2>{title}</h2>
            <p>{desc}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ---------------- Visualization ----------------
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(
        df["AnnualIncome"],
        df["SpendingScore"],
        c=df["Cluster"],
        cmap="tab10",
        alpha=0.6
    )

    ax.scatter(
        income,
        spending,
        c="gold",
        s=200,
        edgecolors="black",
        label="Selected Customer"
    )

    ax.set_xlabel("Annual Income")
    ax.set_ylabel("Spending Score")
    ax.set_title("Customer Density Map")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

# ---------------- Footer ----------------
st.markdown("""
<p style="text-align:center;color:#94a3b8;font-size:12px;">
DBSCAN • Unsupervised Learning • Pattern Discovery
</p>
""", unsafe_allow_html=True)
