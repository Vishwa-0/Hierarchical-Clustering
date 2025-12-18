import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Customer Insight Engine",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------- Styling (DBSCAN UI inspired) ----------------
st.markdown("""
<style>
body {
    background-color: #020617;
}
.card {
    background: rgba(30,41,59,0.75);
    backdrop-filter: blur(8px);
    padding: 2rem;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 1.5rem;
}
.result {
    background: linear-gradient(135deg,#16a34a,#166534);
    padding: 1.5rem;
    border-radius: 16px;
    text-align: center;
}
h1, h2, h3 {
    color: #e5e7eb;
}
p {
    color: #cbd5e1;
}
.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Load Assets ----------------
@st.cache_resource
def load_assets():
    model = joblib.load("hierarchical_model.pkl")
    scaler = joblib.load("scaler.pkl")
    df = pd.read_csv("Mall_Customers.csv")
    return model, scaler, df

model, scaler, df = load_assets()

# ---------------- Preprocess Dataset ----------------
df = df.drop("CustomerID", axis=1)
df["Genre"] = df["Genre"].map({"Male": 1, "Female": 0})

X_scaled = scaler.transform(df)

# Assign training clusters
df["Cluster"] = model.labels_

# Compute cluster centroids
centroids = df.groupby("Cluster").mean().values
centroids_scaled = scaler.transform(centroids)

# ---------------- Human-Friendly Cluster Names ----------------
CLUSTER_PROFILES = {
    0: ("Value Seekers", "Careful spenders focused on essential purchases."),
    1: ("Premium Customers", "High-income customers with confident spending habits."),
    2: ("Impulse Buyers", "Lower income but high enthusiasm for spending."),
    3: ("Balanced Shoppers", "Stable income and consistent purchasing behavior."),
    4: ("Low Engagement Customers", "Minimal spending and limited product interaction.")
}

# ---------------- Header ----------------
st.markdown("""
<div class="card">
    <h1>Customer Insight Engine</h1>
    <p>
        This application analyzes customer attributes and assigns them
        to a behavioral segment using <b>Hierarchical Clustering</b>.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- User Input ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.header("Customer Profile")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 70, 30)
income = st.slider("Annual Income (k$)", 10, 150, 50)
spending = st.slider("Spending Score (1–100)", 1, 100, 50)

analyze = st.button("Analyze Customer", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Assignment Logic ----------------
if analyze:
    gender_val = 1 if gender == "Male" else 0
    user = np.array([[gender_val, age, income, spending]])
    user_scaled = scaler.transform(user)

    distances = np.linalg.norm(centroids_scaled - user_scaled, axis=1)
    assigned_cluster = int(np.argmin(distances))

    title, description = CLUSTER_PROFILES[assigned_cluster]

    st.markdown(
        f"""
        <div class="result">
            <h2>{title}</h2>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------- Footer ----------------
st.markdown("""
<div class="footer">
    Hierarchical Clustering • Unsupervised Learning • Behavioral Segmentation
</div>
""", unsafe_allow_html=True)
