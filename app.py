import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Customer Insight Engine",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------- Styling ----------------
st.markdown("""
<style>
body {
    background-color: #0f172a;
}
h1, h2, h3 {
    color: #e5e7eb;
}
p {
    color: #cbd5e1;
}
.card {
    background: rgba(30, 41, 59, 0.75);
    backdrop-filter: blur(8px);
    padding: 2rem;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 1.5rem;
}
.result {
    background: linear-gradient(135deg, #16a34a, #166534);
    padding: 1.5rem;
    border-radius: 16px;
    text-align: center;
}
.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Load Model & Scaler ----------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("hierarchical_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# ---------------- Cluster → Human Labels ----------------
CLUSTER_LABELS = {
    0: {
        "title": "Value Seekers",
        "description": "Careful spenders who prioritize value and essential purchases."
    },
    1: {
        "title": "Premium Customers",
        "description": "High-income customers with confident and frequent spending habits."
    },
    2: {
        "title": "Impulse Buyers",
        "description": "Enthusiastic shoppers with high spending despite lower income."
    },
    3: {
        "title": "Balanced Shoppers",
        "description": "Customers with stable income and consistent purchasing behavior."
    },
    4: {
        "title": "Low Engagement Customers",
        "description": "Minimal spending activity and limited interaction with products."
    }
}

# ---------------- Header ----------------
st.markdown("""
<div class="card">
    <h1>Customer Insight Engine</h1>
    <p>
        This system analyzes customer attributes and assigns them to a
        behavioral segment using hierarchical clustering.
        No forced categories. No assumptions. Just patterns.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- Input Section ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.header("Customer Profile")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 70, 30)
income = st.slider("Annual Income (k$)", 10, 150, 50)
spending = st.slider("Spending Score (1–100)", 1, 100, 50)

analyze = st.button("Analyze Customer", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Prediction Logic ----------------
if analyze:
    gender_encoded = 1 if gender == "Male" else 0

    user_data = np.array([[gender_encoded, age, income, spending]])
    user_scaled = scaler.transform(user_data)

    # Reconstruct cluster centers manually
    df = pd.read_csv("Mall_Customers.csv")
    df = df.drop("CustomerID", axis=1)
    df["Genre"] = df["Genre"].map({"Male": 1, "Female": 0})
    X_scaled = scaler.transform(df)

    df["Cluster"] = model.labels_
    centers = df.groupby("Cluster").mean().values
    centers_scaled = scaler.transform(centers)

    distances = np.linalg.norm(centers_scaled - user_scaled, axis=1)
    assigned_cluster = int(np.argmin(distances))

    segment = CLUSTER_LABELS.get(assigned_cluster)

    st.markdown(
        f"""
        <div class="result">
            <h2>{segment['title']}</h2>
            <p>{segment['description']}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------- Footer ----------------
st.markdown("""
<div class="footer">
    Hierarchical Clustering • Customer Segmentation • Educational Use
</div>
""", unsafe_allow_html=True)
