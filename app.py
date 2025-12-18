import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="CustomerCluster",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- Global CSS ----------------
st.markdown("""
<style>
body {
    background-color: #0f172a;
}

.block-container {
    padding-top: 2rem;
}

h1, h2, h3 {
    color: #e5e7eb;
}

p {
    color: #cbd5e1;
}

.hero {
    padding: 2.5rem;
    border-radius: 16px;
    background: linear-gradient(135deg, #1e293b, #020617);
    margin-bottom: 2rem;
}

.card {
    background: rgba(30, 41, 59, 0.65);
    border-radius: 16px;
    padding: 1.5rem;
    border: 1px solid rgba(255,255,255,0.05);
}

.result {
    padding: 1.5rem;
    border-radius: 16px;
    text-align: center;
    background: linear-gradient(135deg, #2563eb, #1e40af);
}

.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 0.85rem;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Load Artifacts ----------------
@st.cache_resource
def load_assets():
    model = joblib.load("hierarchical_model.pkl")
    scaler = joblib.load("scaler.pkl")
    df = pd.read_csv("Mall_Customers.csv")
    return model, scaler, df

model, scaler, data = load_assets()

# ---------------- Preprocess Dataset ----------------
df = data.drop("CustomerID", axis=1)
df["Genre"] = df["Genre"].map({"Male": 1, "Female": 0})

# ---------------- Hero ----------------
st.markdown("""
<div class="hero">
    <h1>CustomerCluster</h1>
    <h3>Hierarchical Customer Segmentation</h3>
    <p>
        This application uses Agglomerative Hierarchical Clustering
        to group customers based on income, spending behavior, and demographics.
        Designed for analysis, not fortune-telling.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- Metrics ----------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Algorithm", "Agglomerative")
m2.metric("Linkage", "Ward")
m3.metric("Clusters", "5")
m4.metric("Data Type", "Retail")

# ---------------- Layout ----------------
left, center = st.columns([1.3, 2.7])

# -------- Left Panel --------
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Customer Profile")

    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.slider("Age", 18, 70, 30)
    income = st.slider("Annual Income (k$)", 10, 150, 60)
    spending = st.slider("Spending Score (1–100)", 1, 100, 50)

    analyze = st.button("Analyze Customer", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# -------- Center Panel --------
with center:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Cluster Analysis")

    if analyze:
        gender_val = 1 if gender == "Male" else 0

        user_input = np.array([[gender_val, age, income, spending]])
        user_scaled = scaler.transform(user_input)

        # Assign nearest cluster manually
        centers = []
        for c in range(model.n_clusters):
            centers.append(
                df[model.labels_ == c].mean().values
            )

        centers_scaled = scaler.transform(np.array(centers))
        distances = np.linalg.norm(centers_scaled - user_scaled, axis=1)
        user_cluster = np.argmin(distances)

        st.markdown(
            f"""
            <div class="result">
                <h2>Assigned to Cluster {user_cluster}</h2>
                <p>Based on similarity to existing customer groups</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Visualization
        df["Cluster"] = model.labels_

        fig, ax = plt.subplots(figsize=(7, 5))
        for c in sorted(df["Cluster"].unique()):
            subset = df[df["Cluster"] == c]
            ax.scatter(
                subset["Annual Income (k$)"],
                subset["Spending Score (1-100)"],
                label=f"Cluster {c}",
                alpha=0.6
            )

        ax.scatter(
            income,
            spending,
            s=220,
            c="gold",
            edgecolors="black",
            linewidths=1.5,
            label="Selected Customer"
        )

        ax.set_xlabel("Annual Income (k$)")
        ax.set_ylabel("Spending Score (1–100)")
        ax.set_title("Customer Segmentation Map")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Footer ----------------
st.markdown("""
<div class="footer">
    CustomerCluster • Hierarchical Clustering • Analytical Use Only
</div>
""", unsafe_allow_html=True)
