#this streamlit code is developed with the help of ai.
#this will help to graphically predict users into clusters
import streamlit as st
import joblib
import numpy as np

with open(r"C:\Users\hp\Desktop\customer segmentation\kmeans11.pkl", "rb") as f:
    kmeans_model = joblib.load(f)

with open(r"C:\Users\hp\Desktop\customer segmentation\scaler11.pkl", "rb") as f:
    scaler = joblib.load(f)

def predict_customer_cluster(income, score, scaler, model):
    input_data = np.array([[income, score]])
    input_scaled = scaler.fit_transform(input_data)
    cluster = model.predict(input_scaled)
    return cluster[0]

def get_cluster_message(cluster_id):
    messages = {
        0: "🟡 Cluster 0: Average income and average spending. Average Customers.",
        1: "🟢 Cluster 1: High income, high spending. Premium Customers.",
        2: "🔵 Cluster 2: Low income, low spending. Budget Friendly Customers.",
        3: "🟠 Cluster 3: High income but low spending. Cautious Customers.",
        4: "🔴 Cluster 4: Low income but high spending. Impulsive or trend-driven Customers.",
        5: "⚪ Cluster 5: Special segment — Further analysis may be required.(not yet decided due to small amount of data)"
    }
    return messages.get(cluster_id, "Unknown cluster")


st.title("🧠 Customer Segmentation using K-Means")
st.write("Enter customer details to predict their segment:")

#age = st.number_input("Age", min_value=1, max_value=100, value=30)
income = st.number_input("Annual Income (k$)", min_value=0.0, value=50.0)
score = st.number_input("Spending Score (1-100)", min_value=0.0, max_value=100.0, value=50.0)

if st.button("Predict Cluster"):
    cluster_id = predict_customer_cluster(income, score, scaler, kmeans_model)
    st.success(f"✅ The customer belongs to **Cluster {cluster_id}**")
    st.info(get_cluster_message(cluster_id))
