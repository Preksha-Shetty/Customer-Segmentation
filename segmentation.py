import streamlit as st
import numpy as np
import pandas as pd
import joblib

kmeans = joblib.load('customer_segmentation_model.pkl')
scaler = joblib.load('customer_segmentation_scaler.pkl')

st.title("Customer Segmentation App")
st.write("Enter customer details to predict their segment:")

age = st.number_input("Age", min_value=18, max_value=100, value=35)
income = st.number_input("Income", min_value=0, max_value=20000, value=5000)
Total_Spending = st.number_input("Total Spending (sum of all purchases)", min_value=0, max_value=1000, value=50)  
num_web_purchses = st.number_input("Number of Web Purchases", min_value=0, max_value=100, value=5)
num_store_purchses = st.number_input("Number of Store Purchases", min_value=0, max_value=100, value=5)
web_visits = st.number_input("Number of Web Visits per month", min_value=0, max_value=100, value=10)
recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=365, value=30)

input_data = pd.DataFrame({
    'Age': [age],
    'Total_spending': [Total_Spending],
    'Income': [income],
    'NumWebPurchases': [num_web_purchses],
    'NumStorePurchases': [num_store_purchses],
    'NumWebVisitsMonth': [web_visits],
    'Recency': [recency]
})

input_scaled = scaler.transform(input_data)
if st.button("Predict Segment"):
    cluster = kmeans.predict(input_scaled)[0]
    st.success(f"The customer belongs to Segment: Cluster {cluster}")

st.write("""
Cluster 0: Mature Mid-Tier Spenders
Cluster 1: Wealthy High Spenders
Cluster 2: Young Budget Shoppers
Cluster 3: Affluent Online Shoppers
Cluster 4: Average Mainstream Buyers
Cluster 5: Older Occasional Shoppers
Cluster 6: Young Occasional Shoppers
""")
           