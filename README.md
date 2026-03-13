# 🛍️ Customer Segmentation App

A machine learning web app that segments customers into distinct groups based on their demographics and purchasing behavior — built with KMeans Clustering, scikit-learn, and Streamlit.

---

## 📌 Overview

This project analyzes customer data to identify 7 distinct customer segments. It includes:
- Data preprocessing and feature scaling
- Dimensionality reduction using PCA
- KMeans clustering to group customers
- An interactive Streamlit web app to predict a customer's segment in real time

---

## 🧠 Customer Segments

| Cluster | Segment Name | Description |
|---------|-------------|-------------|
| 0 | Mature Mid-Tier Spenders | Older customers with moderate spending and decent income |
| 1 | Wealthy High Spenders | Top spenders with the highest income, highly engaged |
| 2 | Young Budget Shoppers | Lower income, price-sensitive, low spenders |
| 3 | Affluent Online Shoppers | High spenders who prefer web purchases |
| 4 | Average Mainstream Buyers | Middle-of-the-road across all metrics |
| 5 | High-Income Power Shoppers | Younger, high income, strong in both web and store |
| 6 | Older Occasional Shoppers | Moderate income, infrequent buyers, low engagement |

---

## 🗂️ Project Structure

customer-segmentation/
├── segmentation.py                       # Streamlit web app
├── customer_segmentation_model.pkl       # Trained KMeans model
├── customer_segmentation_scaler.pkl      # Fitted StandardScaler
├── requirements.txt                      # Python dependencies
└── README.md                             # Project documentation

---

## ⚙️ Installation & Setup

1. Clone the repository

2. Install dependencies
pip install -r requirements

3. Run the app
py -m streamlit run segmentation.py

The app will open at http://localhost:8501

---

## 📦 Requirements

streamlit
scikit-learn
joblib
pandas
numpy

---

## 📊 Features Used

- Age: Customer's age
- Income: Annual income
- Total_spending: Total amount spent
- NumWebPurchases: Number of online purchases
- NumStorePurchases: Number of in-store purchases
- NumWebVisitsMonth: Website visits per month
- Recency: Days since last purchase
