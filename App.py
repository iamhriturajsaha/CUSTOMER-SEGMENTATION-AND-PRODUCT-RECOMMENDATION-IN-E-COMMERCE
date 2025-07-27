import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Page Config
st.set_page_config(page_title="Shopper Spectrum", layout="wide")

# Custom CSS Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;600;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(to right, #11998e, #38ef7d); /* Vibrant green gradient */
        background-attachment: fixed;
        color: #ffffff;
    }
    .main {
        background-color: rgba(0, 0, 0, 0.5);
        padding: 2rem;
        border-radius: 16px;
        backdrop-filter: blur(8px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
        margin: 20px;
    }
    h1, h2, h3, h4, h5, label {
        color: #ffffff !important;
    }
    .recommend-card {
        background: rgba(255, 255, 255, 0.1);
        color: #ffffff;
        padding: 15px;
        border-radius: 15px;
        margin-bottom: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        border-left: 6px solid #00ffcc;
    }
    .result-box {
        background: rgba(255, 255, 255, 0.2);
        padding: 20px;
        border-radius: 10px;
        border-left: 6px solid #00e676;
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        margin-top: 20px;
        color: white;
    }
    .stTextInput>div>div>input {
        background-color: rgba(255, 255, 255, 0.15) !important;
        color: #ffffff !important;
        border: 1px solid #00ffcc;
        border-radius: 8px;
        font-weight: 500;
        padding: 10px;
    }
    .stTextInput>div>div>input::placeholder {
        color: #dddddd !important;
    }
    .stButton>button {
        background-color: #00ffcc;
        color: #000000;
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: bold;
        transition: 0.3s ease-in-out;
        box-shadow: 0px 4px 14px rgba(0, 0, 0, 0.2);
    }
    .stButton>button:hover {
        background-color: #00e6b8;
    }
    </style>
""", unsafe_allow_html=True)

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("Retail.csv", encoding='ISO-8859-1')
    df.dropna(subset=['CustomerID'], inplace=True)
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    return df
df = load_data()

# Load models and mappings
try:
    scaler = joblib.load('rfm_scaler.pkl')
    kmeans = joblib.load('rfm_kmeans_model.pkl')
    rfm_labeled = pd.read_csv("rfm_clustered.csv")
except:
    st.error("📁 Please ensure 'rfm_scaler.pkl', 'rfm_kmeans_model.pkl', and 'rfm_clustered.csv' are available.")
    st.stop()

# Build product matrix for recommendations
product_matrix = df.pivot_table(index='CustomerID', columns='StockCode', values='Quantity', aggfunc='sum', fill_value=0)
product_sim_df = pd.DataFrame(cosine_similarity(product_matrix.T),
                              index=product_matrix.columns,
                              columns=product_matrix.columns)
product_map = df[['StockCode', 'Description']].drop_duplicates().dropna().set_index('StockCode')['Description'].to_dict()
name_to_code = {v: k for k, v in product_map.items()}

# UI HEADER
st.title("🛍️ Shopper Spectrum")
st.markdown("##### *Customer Segmentation & Product Recommendation*")
tab1, tab2 = st.tabs(["📦 Product Recommender", "🧠 Customer Segmentation"])

# -----------------------------------------------
# 1️⃣ Product Recommendation Module
# -----------------------------------------------
with tab1:
    st.subheader("🔗 Find Similar Products")
    product_name = st.text_input("Enter Product Name", placeholder="e.g. WHITE HANGING HEART T-LIGHT HOLDER")
    if st.button("✨ Get Recommendations"):
        if product_name not in name_to_code:
            st.error("❌ Product not found in dataset.")
        else:
            code = name_to_code[product_name]
            sims = product_sim_df[code].sort_values(ascending=False)[1:6]
            st.markdown("### 🔁 Top 5 Similar Products")
            for i, (stock_code, score) in enumerate(sims.items(), 1):
                desc = product_map.get(stock_code, "Unknown Product")
                st.markdown(f"""
                <div class="recommend-card">
                    <h5>🔹 {i}. {desc}</h5>
                    <p>Similarity Score: 
                        <span style='background-color:#27ae60; padding:2px 8px; border-radius:4px; color:white; font-weight:bold;'>{score:.2f}</span>
                    </p>
                </div>
                """, unsafe_allow_html=True)

# -----------------------------------------------
# 2️⃣ Customer Segmentation Module
# -----------------------------------------------
with tab2:
    st.subheader("📊 Predict Customer Segment")
    col1, col2, col3 = st.columns(3)
    with col1:
        recency = st.number_input("🕒 Recency (days)", min_value=0, max_value=1000, value=90)
    with col2:
        frequency = st.number_input("📦 Frequency (purchases)", min_value=0, max_value=500, value=10)
    with col3:
        monetary = st.number_input("💰 Monetary (total spend)", min_value=0.0, max_value=100000.0, value=500.0)
    if st.button("🚀 Predict Cluster"):
        input_data = np.array([[recency, frequency, monetary]])
        input_scaled = scaler.transform(input_data)
        cluster = kmeans.predict(input_scaled)[0]
        def label_cluster(row):
            if row['Recency'] <= rfm_labeled['Recency'].quantile(0.25) and row['Frequency'] >= rfm_labeled['Frequency'].quantile(0.75):
                return 'High-Value'
            elif row['Frequency'] >= rfm_labeled['Frequency'].quantile(0.5):
                return 'Regular'
            elif row['Frequency'] <= rfm_labeled['Frequency'].quantile(0.25) and row['Recency'] > rfm_labeled['Recency'].quantile(0.75):
                return 'At-Risk'
            else:
                return 'Occasional'
        temp_df = pd.DataFrame([[recency, frequency, monetary]], columns=['Recency', 'Frequency', 'Monetary'])
        segment = label_cluster(temp_df.iloc[0])
        st.markdown(f"""
        <div class="result-box">
            <h3>🧩 Predicted Segment: <b>{segment}</b></h3>
            <p>Cluster ID: <code>{cluster}</code></p>
        </div>
        """, unsafe_allow_html=True)
