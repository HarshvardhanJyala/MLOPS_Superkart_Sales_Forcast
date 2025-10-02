
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# -----------------------------
# Load the trained model from Hugging Face Hub
# -----------------------------
model_path = hf_hub_download(
    repo_id="JyalaHarsha-2025/MLOPS_Superkart_Sales_Forcast",
    filename="superkart_sales_model_v1.joblib"
)
model = joblib.load(model_path)

# -----------------------------
# Streamlit App UI
# -----------------------------
st.set_page_config(page_title="Superkart Sales Forecasting", layout="centered")
st.title("🛒 Superkart Sales Prediction App")
st.write("""
This app predicts the **total sales revenue** for a specific product in a specific store based on various attributes. 
Fill out the product and store details below to get a sales forecast.
""")

# ---- PRODUCT DETAILS SECTION ----
st.header("📦 Product Information")

product_weight = st.number_input("Product Weight (kg)", min_value=0.0, value=1.5, step=0.1)
sugar_content = st.selectbox("Sugar Content", ["Low", "Regular", "No Sugar"])
allocated_area = st.slider("Product Allocated Area (0 - 1)", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
product_type = st.selectbox("Product Type", [
    "Meat", "Snack Foods", "Hard Drinks", "Dairy", "Canned", "Soft Drinks",
    "Health and Hygiene", "Baking Goods", "Bread", "Breakfast", "Frozen Foods",
    "Fruits and Vegetables", "Household", "Seafood", "Starchy Foods", "Others"
])
product_mrp = st.number_input("Product MRP (₹)", min_value=1.0, value=100.0, step=1.0)

# ---- STORE DETAILS SECTION ----
st.header("🏬 Store Information")

store_est_year = st.number_input("Store Establishment Year", min_value=1980, max_value=2025, value=2010)
store_size = st.selectbox("Store Size", ["High", "Medium", "Low"])
city_type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
store_type = st.selectbox("Store Type", ["Departmental Store", "Supermarket Type 1", "Supermarket Type 2", "Food Mart"])

# ---- Assemble input data for prediction ----
input_data = pd.DataFrame([{
    "Product_Weight": product_weight,
    "Product_Sugar_Content": sugar_content,
    "Product_Allocated_Area": allocated_area,
    "Product_Type": product_type,
    "Product_MRP": product_mrp,
    "Store_Establishment_Year": store_est_year,
    "Store_Size": store_size,
    "Store_Location_City_Type": city_type,
    "Store_Type": store_type
}])

# ---- Display Input Summary ----
st.subheader("📥 Input Summary")
st.dataframe(input_data)

# ---- Predict Button ----
if st.button("Predict Sales"):
    try:
        prediction = model.predict(input_data)[0]
        st.subheader("📈 Predicted Sales Revenue")
        st.success(f"Estimated Product-Store Sales: ₹{prediction:,.2f}")
    except Exception as e:
        st.error("❌ Prediction failed.")
        st.exception(e)
