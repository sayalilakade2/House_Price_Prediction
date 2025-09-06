# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 17:29:11 2024

@author: sayali
"""
import os
import requests
import numpy as np
import streamlit as st
import joblib

# -------- Load Model --------
model_url = "https://github.com/sayalilakade2/House_Price_Prediction/raw/main/finalized_model.sav"
model_path = "finalized_model.sav"

# Download model only if not already present
if not os.path.exists(model_path):
    r = requests.get(model_url)
    if r.status_code == 200:
        with open(model_path, 'wb') as f:
            f.write(r.content)
    else:
        st.error("❌ Failed to download the model file")

# Load model
model = joblib.load(model_path)


# -------- Prediction Function --------
def predict_price(entries):
    try:
        input_data = np.array(entries).reshape(1, -1)
        predicted_price = model.predict(input_data)[0]
        return f"💰 The predicted price is Rs.{predicted_price:,.2f}"
    except Exception as e:
        return f"⚠️ Error: {str(e)}"


# -------- Streamlit UI --------
def main():
    st.title("🏠 House Price Prediction")

    features = [
        'Bedrooms:', 'Bathrooms:', 'Sqft Living:', 'Sqft Lot:', 'Floors:',
        'Waterfront:', 'View:', 'Condition:', 'Sqft Above:', 'Sqft Basement:',
        'Year Built:', 'Year Renovated:'
    ]

    entries = [st.number_input(f, min_value=0.0, step=1.0) for f in features]

    if st.button('Predict Price'):
        result = predict_price(entries)
        st.success(result)


if __name__ == '__main__':
    main()
