# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import joblib

# Load scaler & model
scaler = joblib.load("scaler.pkl")
model = joblib.load("california_model.pkl")

# Streamlit UI
st.title("California Housing Price Prediction")
st.markdown("Predict the median house value in California districts.")

# User inputs
MedInc = st.slider("Median Income (10k USD)", 0.5, 15.0, 3.0)
HouseAge = st.slider("House Age (years)", 1, 52, 20)
AveRooms = st.slider("Average Rooms", 1.0, 10.0, 5.0)
AveBedrms = st.slider("Average Bedrooms", 0.5, 5.0, 1.0)
Population = st.slider("Population", 100, 35000, 1000)
AveOccup = st.slider("Average Occupancy", 1.0, 7.0, 3.0)
Latitude = st.slider("Latitude", 32.0, 42.0, 36.0)
Longitude = st.slider("Longitude", -124.0, -114.0, -119.0)

if st.button("Predict"):
    # Convert to numpy array
    input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])
    
    # Scale input
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)
    st.success(f"Predicted Median House Value: ${prediction[0]*100000:.2f}")
