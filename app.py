import streamlit as st
import numpy as np
import joblib

# Load model & scaler
model = joblib.load("california_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("California House Price Prediction")

MedInc = st.number_input("Median Income", 0.0)
HouseAge = st.number_input("House Age", 0.0)
AveRooms = st.number_input("Average Rooms", 0.0)
AveBedrms = st.number_input("Average Bedrooms", 0.0)
Population = st.number_input("Population", 0.0)
AveOccup = st.number_input("Average Occupancy", 0.0)
Latitude = st.number_input("Latitude", 0.0)
Longitude = st.number_input("Longitude", 0.0)

input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])
scaled_data = scaler.transform(input_data)
prediction = model.predict(scaled_data)

if st.button("Predict"):
    st.success(f"Estimated House Price: ${prediction[0]:.2f}")
