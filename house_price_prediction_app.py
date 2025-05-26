import streamlit as st
import pandas as pd
import pickle

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("🏠 House Price Predictor")

st.markdown("Enter the details below to predict the price of a house.")

income = st.number_input("Avg. Area Income (₹)", min_value=0.0, value=80000.0, step=100.0, key="income")
house_age = st.number_input("Avg. Area House Age (years)", min_value=0.0, value=4.0, step=1.0, key="house_age")
rooms = st.number_input("Avg. Area Number of Rooms", min_value=0.0, value=6.0, step=1.0, key="rooms")
bedrooms = st.number_input("Avg. Area Number of Bedrooms", min_value=0.0, step=1.0, value=3.0, key="bedrooms")
population = st.number_input("Area Population", min_value=0.0, value=12000.0, step=100.0, key="population")

if st.button("Predict Price", key="predict_btn"):
    input_data = pd.DataFrame([{
        'Avg. Area Income': income,
        'Avg. Area House Age': house_age,
        'Avg. Area Number of Rooms': rooms,
        'Avg. Area Number of Bedrooms': bedrooms,
        'Area Population': population
    }])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    st.success(f"🏷️ Predicted Price: **₹{prediction[0]:,.2f}**")
