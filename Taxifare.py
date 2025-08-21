import streamlit as st
import pandas as pd
from datetime import datetime
import pickle
import numpy as np
import os

# Load the model
#model_path = os.getenv("MODEL_PATH", "C:\Users\kgovindarajudev\gbr_model.pkl")
#model_path = os.getenv("MODEL_PATH", "C:\\Users\\kgovindarajudev\\gbr_model.pkl")
#model_path = os.getenv("MODEL_PATH", f"C:\Users\kgovindarajudev\gbr_model.pkl")
#with open('model_path.pkl', 'rb') as file:

with open('C:/Users/kgovindarajudev/gbr_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Streamlit UI
st.title("NYC Taxi Fare Prediction")
st.subheader("Enter trip details:")

# Input fields
passenger_count = st.number_input("Passenger Count", min_value=1, max_value=10, value=1)
payment_type = st.selectbox("Payment Type", options=['Credit Card', 'Cash', 'No Charge', 'Dispute', 'Unknown'])
trip_distance = st.number_input("Trip Distance (miles)", min_value=0.0, step=0.1)
trip_duration_sec = st.number_input("Trip Duration (seconds)", min_value=0)
pickup_am_pm = st.selectbox("Pickup Time", options=['AM', 'PM'])
day_of_week = st.selectbox("Day of Week", options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
month = st.selectbox("Month", options=list(range(1, 13)))
hour = st.slider("Hour of Day", min_value=0, max_value=23)
is_night = st.selectbox("Is Night?", options=[0, 1])

# Convert categorical variables (you must use the same preprocessing as training)
# This is a simplified example assuming you used label encoding or similar
def preprocess_inputs():
    payment_map = {'Credit Card': 0, 'Cash': 1, 'No Charge': 2, 'Dispute': 3, 'Unknown': 4}
    pickup_map = {'AM': 0, 'PM': 1}
    day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
               'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    
    inputs = [
        passenger_count,
        payment_map[payment_type],
        trip_distance,
        trip_duration_sec,
        pickup_map[pickup_am_pm],
        day_map[day_of_week],
        month,
        hour,
        is_night
    ]
    return np.array(inputs).reshape(1, -1)

# Predict
if st.button("Predict Total Fare"):
    input_data = preprocess_inputs()
    prediction = loaded_model.predict(input_data)[0]
    st.success(f"Predicted Total Amount: ${prediction:.2f}")


