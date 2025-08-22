import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the encoders and model
with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

with open('gbr_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Streamlit UI
st.title("NYC Taxi Fare Prediction")
st.subheader("Enter trip details:")

# Input fields
passenger_count = st.number_input("Passenger Count", min_value=1, max_value=10, value=1)
payment_type = st.selectbox("Payment Type", options=['Credit Card', 'Cash', 'No Charge', 'Dispute', 'Unknown'])
trip_distance = st.number_input("Trip Distance (miles)", min_value=0.0, step=0.1)
trip_duration_sec = st.number_input("Trip Duration (seconds)", min_value=0)
hour = st.slider("Hour of Day", min_value=0, max_value=23)
day_of_week = st.selectbox("Day of Week", options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

# Convert categorical variables and apply conditions
def preprocess_inputs():
    # Determine if it's night (is_night) and AM/PM (pickup_am_pm)
    if hour >= 20:  # Between 8 PM and midnight
        is_night = 'Yes'
    else:
        is_night = 'No'
        
    if 17 <= hour <= 23:  # Between 5 PM and 11:59 PM
        pickup_am_pm = 'PM'
    else:  # Otherwise AM
        pickup_am_pm = 'AM'

    # Use the payment_map for the 'payment_type' directly (no need for label encoder here)
    payment_map = {'Credit Card': 0, 'Cash': 1, 'No Charge': 2, 'Dispute': 3, 'Unknown': 4}
    payment_type_encoded = payment_map[payment_type]  # Map directly to an integer

    # Map day_of_week to integer using day_map (no label encoding needed here either)
    day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
               'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    day_of_week_encoded = day_map[day_of_week]  # Direct mapping to integer

    # Label encoding for the following fields (those that need encoding)
    pickup_am_pm_encoded = label_encoder.transform([pickup_am_pm])[0]  # Encode AM/PM
    is_night_encoded = label_encoder.transform([is_night])[0]  # Encode Yes/No

    # Prepare the final input array
    inputs = [
        trip_distance,
        trip_duration_sec,
        hour,
        payment_type_encoded,  # Now it's a number from payment_map, not label_encoder
        passenger_count,
        day_of_week_encoded,  # Directly encoded as an integer
        is_night_encoded,
        pickup_am_pm_encoded
    ]
    
    return np.array(inputs).reshape(1, -1)


# Predict the total fare
if st.button("Predict Total Fare"):
    input_data = preprocess_inputs()
    prediction = loaded_model.predict(input_data)[0]
    st.success(f"Predicted Total Amount: ${prediction:.2f}")
