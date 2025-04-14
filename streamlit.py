import pandas as pd
import numpy as np
import streamlit as st
import joblib

# Load your trained model
model = joblib.load("driving_behavior_model.pkl.gz")

# List of expected columns during training (based on your feature engineering)
expected_columns = [
    'Speed', 'Acceleration', 'Steering Angle', 'Braking', 'Lane Position_Left', 'Lane Position_Center', 
    'Lane Position_Right', 'Time of Day_Morning', 'Time of Day_Afternoon', 'Time of Day_Evening', 
    'Time of Day_Night', 'Road Type_Highway', 'Road Type_City', 'Road Type_Rural', 'Weather Conditions_Clear', 
    'Weather Conditions_Rainy', 'Weather Conditions_Foggy', 'Weather Conditions_Snowy', 
    'Traffic Density_Low', 'Traffic Density_Moderate', 'Traffic Density_High'
]

# Function to map numerical predictions to behavior labels
def map_prediction_to_behavior(prediction):
    behavior_map = {0: 'Cautious', 1: 'Normal', 2: 'Aggressive'}
    return behavior_map.get(prediction, 'Unknown')

# Streamlit form for user input
with st.form("input_form"):
    st.header("Driving Behavior Prediction")

    # User inputs for each feature
    speed = st.slider('Speed (km/h)', min_value=0, max_value=200, value=50)
    acceleration = st.slider('Acceleration (m/s²)', min_value=-5, max_value=5, value=0)
    steering_angle = st.slider('Steering Angle (degrees)', min_value=-30, max_value=30, value=0)
    braking = st.slider('Braking (m/s²)', min_value=-5, max_value=5, value=0)
    lane_position_left = st.radio('Lane Position: Left', [0, 1])
    lane_position_center = st.radio('Lane Position: Center', [0, 1])
    lane_position_right = st.radio('Lane Position: Right', [0, 1])
    time_of_day = st.selectbox('Time of Day', ['Morning', 'Afternoon', 'Evening', 'Night'])
    road_type = st.selectbox('Road Type', ['Highway', 'City', 'Rural'])
    weather_conditions = st.selectbox('Weather Conditions', ['Clear', 'Rainy', 'Foggy', 'Snowy'])
    traffic_density = st.selectbox('Traffic Density', ['Low', 'Moderate', 'High'])

    # Submit button
    submit_button = st.form_submit_button("Predict")

# Process the input when the form is submitted
if submit_button:
    # Create a dictionary with the input data
    input_data = {
        'Speed': speed,
        'Acceleration': acceleration,
        'Steering Angle': steering_angle,
        'Braking': braking,
        'Lane Position_Left': lane_position_left,
        'Lane Position_Center': lane_position_center,
        'Lane Position_Right': lane_position_right,
        'Time of Day_Morning': 1 if time_of_day == 'Morning' else 0,
        'Time of Day_Afternoon': 1 if time_of_day == 'Afternoon' else 0,
        'Time of Day_Evening': 1 if time_of_day == 'Evening' else 0,
        'Time of Day_Night': 1 if time_of_day == 'Night' else 0,
        'Road Type_Highway': 1 if road_type == 'Highway' else 0,
        'Road Type_City': 1 if road_type == 'City' else 0,
        'Road Type_Rural': 1 if road_type == 'Rural' else 0,
        'Weather Conditions_Clear': 1 if weather_conditions == 'Clear' else 0,
        'Weather Conditions_Rainy': 1 if weather_conditions == 'Rainy' else 0,
        'Weather Conditions_Foggy': 1 if weather_conditions == 'Foggy' else 0,
        'Weather Conditions_Snowy': 1 if weather_conditions == 'Snowy' else 0,
        'Traffic Density_Low': 1 if traffic_density == 'Low' else 0,
        'Traffic Density_Moderate': 1 if traffic_density == 'Moderate' else 0,
        'Traffic Density_High': 1 if traffic_density == 'High' else 0
    }

    # Convert the input data to a DataFrame
    input_df = pd.DataFrame([input_data])

    # Ensure the input matches the exact columns the model was trained on
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0  # If the column is missing, set it to 0

    input_df = input_df[expected_columns]  # Ensure the order matches the expected one

    # Make predictions
    prediction = model.predict(input_df)[0]
    behavior = map_prediction_to_behavior(prediction)

    # Display the result
    st.subheader(f"Predicted Driving Behavior: {behavior}")
