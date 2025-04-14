import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gzip

# Load the compressed model
@st.cache_resource
def load_model():
    with gzip.open('driving_behavior_model.pkl.gz', 'rb') as f:
        return joblib.load(f)

model = load_model()

st.title("🚗 Driving Behavior Predictor")

# Input fields
speed = st.slider("Speed (km/h)", 0, 200, 60)
acceleration = st.slider("Acceleration (m/s²)", -10, 10, 0)
lane_position = st.selectbox("Lane Position", ['Left', 'Center', 'Right'])
time_of_day = st.selectbox("Time of Day", ['Morning', 'Afternoon', 'Evening', 'Night'])
road_type = st.selectbox("Road Type", ['Highway', 'City', 'Rural'])
weather = st.selectbox("Weather Conditions", ['Clear', 'Rainy', 'Foggy', 'Snowy'])
traffic = st.selectbox("Traffic Density", ['Low', 'Moderate', 'High'])

# Encode inputs as one-hot (ensure all features are included here)
input_dict = {
    'Speed': speed,
    'Acceleration': acceleration,
    'Lane Position_Left': 1 if lane_position == 'Left' else 0,
    'Lane Position_Center': 1 if lane_position == 'Center' else 0,
    'Lane Position_Right': 1 if lane_position == 'Right' else 0,
    'Time of Day_Morning': 1 if time_of_day == 'Morning' else 0,
    'Time of Day_Afternoon': 1 if time_of_day == 'Afternoon' else 0,
    'Time of Day_Evening': 1 if time_of_day == 'Evening' else 0,
    'Time of Day_Night': 1 if time_of_day == 'Night' else 0,
    'Road Type_Highway': 1 if road_type == 'Highway' else 0,
    'Road Type_City': 1 if road_type == 'City' else 0,
    'Road Type_Rural': 1 if road_type == 'Rural' else 0,
    'Weather Conditions_Clear': 1 if weather == 'Clear' else 0,
    'Weather Conditions_Rainy': 1 if weather == 'Rainy' else 0,
    'Weather Conditions_Foggy': 1 if weather == 'Foggy' else 0,
    'Weather Conditions_Snowy': 1 if weather == 'Snowy' else 0,
    'Traffic Density_Low': 1 if traffic == 'Low' else 0,
    'Traffic Density_Moderate': 1 if traffic == 'Moderate' else 0,
    'Traffic Density_High': 1 if traffic == 'High' else 0
}

# Create DataFrame with correct format
input_df = pd.DataFrame([input_dict])

# Get the model's expected feature names
expected_columns = model.feature_names_in_

# Ensure all expected features are included and missing features are set to 0
for col in expected_columns:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[expected_columns]

# Predict
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.success("🚨 Event likely occurred!")
    else:
        st.info("✅ No event detected.")
