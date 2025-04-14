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

# Encode inputs as one-hot
input_dict = {
    'Speed': speed,
    'Acceleration': acceleration,
    f'Lane Position_{lane_position}': 1,
    f'Time of Day_{time_of_day}': 1,
    f'Road Type_{road_type}': 1,
    f'Weather Conditions_{weather}': 1,
    f'Traffic Density_{traffic}': 1
}

# --- FIX: Explicit list of expected features (same as used in training)
all_columns = [
    'Speed',
    'Acceleration',
    'Lane Position_Left',
    'Lane Position_Center',
    'Lane Position_Right',
    'Time of Day_Morning',
    'Time of Day_Afternoon',
    'Time of Day_Evening',
    'Time of Day_Night',
    'Road Type_Highway',
    'Road Type_City',
    'Road Type_Rural',
    'Weather Conditions_Clear',
    'Weather Conditions_Rainy',
    'Weather Conditions_Foggy',
    'Weather Conditions_Snowy',
    'Traffic Density_Low',
    'Traffic Density_Moderate',
    'Traffic Density_High'
]

# Create DataFrame with correct format
input_df = pd.DataFrame([input_dict])

# Ensure all required columns are present
for col in all_columns:
    if col not in input_df.columns:
        input_df[col] = 0  # Add missing columns with 0 (indicating no presence of the feature)

input_df = input_df[all_columns]  # Reorder columns to match model's expected input order

# Predict
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.success("🚨 Event likely occurred!")
    else:
        st.info("✅ No event detected.")
