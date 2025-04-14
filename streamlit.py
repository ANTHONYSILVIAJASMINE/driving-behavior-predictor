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

# Get all columns from model
input_df = pd.DataFrame([input_dict])
for col in model.feature_names_in_:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[model.feature_names_in_]

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.success("🚨 Event likely occurred!")
    else:
        st.info("✅ No event detected.")
