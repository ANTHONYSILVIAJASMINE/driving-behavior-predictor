import streamlit as st
import pandas as pd
import joblib
import gzip
import numpy as np

# Title
st.title("🚗 Driving Behavior Predictor")

# Description
st.write("This app predicts if a driving event occurred based on various road and driver conditions.")

# Load the compressed model
@st.cache_resource
def load_model():
    with gzip.open('driving_behavior_model_compressed.pkl.gz', 'rb') as f:
        model = joblib.load(f)
    return model

model = load_model()

# User input fields
st.header("Enter Driving Conditions")

lane_position = st.selectbox("Lane Position", ['Left', 'Center', 'Right'])
time_of_day = st.selectbox("Time of Day", ['Morning', 'Afternoon', 'Evening', 'Night'])
road_type = st.selectbox("Road Type", ['Highway', 'City', 'Rural'])
weather = st.selectbox("Weather Conditions", ['Clear', 'Rainy', 'Foggy', 'Snowy'])
traffic_density = st.selectbox("Traffic Density", ['Low', 'Medium', 'High'])

# Dummy input for numerical fields — adjust based on your actual features
speed = st.slider("Speed (km/h)", 0, 200, 60)
acceleration = st.slider("Acceleration (m/s²)", -10, 10, 0)

# Collect all inputs in a dataframe
input_data = pd.DataFrame({
    'Lane Position_' + lane_position: [1],
    'Time of Day_' + time_of_day: [1],
    'Road Type_' + road_type: [1],
    'Weather Conditions_' + weather: [1],
    'Traffic Density_' + traffic_density: [1],
    'Speed': [speed],
    'Acceleration': [acceleration]
})

# Add zeroes for missing one-hot encoded columns
all_columns = model.feature_names_in_
for col in all_columns:
    if col not in input_data.columns:
        input_data[col] = 0

# Reorder columns to match training data
input_data = input_data[all_columns]

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("🚨 Event likely occurred!")
    else:
        st.info("✅ No event detected.")

