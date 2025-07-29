import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Load the saved model (using st.cache_resource for efficiency)
@st.cache_resource 
def load_ann_model():
    return load_model("model1.keras") 

model = load_ann_model()

st.title("Employee Performance Prediction")
st.write("Enter the employee's details to predict their performance scale (1-3):")

# Input widgets for features
extension_requests_count = st.slider(
    "Extension Requests Count", min_value=0, max_value=10, value=2
)
extension_approval_rate = st.number_input(
    "Extension Approval Rate", min_value=0.0, max_value=1.0, value=0.75, step=0.01
)
feedback_360_score = st.slider(
    "360 Feedback Score", min_value=1, max_value=5, value=3
)
overtime_hours = st.number_input(
    "Overtime Hours", min_value=0, max_value=500, value=50, step=10
)
absenteeism_rate = st.number_input(
    "Absenteeism Rate", min_value=0.0, max_value=100.0, value=5.0, step=0.1
)
weighted_task_completion_rate = st.number_input(
    "Weighted Task Completion Rate",
    min_value=0.0,
    max_value=100.0,
    value=85.0,
    step=0.1,
)
manager_appraisal_score = st.slider(
    "Manager Appraisal Score", min_value=1, max_value=10, value=7
)

# Button to trigger prediction
if st.button("Predict Performance"):
    # Prepare input data as a NumPy array (match model's expected input shape)
    input_data = np.array(
        [
            [
                extension_requests_count,
                extension_approval_rate,
                feedback_360_score,
                overtime_hours,
                absenteeism_rate,
                weighted_task_completion_rate,
                manager_appraisal_score,
            ]
        ]
    )

    # Make the prediction
    prediction = model.predict(input_data)

    # Map prediction to performance labels
    performance_labels = ["low performance", "medium performance", "high performance"]
    predicted_index = np.argmax(prediction, axis=1)[0]
    predicted_label = performance_labels[predicted_index]

    st.success(f"The predicted performance scale is: {predicted_label}")

