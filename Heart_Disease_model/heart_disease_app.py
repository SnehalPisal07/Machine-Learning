import pickle
import numpy as np
import streamlit as st

def infer_heart_disease(model_path, scaler_path, input_features):
    """
    Perform inferencing on the heart disease dataset.

    Parameters:
    - model_path (str): Path to the saved model pickle file.
    - scaler_path (str): Path to the saved scaler pickle file.
    - input_features (list): List of input features in the order:
      [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

    Returns:
    - str: "Disease Detected" if target is 1, otherwise "No Disease".
    """
    try:
        # Load the trained model and scaler
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)

        with open(scaler_path, 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)

        # Ensure the input features are in the correct format
        input_array = np.array([input_features]).reshape(1, -1)

        # Scale the input features
        scaled_input = scaler.transform(input_array)

        # Predict outcome
        prediction = model.predict(scaled_input)

        # Return result
        return "Disease Detected" if prediction[0] == 1 else "No Disease"

    except Exception as e:
        return f"An error occurred: {e}"

# Streamlit UI
st.title("Heart Disease Prediction App")

# User input fields
age = st.number_input("Age", min_value=1, max_value=120, value=45)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3, value=1)
trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=200, value=120)
chol = st.number_input("Cholesterol Level", min_value=100, max_value=600, value=233)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [0, 1])
restecg = st.number_input("Resting ECG Results (0-2)", min_value=0, max_value=2, value=1)
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250, value=150)
exang = st.selectbox("Exercise Induced Angina (0 = No, 1 = Yes)", [0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=6.2, value=2.3)
slope = st.number_input("Slope of the Peak Exercise ST Segment (0-2)", min_value=0, max_value=2, value=0)
ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy (0-4)", min_value=0, max_value=4, value=0)
thal = st.number_input("Thalassemia (0-3)", min_value=0, max_value=3, value=1)

# Perform prediction
if st.button("Predict"): 
    input_features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    model_path = "heart_disease_model.pkl"
    scaler_path = "heart_disease_scaler.pkl"
    result = infer_heart_disease(model_path, scaler_path, input_features)
    st.write("### Prediction Result:", result)
