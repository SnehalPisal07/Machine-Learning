import pickle
import numpy as np
import streamlit as st

def predict_diabetes(input_features):
    """
    Load model and scaler to make predictions.
    
    Parameters:
    - input_features (list): List of input features in the order:
      [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
    
    Returns:
    - str: "Survived" if prediction is 1, otherwise "Not Survived".
    """
    try:
        # Load the trained model and scaler
        with open(r"C:\Users\HP\Desktop\IICET INTERNSHIP MACHINE LEARNING PROJECTS\diabetes_model.pkl", "rb") as model_file:
            model = pickle.load(model_file)
        
        with open(r"C:\Users\HP\Desktop\IICET INTERNSHIP MACHINE LEARNING PROJECTS\diabetes_scaler.pkl", "rb") as scaler_file:
            scaler = pickle.load(scaler_file)
        
        # Ensure the input features are in the correct format
        input_array = np.array([input_features]).reshape(1, -1)

        # Scale the input features
        scaled_input = scaler.transform(input_array)

        # Predict outcome
        prediction = model.predict(scaled_input)

        # Return result
        return "Diabetes detected" if prediction[0] == 1 else "No Diabetes"
    
    except Exception as e:
        return f"An error occurred: {e}"

# Streamlit UI
st.title("Diabetes Prediction App")

# User input fields
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=100)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin Level", min_value=0, max_value=1000, value=79)
bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age", min_value=0, max_value=120, value=30)

# Perform prediction
if st.button("Predict"):
    input_features = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]
    result = predict_diabetes(input_features)
    st.write("### Prediction Result:", result)
