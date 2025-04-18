import streamlit as st
import pickle
import numpy as np

# Prediction function
def predict_iris_species(model_path, scaler_path, input_features):
    try:
        with open(r'C:\Users\HP\Desktop\IICET INTERNSHIP MACHINE LEARNING PROJECTS\Iris model\iris_logistic_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open(r'C:\Users\HP\Desktop\IICET INTERNSHIP MACHINE LEARNING PROJECTS\Iris model\iris_scaler.pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)

        input_array = np.array([input_features])
        input_scaled = scaler.transform(input_array)

        prediction = model.predict(input_scaled)[0]
        species = ['setosa', 'versicolor', 'virginica']
        return f"🌼 Predicted Iris species: **{species[prediction]}**"
    except Exception as e:
        return f"An error occurred during prediction: {e}"

# Streamlit UI
st.title("🌸 Iris Species Predictor")
st.write("Enter the flower measurements to predict the Iris species.")

# Input fields
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, step=0.1)

# Prediction button
if st.button("Predict"):
    input_features = [sepal_length, sepal_width, petal_length, petal_width]
    model_path = "iris_logistic_model.pkl"
    scaler_path = "iris_scaler.pkl"
    result = predict_iris_species(model_path, scaler_path, input_features)
    st.success(result)
