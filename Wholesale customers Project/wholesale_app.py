import streamlit as st
import pickle
import numpy as np

# Title
st.title("Wholesale Customer Cluster Prediction")

st.markdown("""
This app predicts the **cluster** category of a wholesale customer based on their purchase behavior using a pre-trained Random Forest model.
""")

# Input fields
channel = st.selectbox("Channel", options=[1, 2], help="1 = Horeca (Hotel/Restaurant/Cafe), 2 = Retail")
region = st.selectbox("Region", options=[1, 2, 3], help="1 = Lisbon, 2 = Oporto, 3 = Other Region")
fresh = st.number_input("Fresh", min_value=0, step=100)
milk = st.number_input("Milk", min_value=0, step=100)
grocery = st.number_input("Grocery", min_value=0, step=100)
frozen = st.number_input("Frozen", min_value=0, step=100)
detergents_paper = st.number_input("Detergents_Paper", min_value=0, step=100)
delicassen = st.number_input("Delicassen", min_value=0, step=100)

# Inference function
def infer_wholesale_cluster(model_path, scaler_path, input_features):
    try:
        with open(r'C:\Users\HP\Desktop\IICET INTERNSHIP MACHINE LEARNING PROJECTS\Wholesale customers Project\random_forest_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)

        with open(r'C:\Users\HP\Desktop\IICET INTERNSHIP MACHINE LEARNING PROJECTS\Wholesale customers Project\scaler.pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)

        input_array = np.array([input_features]).reshape(1, -1)
        scaled_input = scaler.transform(input_array)
        prediction = model.predict(scaled_input)[0]

        return f"🟢 **Predicted Cluster:** {prediction}"
    except Exception as e:
        return f"🔴 An error occurred: {e}"

# Button to trigger prediction
if st.button("Predict Cluster"):
    input_values = [channel, region, fresh, milk, grocery, frozen, detergents_paper, delicassen]
    result = infer_wholesale_cluster(
        model_path="random_forest_model.pkl",
        scaler_path="scaler.pkl",
        input_features=input_values
    )
    st.success(result)
