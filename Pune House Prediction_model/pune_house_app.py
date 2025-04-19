import streamlit as st
import pickle
import numpy as np

# Load model and column data
model = pickle.load(open(r'C:\Users\HP\Desktop\IICET INTERNSHIP MACHINE LEARNING PROJECTS\Pune House Prediction_model\pune_house_model.pkl', 'rb'))
with open(r'C:\Users\HP\Desktop\IICET INTERNSHIP MACHINE LEARNING PROJECTS\Pune House Prediction_model\columns.pkl', 'rb') as f:
    data_columns = pickle.load(f)

columns = data_columns['data_columns']
locations = data_columns['locations']
area_types = data_columns['area_types']
availabilities = data_columns['availabilities']

# Prediction function
def predict_price(location, bhk, bath, balcony, sqft, area_type, availability):
    loc_index = -1
    area_index = -1
    avail_index = -1

    if location.lower() != 'other' and location in columns:
        loc_index = columns.index(location)

    if area_type in columns:
        area_index = columns.index(area_type)
    
    if availability in columns:
        avail_index = columns.index(availability)

    x = np.zeros(len(columns))
    x[0] = bath
    x[1] = balcony
    x[2] = bhk
    x[3] = sqft

    if loc_index >= 0:
        x[loc_index] = 1
    if area_index >= 0:
        x[area_index] = 1
    if avail_index >= 0:
        x[avail_index] = 1

    return round(model.predict([x])[0], 2)

# Streamlit UI
st.title("🏡 Pune House Price Predictor")

st.header("Enter the property details:")

location = st.selectbox("Location", ['Other'] + locations)
area_type = st.selectbox("Area Type", area_types)
availability = st.selectbox("Availability", availabilities)
bhk = st.slider("BHK (Bedrooms)", 1, 10, 2)
bath = st.slider("Bathrooms", 1, 5, 2)
balcony = st.slider("Balconies", 0, 5, 1)
sqft = st.number_input("Total Square Feet", min_value=300, max_value=10000, step=50)

if st.button("Predict Price"):
    price = predict_price(location, bhk, bath, balcony, sqft, area_type, availability)
    st.success(f"Estimated Price: ₹ {price} Lakhs")
