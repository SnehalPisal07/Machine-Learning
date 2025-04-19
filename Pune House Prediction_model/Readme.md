# 🏡 Pune House Price Prediction

This project predicts the price of houses in Pune using machine learning. The model is trained on housing features such as area, location, number of bedrooms, bathrooms, balconies, area type, and property availability status.

---

## 📌 Features

- Predicts house prices in Pune in Lakhs.
- Interactive web app using Streamlit.
- Accepts user input for key features like:
  - Location
  - Area Type (e.g., Built-up, Carpet, etc.)
  - Availability Status
  - Number of Bedrooms (BHK)
  - Number of Bathrooms
  - Number of Balconies
  - Square Foot Area

---

## 🧠 Model Details

- **Algorithm Used**: Linear Regression (can be replaced or improved later)
- **Training Dataset**: Cleaned and preprocessed Pune housing dataset
- **Preprocessing Includes**:
  - One-hot encoding for categorical features (Location, Area Type, Availability)
  - Handling missing values
  - Feature engineering for consistent inputs

---

## 📁 Project Structure

pune-house-prediction/ │ ├── app.py # Streamlit app ├── pune_house_model.pkl # Trained ML model ├── columns.pkl # Column metadata for encoding ├── README.md # This file └── requirements.txt # List of dependencies



---

## 🚀 How to Run

### 1. Clone the repository:
```bash
git clone https://github.com/yourusername/pune-house-prediction.git
cd pune-house-prediction
2. Install dependencies:

pip install -r requirements.txt
3. Start the Streamlit app:

streamlit run app.py
🧾 Example Input

Feature	Value
Location	Baner
Area Type	Built-up Area
Availability	Ready To Move
BHK	3
Bathrooms	2
Balconies	1
Total Sqft	1200
💡 Output: ₹ 89.5 Lakhs (Example prediction)

📦 Requirements

streamlit
numpy
pandas
scikit-learn
Put these in a requirements.txt file to set up easily.

✍️ Author
Made with ❤️ by [Your Name]

📌 Note
This is a basic prototype. Accuracy can be improved with:

Better feature selection

Advanced models (e.g., Random Forest, XGBoost)

Larger and cleaner dataset
