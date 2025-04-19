Iris Flower Species Prediction using Logistic Regression
This project uses the classic Iris dataset to train a Logistic Regression model that can classify Iris flowers into three species:
Setosa, Versicolor, and Virginica based on the input features.

📁 Project Structure
bash
Copy code
iris-flower-prediction/
├── iris_app.py                 # Streamlit app for interactive predictions
├── iris_dataset.ipynb          # Script to train and save the model and scaler
├── iris_logistic_model.pkl     # Trained Logistic Regression model (pickle)
├── iris_scaler.pkl             # Scaler used for feature normalization
└── README.md                   # Project overview and instructions
📊 Dataset
The Iris dataset contains 150 samples of iris flowers with the following features:

Sepal Length (cm)

Sepal Width (cm)

Petal Length (cm)

Petal Width (cm)

Target:

0 = Setosa

1 = Versicolor

2 = Virginica

🧪 Model Training
We use StandardScaler to normalize the data and train a LogisticRegression model:


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import pickle

After training, the model and scaler are saved using pickle:


with open("iris_logistic_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("iris_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
    
🧠 Inference Function
Use the saved model and scaler to make predictions on new input features:


from predict import predict_iris_species

result = predict_iris_species("iris_logistic_model.pkl", "iris_scaler.pkl", [5.1, 3.5, 1.4, 0.2])
print(result)

🌐 Run the Streamlit App

streamlit run iris_app.py
Enter the input features in the UI and it will predict the Iris species.

📦 Requirements
Install required libraries:


pip install -r requirements.txt
requirements.txt may include:


scikit-learn
numpy
streamlit

✅ Output Example

Input: [5.1, 3.5, 1.4, 0.2]
Prediction: Setosa
