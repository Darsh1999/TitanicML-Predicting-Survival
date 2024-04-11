from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

# Load the trained model
best_model = joblib.load("/Users/darshdave/Documents/Projects/TitanicML-Predicting-Survival/artifacts/best_model.pkl")

# Load the feature names
feature_names = np.load("/Users/darshdave/Documents/Projects/TitanicML-Predicting-Survival/artifacts/feature_names.npy", allow_pickle=True)

app = Flask(__name__)

# Define home route to render HTML interface
@app.route('/')
def home():
    return render_template('home.html')

# Define route to handle form submission and return prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Extract input data from form
    input_data = [request.form.get(feature) for feature in feature_names]

    # Preprocess categorical features
    input_data[1] = 0 if input_data[1] == 'male' else 1

    # Convert data to DataFrame
    data = pd.DataFrame([input_data], columns=feature_names)

    # Make prediction
    prediction = best_model.predict(data)

    # Return prediction
    return render_template('home.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)