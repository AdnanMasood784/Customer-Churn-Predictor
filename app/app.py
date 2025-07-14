from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load model and feature names
base_path = os.path.dirname(__file__)
model = joblib.load(os.path.join(base_path, 'churn_model.pkl'))
input_columns = joblib.load(os.path.join(base_path, 'input_columns.pkl'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Capture user input
    gender = request.form['gender']
    senior = request.form['senior']
    partner = request.form['partner']
    tenure = float(request.form['tenure'])
    monthly = float(request.form['monthly'])
    total = float(request.form['total'])

    # Build raw dict to simulate user input in training format
    input_dict = {
        'gender_Male': 1 if gender == 'Male' else 0,
        'SeniorCitizen': 1 if senior == 'Yes' else 0,
        'Partner_Yes': 1 if partner == 'Yes' else 0,
        'tenure': tenure,
        'MonthlyCharges': monthly,
        'TotalCharges': total
    }

    # Create a DataFrame with all expected columns
    df = pd.DataFrame([input_dict])
    df = df.reindex(columns=input_columns, fill_value=0)  # Ensure all missing cols are 0

    # Predict
    result = model.predict(df)[0]

    return render_template('index.html', result="Churn" if result == 1 else "Not Churn")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
