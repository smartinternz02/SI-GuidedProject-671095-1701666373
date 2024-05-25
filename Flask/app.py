# -*- coding: utf-8 -*-
"""
Created on Wed May 15 22:00:54 2024

@author: nares
"""

from flask import Flask, render_template, request
import pandas as pd
from keras.models import load_model
import pickle
import joblib

app = Flask(__name__)

# Load the saved ANN model, scaler, and column transformer
model = load_model('model1.h5')
scaler = pickle.load(open('scale1.pkl', 'rb'))
ct = joblib.load('column')
# Function to preprocess new data
def preprocess_input(data):
    transformed_data = ct.transform(data)
    scaled_data = scaler.transform(transformed_data.toarray())
    return scaled_data
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/form')
def form():
    return render_template('form.html')
@app.route('/predict', methods=['POST'])
def predict():
    # Collect data from form
    form_data = request.form.to_dict()
    data = pd.DataFrame([form_data.values()], columns=form_data.keys())
    # Preprocess the input data
    preprocessed_data = preprocess_input(data)
    # Make predictions using the ANN model
    prediction = model.predict(preprocessed_data)
    predicted_price = prediction[0][0]
    return render_template('predict.html', prediction_text=f'The predicted price of the diamond is ${predicted_price:.2f}')

if __name__ == '__main__':
    app.run(debug=True)
