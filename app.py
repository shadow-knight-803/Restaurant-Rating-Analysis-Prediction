from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import os
import sys
from dataclasses import dataclass

from src.pipelines.prediction_pipeline import CustomData, PredictionPipeline
from src.logger import logging
from src.exception import CustomException

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    try:
        return render_template('form.html')
    except Exception as e:
        logging.error(f"Exception occurred in home route: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = CustomData(
            longitude=float(request.form.get('longitude')),
            latitude=float(request.form.get('latitude')),
            country_code=int(request.form.get('country_code')),
            city=request.form.get('city'),
            cuisines=request.form.get('cuisines'),
            average_cost_for_two=float(request.form.get('average_cost_for_two')),
            currency=request.form.get('currency'),
            has_table_booking=request.form.get('has_table_booking'),
            has_online_delivery=request.form.get('has_online_delivery'),
            is_delivering_now=request.form.get('is_delivering_now'),
            price_range=int(request.form.get('price_range')),
            votes=int(request.form.get('votes')),
            rating_text=request.form.get('rating_text')
        )
        
        # Convert to DataFrame
        df = data.get_data_as_dataframe()
        logging.info("DataFrame created for prediction")
        
        # Initialize prediction pipeline
        pred_pipeline = PredictionPipeline()
        logging.info("Prediction Pipeline initialized")
        
        # Make prediction
        results = pred_pipeline.predict(df)
        logging.info("Prediction completed")
        
        # Return results
        return render_template('form.html', 
                              results=f"Predicted Restaurant Rating: {results[0]:.2f}", 
                              data=request.form)
        
    except Exception as e:
        logging.error(f"Exception occurred in predict route: {e}")
        return render_template('form.html', 
                              error=f"Error occurred: {e}", 
                              data=request.form)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)