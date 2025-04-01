import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

@dataclass
class PredictionPipelineConfig:
    """
    Configuration class for Prediction Pipeline
    """
    model_path: str = os.path.join("artifacts", "model.pkl")
    preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl")

class PredictionPipeline:
    def __init__(self):
        self.prediction_config = PredictionPipelineConfig()
    
    def predict(self, features):
        """
        Method to make predictions using the trained model
        Args:
            features (pandas.DataFrame): Input features for prediction
        Returns:
            numpy.ndarray: Predictions
        """
        try:
            logging.info("Prediction Pipeline started")
            
            # Load preprocessor and model
            logging.info(f"Loading preprocessor from {self.prediction_config.preprocessor_path}")
            preprocessor = load_object(file_path=self.prediction_config.preprocessor_path)
            
            logging.info(f"Loading model from {self.prediction_config.model_path}")
            model = load_object(file_path=self.prediction_config.model_path)
            
            # Preprocess the data
            logging.info("Preprocessing the input data")
            data_scaled = preprocessor.transform(features)
            
            # Make predictions
            logging.info("Making predictions")
            predictions = model.predict(data_scaled)
            
            logging.info("Prediction completed successfully")
            return predictions
            
        except Exception as e:
            logging.error(f"Exception occurred during prediction: {e}")
            raise CustomException(e, sys)

class CustomData:
    """
    Class to map user input to DataFrame
    """
    def __init__(
        self,
        longitude: float,
        latitude: float,
        country_code: int,
        city: str,
        cuisines: str,
        average_cost_for_two: float,
        currency: str,
        has_table_booking: str,
        has_online_delivery: str,
        is_delivering_now: str,
        price_range: int,
        votes: int,
        rating_text: str
    ):
        self.longitude = longitude
        self.latitude = latitude
        self.country_code = country_code
        self.city = city
        self.cuisines = cuisines
        self.average_cost_for_two = average_cost_for_two
        self.currency = currency
        self.has_table_booking = has_table_booking
        self.has_online_delivery = has_online_delivery
        self.is_delivering_now = is_delivering_now
        self.price_range = price_range
        self.votes = votes
        self.rating_text = rating_text
    
    def get_data_as_dataframe(self):
        """
        Converts user inputs to a pandas DataFrame
        Returns:
            pandas.DataFrame: Input features as DataFrame
        """
        try:
            custom_data_input_dict = {
                'Longitude': [self.longitude],
                'Latitude': [self.latitude],
                'Country Code': [self.country_code],
                'City': [self.city],
                'Cuisines': [self.cuisines],
                'Average Cost for two': [self.average_cost_for_two],
                'Currency': [self.currency],
                'Has Table booking': [self.has_table_booking],
                'Has Online delivery': [self.has_online_delivery],
                'Is delivering now': [self.is_delivering_now],
                'Price range': [self.price_range],
                'Votes': [self.votes],
                'Rating text': [self.rating_text]
            }
            
            # Create DataFrame from dictionary
            df = pd.DataFrame(custom_data_input_dict)
            logging.info("DataFrame created successfully")
            
            # Apply the same preprocessing steps as done in data_transformation.py
            # Note: This is a simplified version. In production, you should use the same preprocessing logic
            # or import functions from data_transformation.py
            
            # Map binary categorical features
            has_table_booking_map = {'Yes': 1, 'No': 0}
            has_online_delivery_map = {'Yes': 1, 'No': 0}
            is_delivering_now_map = {'Yes': 1, 'No': 0}
            
            # Map rating text
            rating_text_map = {'Excellent': 5, 'Very Good': 4, 'Good': 3, 'Average': 2, 'Poor': 1}
            
            # Apply mappings
            df['Has Online delivery'] = df['Has Online delivery'].map(has_online_delivery_map)
            df['Has Table booking'] = df['Has Table booking'].map(has_table_booking_map)
            df['Is delivering now'] = df['Is delivering now'].map(is_delivering_now_map)
            
            # Handle Rating text
            df['Rating text'] = df['Rating text'].map(rating_text_map)
            
            # For Cuisines and City, we would normally use the same mapping as in training
            # Here we'll assume these are already processed values
            
            return df
            
        except Exception as e:
            logging.error(f"Exception occurred in get_data_as_dataframe: {e}")
            raise CustomException(e, sys)


