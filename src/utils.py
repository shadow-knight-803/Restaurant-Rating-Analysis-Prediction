import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def save_object(file_path, obj):
    """
    Function to save an object using pickle
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            
        logging.info(f"Object successfully saved at {file_path}")
            
    except Exception as e:
        logging.error(f"Exception occurred while saving object: {str(e)}")
        raise CustomException(e, sys)

def evaluate_model(X_train, y_train, X_test, y_test, models):
    """
    Function to evaluate multiple models and return performance report
    """
    try:
        logging.info("Started model evaluation")
        report = {}
        
        for i in range(len(models)):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            
            # Train model
            logging.info(f"Training model: {model_name}")
            model.fit(X_train, y_train)
            
            # Predict on test data
            logging.info(f"Predicting with model: {model_name}")
            y_test_pred = model.predict(X_test)
            
            # Calculate metrics
            test_r2_score = r2_score(y_test, y_test_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            
            # Store results
            report[model_name] = {
                'r2_score': test_r2_score,
                'mae': test_mae,
                'rmse': test_rmse
            }
            
            logging.info(f"Model {model_name} evaluation - R2: {test_r2_score}, MAE: {test_mae}, RMSE: {test_rmse}")
            
        return report
        
    except Exception as e:
        logging.error(f"Exception occurred during model training: {str(e)}")
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Function to load a pickle object
    """
    try:
        logging.info(f"Loading object from {file_path}")
        with open(file_path, 'rb') as file_obj:
            obj = pickle.load(file_obj)
        logging.info(f"Object successfully loaded from {file_path}")
        return obj
        
    except Exception as e:
        logging.error(f"Exception occurred while loading object: {str(e)}")
        raise CustomException(e, sys)

def calculate_metrics(y_true, y_pred):
    """
    Calculate and return common regression metrics
    """
    try:
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        return {
            'r2_score': r2,
            'mae': mae,
            'rmse': rmse
        }
        
    except Exception as e:
        logging.error(f"Exception in metrics calculation: {str(e)}")
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Load a pickle file
    Args:
        file_path (str): Path to the pickle file
    Returns:
        object: Loaded pickle object
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)