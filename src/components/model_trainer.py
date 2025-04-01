# Basic Import
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_model

from dataclasses import dataclass
import sys
import os

@dataclass
class ModelTrainerConfig:
    """
    Configuration class for Model Trainer
    """
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Splitting dependent and independent variables from train and test data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
                
            logging.info("Setting up models for training")
            models = {
                'LinearRegression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'ElasticNet': ElasticNet(),
                'DecisionTree': DecisionTreeRegressor(),
                'RandomForest': RandomForestRegressor()
            }
                
            logging.info("Starting model evaluation")
            model_report = evaluate_model(X_train, y_train, X_test, y_test, models)
                
            # Print the model report
            print("\n====================================================================================\n")
            print("Model Performance Report:")
            for model_name, metrics in model_report.items():
                print(f"{model_name}: R2 Score = {metrics}")
            print("\n====================================================================================\n")
                
            logging.info(f"Model Report: {model_report}")
                
            # Get the best model based on r2_score
            best_model_score = -float("inf")  # Initialize with negative infinity
            best_model_name = None
                
            for model_name, metrics in model_report.items():
                r2_score = metrics['r2_score']  # Extract the r2_score from metrics dictionary
                if r2_score > best_model_score:
                    best_model_score = r2_score
                    best_model_name = model_name
                
            best_model = models[best_model_name]
                
            print(f"Best Model Found, Model Name: {best_model_name}, R2 Score: {best_model_score}")
            print("\n====================================================================================\n")
                
            logging.info(f"Best Model Found, Model Name: {best_model_name}, R2 Score: {best_model_score}")
                
            # Save the trained model
            logging.info(f"Saving the best model at {self.model_trainer_config.trained_model_file_path}")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
                
            return best_model_score
            
        except Exception as e:
            logging.error("Exception occurred at Model Training")
            raise CustomException(e, sys)