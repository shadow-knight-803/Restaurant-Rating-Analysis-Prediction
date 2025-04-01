import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException

## Initialize the Data Ingestion Configuration
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw.csv")

## Data Ingestion Class
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """Method to load data, perform train-test split, and save the files"""
        logging.info("Data Ingestion process started.")
        try:
            # Load dataset
            df = pd.read_csv(os.path.join("notebooks/data", "Dataset.csv"))
            logging.info("Dataset successfully loaded into a pandas DataFrame.")

            # Ensure artifact directory exists
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Save raw dataset
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Raw data saved.")

            # Perform Train-Test Split
            train_set, test_set = train_test_split(df, test_size=0.30, random_state=42)
            logging.info("Train-test split completed.")

            # Save train and test data
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Train and test data saved successfully.")

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            logging.error(f"Error occurred during Data Ingestion: {e}")
            raise CustomException(e, sys)
