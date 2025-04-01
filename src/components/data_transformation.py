import os
import sys
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    """
    Configuration class for data transformation
    """
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        """
        This function creates the data transformation pipeline
        """
        try:
            logging.info('Data Transformation initiated')
            
            # Define numerical and categorical columns
            numerical_cols = ['Longitude', 'Latitude', 'Average Cost for two', 'Votes']
            categorical_cols = ['Country Code', 'City', 'Cuisines', 'Currency', 'Has Table booking', 
                                'Has Online delivery', 'Is delivering now', 'Price range', 'Rating text']
            
            logging.info('Pipeline Initiated')
            
            # Numerical Pipeline
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            # Categorical Pipeline - Set handle_unknown='ignore' to handle new categories
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
                ('scaler', StandardScaler())
            ])
            
            # Combine Preprocessing Steps
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_cols),
                ('cat_pipeline', cat_pipeline, categorical_cols)
            ])
            
            logging.info('Pipeline Completed')
            
            return preprocessor
        
        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e, sys)
    
    def preprocess_dataframe(self, df):
        """
        This function applies the initial preprocessing steps on the dataframe
        """
        try:
            logging.info("Started preprocessing dataframe")
            
            # Drop unnecessary columns if they exist
            if 'Restaurant ID' in df.columns:
                df = df.drop(labels=['Restaurant ID'], axis=1)
            
            if 'Restaurant Name' in df.columns:
                df = df.drop(labels=['Restaurant Name'], axis=1)
                
            logging.info("Dropped Restaurant ID and Restaurant Name columns if present")
            
            # Drop rows with missing Cuisines
            df = df.dropna(subset=["Cuisines"])
            logging.info("Dropped rows with missing Cuisines")
            
            # Drop duplicate rows
            df = df.drop_duplicates()
            logging.info("Dropped duplicate rows")
            
            # Map binary categorical features
            has_table_booking_map = {'Yes': 1, 'No': 0}
            has_online_delivery_map = {'Yes': 1, 'No': 0}
            is_delivering_now_map = {'Yes': 1, 'No': 0}
            
            # Map rating colors and text
            rating_color_map = {'Dark Green': 5, 'Green': 4, 'Yellow': 3, 'Orange': 2, 'Red': 1}
            rating_text_map = {'Excellent': 5, 'Very Good': 4, 'Good': 3, 'Average': 2, 'Poor': 1}
            
            # Apply mappings
            df['Has Online delivery'] = df['Has Online delivery'].map(has_online_delivery_map)
            df['Has Table booking'] = df['Has Table booking'].map(has_table_booking_map)
            df['Is delivering now'] = df['Is delivering now'].map(is_delivering_now_map)
            df['Rating color'] = df['Rating color'].map(rating_color_map)
            logging.info("Applied mappings to binary features and rating features")
            
            # Handle 'Not rated' in Rating text
            mode_value = df[df["Rating text"] != "Not rated"]["Rating text"].mode()[0]
            df["Rating text"] = df["Rating text"].replace("Not rated", mode_value)
            df['Rating text'] = df['Rating text'].map(rating_text_map)
            logging.info("Handled 'Not rated' in Rating text")
            
            # Fill missing Rating color with mode
            if df["Rating color"].isnull().sum() > 0:
                df["Rating color"] = df["Rating color"].fillna(df["Rating color"].mode()[0])
                logging.info("Filled missing Rating color with mode")
            
            # Process Cuisines
            cuisine_counts = df["Cuisines"].value_counts()
            df["Cuisines"] = df["Cuisines"].map(cuisine_counts)
            logging.info("Processed Cuisines column")
            
            # Process City
            city_counts = df["City"].value_counts()
            df["City"] = df["City"].map(city_counts)
            logging.info("Processed City column")
            
            # Drop redundant address-related columns if they exist
            columns_to_drop = []
            for col in ["Address", "Locality", "Locality Verbose"]:
                if col in df.columns:
                    columns_to_drop.append(col)
            
            if columns_to_drop:
                df.drop(columns=columns_to_drop, inplace=True)
                logging.info(f"Dropped columns: {columns_to_drop}")
            
            # Process Currency using LabelEncoder
            if 'Currency' in df.columns:
                label_encoder = LabelEncoder()
                df["Currency"] = label_encoder.fit_transform(df["Currency"])
                logging.info("Applied Label Encoding to Currency column")
            
            # Drop Switch to order menu column if it exists
            if "Switch to order menu" in df.columns:
                df.drop(columns=["Switch to order menu"], inplace=True)
                logging.info("Dropped Switch to order menu column")
            
            logging.info("Completed preprocessing dataframe")
            return df
        
        except Exception as e:
            logging.info(f"Exception occurred during preprocessing dataframe: {str(e)}")
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        """
        This function initiates the data transformation process
        """
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Shape: {train_df.shape}')
            logging.info(f'Test Dataframe Shape: {test_df.shape}')
            
            # Preprocess train and test dataframes
            logging.info('Starting dataframe preprocessing')
            train_df = self.preprocess_dataframe(train_df)
            test_df = self.preprocess_dataframe(test_df)
            logging.info('Completed dataframe preprocessing')
            
            logging.info('Obtaining preprocessing object')
            # Get preprocessor object
            preprocessing_obj = self.get_data_transformer_object()
            
            # Define target column
            target_column_name = 'Aggregate rating'
            
            # Create feature and target datasets for training
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            # Create feature and target datasets for testing
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            # Log column names for debugging
            logging.info(f'Train columns: {input_feature_train_df.columns.tolist()}')
            logging.info(f'Test columns: {input_feature_test_df.columns.tolist()}')
            
            logging.info('Applying preprocessing object on training and testing datasets')
            try:
                # Transform features
                input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
                input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
                logging.info('Successfully transformed features')
            except Exception as e:
                logging.error(f'Error in transformation: {str(e)}')
                # Try preprocessing the data again with extra safety checks
                for col in input_feature_test_df.columns:
                    if col in input_feature_train_df.columns:
                        test_values = set(input_feature_test_df[col].unique())
                        train_values = set(input_feature_train_df[col].unique())
                        diff = test_values - train_values
                        if diff and len(diff) > 0:
                            logging.warning(f"Column {col} has {len(diff)} values in test not in train: {diff}")
                raise CustomException(e, sys)
            
            # Convert to numpy arrays
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]
            
            # Save preprocessing object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            logging.info('Preprocessor pickle file saved')
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            logging.error(f"Exception occurred in the initiate_data_transformation: {str(e)}")
            raise CustomException(e, sys)

