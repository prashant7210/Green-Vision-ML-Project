import sys
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.forest.constants import *
from src.forest.exception import ForestExpection
from src.forest.logger import logging
from src.forest.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file
from src.forest.constants.training_pipeline import TARGET_COLUMN, SCHEMA_FILE_PATH
from src.forest.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact
from src.forest.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_transformation_config: DataTransformationConfig):
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_transformation_config = data_transformation_config

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise ForestExpection(e, sys)
        
    def get_data_transformer_object(self) -> object:
        logging.info("Got numerical and categorical columns from schema config")
        try:
            _schema_config = read_yaml_file(file_path = SCHEMA_FILE_PATH)
            num_features = _schema_config['numerical_columns']
            cat_features = _schema_config['categorical_columns']

            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            # preprocessor = ColumnTransformer(['Numeric_Pipeline', numerical_pipeline, num_features])  # Error while Debugging
            preprocessor = ColumnTransformer(
                transformers=[
                    ("Numeric_Pipeline", numerical_pipeline, num_features)
                ],
                remainder="passthrough"   # optional: keeps categorical columns if you want
            )
            logging.info("Created numerical pipeline and preprocessor object")
            return preprocessor
        except Exception as e:
            raise ForestExpection(e, sys) from e
        
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Method Name: initiate_data_validation
        Description: This method initiates the data validation components of training pipeline
        Output: Returns te boolean status of data validation
        On Failure: Write an exception log and then raise an exception
        Revisions: moved setup to cloud
        """
        logging.info("Initiating data transformation")
        try:
            logging.info("Data transformation started")
            preprocessor = self.get_data_transformer_object()
            logging.info("Got the preprocessor object")

            train_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.training_file_path)
            test_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.testing_file_path)

            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            logging.info("Got the input and target features for training data")

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            logging.info("Got the input and target features for testing data")

            logging.info("Applying the preprocessor object on training and testing data")
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            logging.info("Applied the preprocessor object on training and testing data")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)

            logging.info("Saved the preprocessor object and transformed data into respective file paths")
            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path = self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path = self.data_transformation_config.transformed_test_file_path,
                transformed_object_file_path = self.data_transformation_config.transformed_object_file_path
            )
            return data_transformation_artifact
        
        except Exception as e:
            raise ForestExpection(e, sys) from e