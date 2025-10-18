import sys
import pandas as pd
from src.forest.exception import ForestExpection
from src.forest.logger import logging
from src.forest.utils.main_utils import read_yaml_file
from src.forest.constants.training_pipeline import SCHEMA_FILE_PATH
from src.forest.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.forest.entity.config_entity import DataValidationConfig


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_validation_config = data_validation_config
        self._schema_config = read_yaml_file(file_path = SCHEMA_FILE_PATH)

    def validate_number_of_columns(self, dataframe:pd.DataFrame) -> bool:
        try:
            status = len(dataframe.columns) == len(self._schema_config['columns'])
            logging.info(f"Number of columns validation status: {status}")
            return status
        except Exception as e:
            raise ForestExpection(e, sys)
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise ForestExpection(e, sys)
            
    def is_numerical_column_exists(self, dataframe: pd.DataFrame) -> bool:
        try:
            dataframe_columns = dataframe.columns
            status = True
            missing_numerical_columns = []
            for col in self._schema_config['numerical_columns']:
                if col not in dataframe_columns:
                    status = False
                missing_numerical_columns.append(col)
                
            if not status: logging.info(f"Missing numerical columns: {missing_numerical_columns}")
            return status
        except Exception as e:
            raise ForestExpection(e, sys) from e
            
    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Method Name: initiate_data_validation
        Description: This method initiates the data validation components of training pipeline
        Output: Returns te boolean status of data validation
        On Failure: Write an exception log and then raise an exception
        Revisions: moved setup to cloud
        """
        logging.info("Entered into initiate_data_validation method of DataValidation class")
        try: 
            validation_error_msg = ""
            train_df = DataValidation.read_data(file_path = self.data_ingestion_artifact.training_file_path)
            test_df = DataValidation.read_data(file_path = self.data_ingestion_artifact.testing_file_path)

            logging.info("Data validation started")

            status = self.validate_number_of_columns(train_df)
            if not status: validation_error_msg += "Columns are missing in training dataframe\n"

            status = self.validate_number_of_columns(test_df)
            if not status: validation_error_msg += "Columns are missing in testing dataframe\n"

            status = self.is_numerical_column_exists(train_df)
            if not status: validation_error_msg += "Numerical columns are missing in training dataframe\n"

            status = self.is_numerical_column_exists(test_df)
            if not status: validation_error_msg += "Numerical columns are missing in testing dataframe\n"

            validation_status = validation_error_msg == ""
            if not validation_status:
                logging.error(f"Data validation failed with errors: {validation_error_msg}")
                data_validation_artifact = DataValidationArtifact(
                    validation_status = False,
                    valid_train_file_path = self.data_ingestion_artifact.training_file_path,
                    valid_test_file_path = self.data_ingestion_artifact.testing_file_path,
                    invalid_train_file_path = self.data_validation_config.invalid_train_file_path,
                    invalid_test_file_path = self.data_validation_config.invalid_test_file_path,
                    drift_report_file_path = self.data_validation_config.drift_report_file_path
                )
                logging.info(f"Data validation artifact: {data_validation_artifact}")
                return data_validation_artifact

        except Exception as e:
            raise ForestExpection(e, sys) 
