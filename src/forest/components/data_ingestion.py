import sys, os
import shutil
import pandas as pd
from zipfile import ZipFile
from sklearn.model_selection import train_test_split
from src.forest.entity.config_entity import DataIngestionConfig
from src.forest.entity.artifact_entity import DataIngestionArtifact
from src.forest.exception import ForestExpection
from src.forest.logger import logging
from src.forest.utils.main_utils import read_yaml_file, create_directories
from src.forest.constants.training_pipeline import SCHEMA_FILE_PATH
from src.forest.data_access.forest_data import ForestData


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        try:
           self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise ForestExpection(e, sys)

    def export_data_into_feature_store(self) -> pd.DataFrame:
        try:
           logging.info("Exporting data from MogoDB into feature store")
           sensor_data = ForestData()
           df = sensor_data.export_collection_as_dataframe(collection_name=self.data_ingestion_config.collection_name)

           logging.info(f"Shape of data frame: {df.shape}")
           feature_store_file_path = self.data_ingestion_config.feature_store_file_path
           dir_path = os.path.dirname(feature_store_file_path)
           os.makedirs(dir_path, exist_ok=True)
           logging.info(f"Saving data into feature store file path: {feature_store_file_path}")
           df.to_csv(feature_store_file_path, index=False, header=True)
           return df
        
        except Exception as e:
            raise ForestExpection(e, sys) 
        
    def split_data_as_train_test(self, df: pd.DataFrame) -> None:
        logging.info("Splitting data into train and test")
        try:
            train_set, test_set = train_test_split(df, test_size=self.data_ingestion_config.train_test_split_ratio)
            logging.info("Train Test split completed")
            
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)
            logging.info("Exported train and test data into respective file paths")
        except Exception as e:
            raise ForestExpection(e, sys)
        
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Method Name: initiate_data_ingestion
        Description: This method initiates the data ingestion components of training pipeline
        Output: train set and test set are returned as the artifacts of data ingestion components
        On Failure: Write an exception log and then raise an exception
        Revisions :
        moved setup to cloud
        """
        logging.info("Data ingestion initiated")
        try:
            df = self.export_data_into_feature_store()
            _schema_config = read_yaml_file(file_path = SCHEMA_FILE_PATH)
            df = df.drop(columns=_schema_config['drop_columns'], axis=1)

            logging.info("Data exported into feature store successfully")
            self.split_data_as_train_test(df=df)
            logging.info("Exited from initiate_data_ingestion method of DataIngestion class")

            data_ingestion_artifact = DataIngestionArtifact(training_file_path=self.data_ingestion_config.training_file_path, testing_file_path=self.data_ingestion_config.testing_file_path)
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact

        except Exception as e:
            raise ForestExpection(e, sys)   
        