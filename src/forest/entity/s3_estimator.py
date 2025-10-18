import sys
from src.forest.exception import ForestExpection
from src.forest.logger import logging
from src.forest.cloud_storage.aws_storage import SimpleStorageService
from src.forest.entity.estimator import SensorModel
import pandas as pd

class SensorEstimator:
    """
    This class is used to save ans SensorModel in s3 bucket  and to do prediction
    """
    def __init__(self, bucket_name, model_path):
        """
        bucket_name: Name of your model bucket
        model_path: Location of your model in bucket
        """
        self.bucket_name = bucket_name
        self.model_path = model_path
        self.s3 = SimpleStorageService()
        self.loaded_model: SensorModel = None

    def is_model_present(self, model_path):
        try:
            return self.s3.s3_key_path_available(bucket_name=self.bucket_name, s3_key=model_path)
        except ForestExpection as e:
            print(e)
            return False

    def load_model(self) -> SensorModel:
        """
        Load the model from model path
        """
        return self.s3.load_model(model_name = self.model_path, bucket_name = self.bucket_name)


    def save_model(self, from_file, remove: bool = False) -> None:
        """
        Save the model to the model path
        from_file: Your local system model path
        remove: By deafault it is false that mean you will have your model locally in your system folder
        """
        try:
            self.s3.upload_file(from_file, to_filename = self.model_path, bucket_name = self.bucket_name)

        except Exception as e:
            raise ForestExpection(e, sys)

    def predict(self, df: pd.DataFrame):
        try:
            if self.loaded_model is None:
                self.loaded_model = self.load_model()
                logging.info(f"Model loaded successfully for prediction: {self.loaded_model}")
            return self.loaded_model.predict(df)
        
        except Exception as e:
            raise ForestExpection(e, sys)