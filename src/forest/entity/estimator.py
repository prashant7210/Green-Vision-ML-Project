import sys
import pandas as pd
from sklearn.pipeline import Pipeline
from src.forest.exception import ForestExpection
from src.forest.logger import logging
from dataclasses import dataclass

class TargetValueMapping:
    def __init__(self):
        self.neg: int = 0
        self.pos: int = 1
    
    def _asdict(self):
        return self.__dict__
    
    def reverse_mapping(self):
        mapping_response = self._asdict()
        return dict(zip(mapping_response.values(), mapping_response.keys()))
    

class SensorModel:
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Entered prediction method into Sensor Model")
        try:
            logging.info("Using the trained model to get predictions")
            transformed_feature = self.preprocessing_object.transform(df)
            logging.info("Used the trained model to get predictions")
            return self.trained_model_object.predict(transformed_feature)
        
        except Exception as e:
            raise ForestExpection(e, sys) from e
        
    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"
    
    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"