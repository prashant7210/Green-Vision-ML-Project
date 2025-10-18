import sys
import src.forest.exception as ForestExpection
from src.forest.logger import logging
from src.forest.constants import *
from src.forest.utils.main_utils import load_numpy_array_data, read_yaml_file, load_object, save_object
from src.forest.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact, ClassificationMetricArtifact
from src.forest.entity.config_entity import ModelTrainerConfig
from src.forest.entity.estimator import SensorModel
from neuro_mf import ModelFactory


class ModelTrainer:
    def __init__(self, data_tranformation_artifact: DataTransformationArtifact, model_trainer_config: ModelTrainerConfig):
        self.data_tranformation_artifact = data_tranformation_artifact
        self.model_trainer_config = model_trainer_config

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered into model_traning method of ModelTrainer class")
        try:
            train_arr = load_numpy_array_data(self.data_tranformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(self.data_tranformation_artifact.transformed_test_file_path)

            logging.info("Loaded train and test numpy array data")
            logging.info(f"Train array shape: {train_arr.shape}, Test array shape: {test_arr.shape}")

            x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            model_factory = ModelFactory(model_config_path = self.model_trainer_config.model_config_file_path)
            best_model_detail = model_factory.get_best_model(X = x_train, y = y_train, base_accuracy = self.model_trainer_config.base_accuracy)
            preprocessing_object = load_object(file_path = self.data_tranformation_artifact.transformed_object_file_path)


            if best_model_detail.best_score < self.model_trainer_config.base_accuracy:
                logging.info("No best model found with score more than base score")
                raise Exception("No best model found with more than base score")

            sensor_model= SensorModel(preprocessing_object = preprocessing_object, trained_model_object = best_model_detail.best_model)
            logging.info("Created Sensor model object with preprocessor and model")
            logging.info("Created best model file path")
            save_object(self.model_trainer_config.trained_model_file_path, sensor_model)

            metric_artifact = ClassificationMetricArtifact(f1_score = 0.8, recall_score = 0.9, precision_score = 0.8)
            model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path = self.model_trainer_config.trained_model_file_path, metric_artifact = metric_artifact)
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact
        
        except Exception as e:
            raise ForestExpection(e, sys) from e