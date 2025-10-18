import sys
from src.forest.exception import ForestExpection
from src.forest.logger import logging
from src.forest.components.data_ingestion import DataIngestion
from src.forest.components.data_validation import DataValidation
from src.forest.components.data_transformation import DataTransformation
from src.forest.components.model_trainer import ModelTrainer
from src.forest.components.model_evaluation import ModelEvaluation
from src.forest.components.model_pusher import ModelPusher
from src.forest.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig, ModelEvaluationConfig, ModelPusherConfig
from src.forest.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact, ModelTrainerArtifact, ModelEvaluationArtifact, ModelPusherArtifact


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        self.model_pusher_config = ModelPusherConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        logging.info("Entered the start_data_ingestion_method of TrainPipeline class")
        try:
            logging.info("Getting the data from MongoDB")
            data_ingestion = DataIngestion(data_ingestion_config = self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Got the train and test set from MongoDB")
            return data_ingestion_artifact
        
        except Exception as e:
            raise ForestExpection(e, sys) from e
        
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        logging.info("Entered into the start_data_validation method of TrainPipeline class")
        try:
            data_validation = DataValidation(data_validation_config = self.data_validation_config, data_ingestion_artifact = data_ingestion_artifact)
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info("performed the data validation")
            return data_validation_artifact

        except Exception as e:
            raise ForestExpection(e, sys) from e
        
    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataTransformationArtifact:
        logging.info("Entered into the start_data_transformation method of TrainPipeline class")
        try:
            logging.info("Data Transformation started...")
            data_transformation = DataTransformation(data_transformation_config = self.data_transformation_config, data_ingestion_artifact = data_ingestion_artifact)
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info("Data Transformed")
            return data_transformation_artifact

        except Exception as e:
            raise ForestExpection(e, sys) from e
    
    def start_model_training(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        logging.info("Entered into the start_model_training method of TrainPipeline class")
        try:
            logging.info("Model Training Started...")
            model_trainer = ModelTrainer(model_trainer_config = self.model_trainer_config, data_tranformation_artifact = data_transformation_artifact)
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info("Model Training Successful")
            return model_trainer_artifact
        
        except Exception as e:
            raise ForestExpection(e, sys) from e
    
    def start_model_evaluation(self, data_ingestion_artifact: DataIngestionArtifact, model_trainer_artifact: ModelTrainerArtifact) -> ModelEvaluationArtifact:
        logging.info("Entered into the start_model_evaluation method of TrainPipeline class")
        try:
            logging.info("Model Evaluation Started...")
            model_evaluation = ModelEvaluation(model_evaluation_config = self.model_evaluation_config, data_ingestion_artifact = data_ingestion_artifact, model_trainer_artifact = model_trainer_artifact)
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
            logging.info("Model Evaluation Completed")
            return model_evaluation_artifact
        
        except Exception as e:
            raise ForestExpection(e, sys)
        

    def start_model_pushing(self, model_trainer_artifact: ModelTrainerArtifact) -> ModelPusherArtifact:
        logging.info("Entered into the start_model_pushing method of TrainPipeline class")
        try:
            logging.info("Model Pushing Started...")
            model_pusher = ModelPusher(model_pusher_config = self.model_pusher_config, model_trainer_artifact = model_trainer_artifact)
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            logging.info("Model Pushed")
            return model_pusher_artifact
        
        except Exception as e:
            raise ForestExpection(e, sys) from e
        

    def run_pipeline(self) -> None:
        logging.info("Eneted into the run_pipeline method of TrainPipeline class")
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact = data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact = data_ingestion_artifact)
            model_trainer_artifact = self.start_model_training(data_transformation_artifact = data_transformation_artifact)
            model_evaluation_artifact = self.start_model_evaluation(data_ingestion_artifact = data_ingestion_artifact, model_trainer_artifact = model_trainer_artifact)

            if not model_evaluation_artifact.is_model_accepted:
                logging.info("Model not accepted")
                return None
            
            model_pusher_artifact = self.start_model_pushing(model_trainer_artifact = model_evaluation_artifact)

            logging.info("Exited the run_pipeline method of TrainPipeline class")

        except Exception as e:
            raise ForestExpection(e, sys) from e