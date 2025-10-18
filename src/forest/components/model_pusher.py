import sys
from src.forest.exception import ForestExpection
from src.forest.logger import logging
from src.forest.cloud_storage.aws_storage import SimpleStorageService
from src.forest.entity.artifact_entity import ModelTrainerArtifact, ModelPusherArtifact
from src.forest.entity.config_entity import ModelPusherConfig
from src.forest.entity.s3_estimator import SensorEstimator

class ModelPusher:
    def __init__(self, model_trainer_artifact: ModelTrainerArtifact, model_pusher_config: ModelPusherConfig):
        self.model_trainer_artifact = model_trainer_artifact
        self.model_pusher_config = model_pusher_config
        self.s3 = SimpleStorageService
        self.sensor_estimator = SensorEstimator(bucket_name = model_pusher_config.bucket_name, model_path = model_pusher_config.s3_model_key_path)

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        logging.info("Initiating model pusher method inside ModelPusher class")
        try:
            logging.info("Uploading artifact folder to s3 bucket")
            self.sensor_estimator.save_model(from_file = self.model_trainer_artifact.trained_model_file_path)

            model_pusher_artifact = ModelPusherArtifact(
                bucket_name = self.model_pusher_config.bucket_name,
                s3_model_path = self.model_pusher_config.s3_model_key_path
            )
            logging.info("Uploaded artifacts to s3 buckets")
            logging.info(f"Model pusher artifact: {model_pusher_artifact}")
            return model_pusher_artifact
        
        except Exception as e:
            raise ForestExpection(e, sys) from e