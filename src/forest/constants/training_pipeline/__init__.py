import os
from src.forest.constants.s3_bucket import TRAINING_BUCKET_NAME

# Pipeline name and other root constant
TARGET_COLUMN = 'Cover_Type'
PIPELINE_NAME: str = 'covtype'
ARTIFACT_DIR: str = 'artifact'

# Common file name
FILE_NAME: str = 'covtype.csv'
TRAIN_FILE_NAME: str = 'train.csv'
TEST_FILE_NAME: str = 'test.csv'
PREPROCESSING_OBJECT_FILE_NAME = 'preprocessing.pkl'
MODEL_FILE_NAME = 'model.pkl'
SCHEMA_FILE_PATH = os.path.join('config', 'schema.yaml')


"""
Data Ingestion related constants starts with `DATA_INGESTION` VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str = 'forest'
DATA_INGESTION_DIR_NAME: str = 'data_ingestion'
DATA_INGESTION_FEATURE_STORE_DIR: str = 'feature_store'
DATA_INGESTION_INGESTED_DIR: str = 'ingested'
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2


"""
Data Validation related constants starts with `DATA_VALIDATION` VAR NAME
"""
DATA_VALIDATION_DIR_NAME: str = 'data_validation'
DATA_VALIDATION_VALID_DIR: str = 'validated'        
DATA_VALIDATION_INVALID_DIR: str = 'invalid'
DATA_VALIDATION_DRIFT_REPORT_DIR: str = 'drift_report'
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = 'report.yaml'

"""
Data Transformation related constants starts with `DATA_TRANSFORMATION` VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME: str = 'data_transformation'
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR_NAME: str = 'transformed_data'
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = 'transformed_object'

"""
Model Trainer related constants starts with `MODEL_TRAINER` VAR NAME
"""
MODEL_TRAINER_DIR_NAME: str = 'model_trainer'
MODEL_TRAINER_TRAINED_MODEL_DIR: str = 'trained_model'
MODEL_FILE_NAME: str = 'model.pkl'
MODEL_TRAINER_BASE_ACCURACY: float = 0.6
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")

"""
Model Evaluation related constants starts with `MODEL_EVALUATION` VAR NAME
"""
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02


"""
Model Pusher related constants starts with `MODEL_PUSHER` VAR NAME
"""
MODEL_PUSHER_BUCKET_NAME = TRAINING_BUCKET_NAME
MODEL_PUSHER_S3_KEY = "model-registry"