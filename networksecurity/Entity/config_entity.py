'''
Config classes for each stage of the training pipeline.
Nothing gets trained here, these just hold the paths and settings
each component needs (where to read from, where to write to).
'''

from datetime import datetime
import os

from networksecurity.constants import training_pipeline

print(training_pipeline.PIPELINE_NAME)   # NetworkSecurity
print(training_pipeline.ARTIFACT_DIR)    # Artifacts

class TrainingPipelineConfig:
    '''Main config for a run. Every run gets its own timestamped folder under Artifacts/.'''

    def __init__(self, timestamp: datetime = None):
        # default to None, otherwise datetime.now() gets fixed at import time
        timestamp = timestamp or datetime.now()
        timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")

        self.pipeline_name: str = training_pipeline.PIPELINE_NAME
        self.artifact_name: str = training_pipeline.ARTIFACT_DIR
        self.artifact_dir: str = os.path.join(self.artifact_name, timestamp)
        self.model_dir: str = os.path.join("final_model")  # final model used for prediction
        self.timestamp: str = timestamp


class DataIngestionConfig:
    '''Paths for pulling data from MongoDB and splitting it into train/test.'''

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_ingestion_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, training_pipeline.DATA_INGESTION_DIR_NAME
        )

        # full raw export from mongo
        self.feature_store_file_path: str = os.path.join(
            self.data_ingestion_dir, training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR, training_pipeline.FILE_NAME
        )

        self.training_file_path: str = os.path.join(
            self.data_ingestion_dir, training_pipeline.DATA_INGESTION_INGESTED_DIR, training_pipeline.TRAIN_FILE_NAME
        )
        self.testing_file_path: str = os.path.join(
            self.data_ingestion_dir, training_pipeline.DATA_INGESTION_INGESTED_DIR, training_pipeline.TEST_FILE_NAME
        )

        self.train_test_split_ratio: float = training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATION
        self.collection_name: str = training_pipeline.DATA_INGESTION_COLLECTION_NAME
        self.database_name: str = training_pipeline.DATA_INGESTION_DATABASE_NAME


class DataValidationConfig:
    '''Paths for schema/drift checks. Data goes to valid or invalid folder based on the result.'''

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_validation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, training_pipeline.DATA_VALIDATION_DIR_NAME
        )
        self.valid_data_dir: str = os.path.join(self.data_validation_dir, training_pipeline.DATA_VALIDATION_VALID_DIR)
        self.invalid_data_dir: str = os.path.join(self.data_validation_dir, training_pipeline.DATA_VALIDATION_INVALID_DIR)

        self.valid_train_file_path: str = os.path.join(self.valid_data_dir, training_pipeline.TRAIN_FILE_NAME)
        self.valid_test_file_path: str = os.path.join(self.valid_data_dir, training_pipeline.TEST_FILE_NAME)
        self.invalid_train_file_path: str = os.path.join(self.invalid_data_dir, training_pipeline.TRAIN_FILE_NAME)
        self.invalid_test_file_path: str = os.path.join(self.invalid_data_dir, training_pipeline.TEST_FILE_NAME)

        self.drift_report_file_path: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME,
        )


class DataTransformationConfig:
    '''Paths for the transformed numpy arrays and the saved preprocessor.'''

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_transformation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, training_pipeline.DATA_TRANSFORMATION_DIR_NAME
        )

        # same file names as ingestion, just saved as .npy
        self.transformed_train_file_path: str = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            training_pipeline.TRAIN_FILE_NAME.replace("csv", "npy"),
        )
        self.transformed_test_file_path: str = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            training_pipeline.TEST_FILE_NAME.replace("csv", "npy"),
        )

        # fitted preprocessor, reused later at prediction time
        self.transformed_object_file_path: str = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
            training_pipeline.PREPROCESSING_OBJECT_FILE_NAME,
        )


class ModelTrainerConfig:
    '''Where the trained model is saved, plus the score thresholds it has to meet.'''

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.model_trainer_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, training_pipeline.MODEL_TRAINER_DIR_NAME
        )
        self.trained_model_file_path: str = os.path.join(
            self.model_trainer_dir, training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR, training_pipeline.MODEL_FILE_NAME
        )

        self.expected_accuracy: float = training_pipeline.MODEL_TRAINER_EXPECTED_SCORE  # reject model below this
        # max allowed gap between train and test score
        self.overfitting_underfitting_threshold: float = training_pipeline.MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD