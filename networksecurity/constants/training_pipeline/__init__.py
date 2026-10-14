'''
All constants for the training pipeline in one place.
Change a name or setting here and every stage picks it up.
'''

import os
import sys
import numpy as np
import pandas as pd

"""
Common constants used across the pipeline
"""
TARGET_COLUMN = "Result"  # label column (-1 phishing, 1 legitimate)
PIPELINE_NAME: str = "NetworkSecurity"
ARTIFACT_DIR: str = "Artifacts"  # root folder, each run gets a timestamped subfolder
FILE_NAME: str = "phisingData.csv"  # raw data file saved in the feature store

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")  # expected columns, used in validation

SAVED_MODEL_DIR = os.path.join("saved_models")  # final model kept outside Artifacts
MODEL_FILE_NAME = "model.pkl"

"""
Data Ingestion constants (start with DATA_INGESTION)
"""
DATA_INGESTION_COLLECTION_NAME: str = "NetworkData"  # MongoDB collection
DATA_INGESTION_DATABASE_NAME: str = "VIKRAMAI"  # MongoDB database
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"  # raw copy of the data
DATA_INGESTION_INGESTED_DIR: str = "ingested"  # train/test split files
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.2  # 20% test

"""
Data Validation constants (start with DATA_VALIDATION)
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"  # data that passed checks
DATA_VALIDATION_INVALID_DIR: str = "invalid"  # data that failed checks
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"  # KS test p-value per column

"""
Data Transformation constants (start with DATA_TRANSFORMATION)
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"  # train.npy, test.npy
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"  # preprocessing.pkl
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"  # fitted preprocessor, reused at prediction

# KNN imputer settings: fill each NaN using the 3 most similar rows, all weighted equally
DATA_TRANSFORMATION_IMPUTER_PARAMS: dict = {
    "missing_values": np.nan,
    "n_neighbors": 3,
    "weights": "uniform",
}

DATA_TRANSFORMATION_TRAIN_FILE_PATH: str = "train.npy"
DATA_TRANSFORMATION_TEST_FILE_PATH: str = "test.npy"

"""
Model Trainer constants (start with MODEL_TRAINER)
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6  # minimum score, below this the model is rejected
MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD: float = 0.05  # max allowed train vs test score gap

TRAINING_BUCKET_NAME = "netwworksecurity"  # cloud (S3) bucket for syncing artifacts later