'''
Data validation component.
Checks the train/test files from ingestion (column count, drift)
and returns a DataValidationArtifact for the transformation stage.
'''

from networksecurity.Entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from networksecurity.Entity.config_entity import DataValidationConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.constants.training_pipeline import SCHEMA_FILE_PATH  # path to schema.yaml
from scipy.stats import ks_2samp  # KS test, compares two distributions
import pandas as pd
import os, sys
from networksecurity.utils.main_utils.utils import read_yaml_file, write_yaml_file


class DataValidation:
    '''Validates ingested data against the schema and checks for drift.'''

    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact  # where train/test csv are
            self.data_validation_config = data_validation_config  # where to save outputs
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)  # expected columns
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        '''Read a csv into a dataframe. Static since it doesn't need self.'''
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        '''Return True if the dataframe has the same number of columns as the schema.'''
        try:
            # number_of_columns = len(self._schema_config)
            number_of_columns = len(self._schema_config["columns"])
            logging.info(f"Required number of columns:{number_of_columns}")
            logging.info(f"Data frame has columns:{len(dataframe.columns)}")
            if len(dataframe.columns) == number_of_columns:
                return True
            return False
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def detect_dataset_drift(self, base_df, current_df, threshold=0.05) -> bool:
        '''
        Run a KS test on each column (train vs test) and save a drift report.
        Returns True if no column drifted, False otherwise.
        '''
        try:
            status = True  # assume no drift until a column says otherwise
            report = {}
            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                is_same_dist = ks_2samp(d1, d2)
                # p-value >= threshold means distributions look the same
                if threshold <= is_same_dist.pvalue:
                    is_found = False
                else:
                    is_found = True
                    status = False  # one drifted column is enough to flag it
                report.update({column: {
                    "p_value": float(is_same_dist.pvalue),
                    "drift_status": is_found
                }})

            drift_report_file_path = self.data_validation_config.drift_report_file_path
            os.makedirs(os.path.dirname(drift_report_file_path), exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path, content=report)
            return status  # was missing, without it status comes back as None
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        '''Run all checks in order and return the validation artifact.'''
        try:
            # paths come from the ingestion artifact
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)

            # column count check (only builds a message for now, doesn't stop the run)
            status = self.validate_number_of_columns(dataframe=train_dataframe)
            if not status:
                error_message = f"Train dataframe does not contain all columns.\n"
            status = self.validate_number_of_columns(dataframe=test_dataframe)
            if not status:
                error_message = f"Test dataframe does not contain all columns.\n"

            # drift check, train is the reference
            status = self.detect_dataset_drift(base_df=train_dataframe, current_df=test_dataframe)

            # save validated copies
            os.makedirs(os.path.dirname(self.data_validation_config.valid_train_file_path), exist_ok=True)
            train_dataframe.to_csv(self.data_validation_config.valid_train_file_path, index=False, header=True)
            test_dataframe.to_csv(self.data_validation_config.valid_test_file_path, index=False, header=True)

            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                # valid_train_file_path=self.data_ingestion_artifact.trained_file_path,
                # valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )
            return data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)