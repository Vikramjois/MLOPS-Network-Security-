'''
Artifact classes: the output "receipt" of each pipeline stage.
Each one holds file paths (and metrics) that the next stage reads.
'''

from dataclasses import dataclass  # @dataclass decorator turns a class into a data container, auto-creates __init__ from the listed fields
                                   #creates variable for an empty class


@dataclass
class DataIngestionArtifact:
    '''Output of ingestion, input to validation.'''
    trained_file_path: str
    test_file_path: str


@dataclass
class DataValidationArtifact:
    '''Output of validation, input to transformation.'''
    validation_status: bool  # True if checks passed and no drift
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str  # None for now, not written yet
    invalid_test_file_path: str
    drift_report_file_path: str


@dataclass
class DataTransformationArtifact:
    '''Output of transformation, input to model trainer.'''
    transformed_object_file_path: str  # preprocessing.pkl
    transformed_train_file_path: str  # train.npy
    transformed_test_file_path: str  # test.npy


@dataclass
class ClassificationMetricArtifact:
    '''
    Scores for one dataset (train or test). Used twice inside ModelTrainerArtifact.
    Comparing train vs test scores shows overfitting.
    '''
    f1_score: float  # balance of precision and recall
    precision_score: float  # of rows predicted as class 1, how many really were 1
    recall_score: float  # of actual class 1 rows, how many the model caught


@dataclass
class ModelTrainerArtifact:
    '''Output of model trainer: saved model plus its train and test scores.'''
    trained_model_file_path: str  # model.pkl
    train_metric_artifact: ClassificationMetricArtifact
    test_metric_artifact: ClassificationMetricArtifact