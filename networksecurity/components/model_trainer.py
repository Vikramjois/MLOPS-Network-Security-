'''
Model trainer component.
Loads transformed arrays, tunes and compares several classifiers,
logs scores to MLflow (DagsHub), and saves the best one (with the preprocessor) as model.pkl.
'''

import os
import sys

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.Entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from networksecurity.Entity.config_entity import ModelTrainerConfig

from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import save_object, load_object
from networksecurity.utils.main_utils.utils import load_numpy_array_data, evaluate_models
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
import mlflow  # experiment tracking: logs metrics and models for each run
from urllib.parse import urlparse  # used to check if tracking is remote (https) or local (file)

from dotenv import load_dotenv
load_dotenv()  # reads the MLflow/DagsHub settings below from .env

# MLflow reads these 3 variables automatically, so no browser login is needed

# os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI")
# os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
# os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

# only set MLflow variables if they exist (they're missing inside Docker unless passed with -e)
for key in ["MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"]:
    if os.getenv(key):
        os.environ[key] = os.getenv(key)

# not needed when the 3 variables above are set, and it triggers the browser login
# import dagshub
# dagshub.init(repo_owner='Vikramjois', repo_name='MLOPS-Network-Security-', mlflow=True)


class ModelTrainer:
    '''Trains candidate models, picks the best, and saves it as a NetworkModel.'''

    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config  # model path + thresholds
            self.data_transformation_artifact = data_transformation_artifact  # paths to npy files + preprocessor
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def track_mlflow(self, best_model, classificationmetric):
        '''Log F1, precision, recall and the model to MLflow (DagsHub) as one run.'''
        mlflow.set_registry_uri(os.getenv("MLFLOW_TRACKING_URI"))  # register models on the same DagsHub server
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme  # "https" for DagsHub, "file" for local

        with mlflow.start_run():
            mlflow.log_metric("f1_score", classificationmetric.f1_score)
            mlflow.log_metric("precision", classificationmetric.precision_score)
            mlflow.log_metric("recall_score", classificationmetric.recall_score)

            if tracking_url_type_store != "file":
                # remote server (DagsHub): log the model and register it under a name
                mlflow.sklearn.log_model(
                    best_model,
                    name="model",
                    registered_model_name="NetworkSecurityModel",  # must be a string, not the model object
                    serialization_format="cloudpickle",  # avoids the skops untrusted-type error
                )
            else:
                # local file store: model registry not supported, just log the model
                mlflow.sklearn.log_model(best_model, name="model", serialization_format="cloudpickle")

    def train_model(self, X_train, y_train, x_test, y_test):
        '''Compare models, pick the best by test score, log metrics, save model, return artifact.'''

        # candidate models (the "model factory"), verbose=1 prints training progress
        models = {
            "Random Forest": RandomForestClassifier(verbose=1),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(verbose=1),
            "Logistic Regression": LogisticRegression(verbose=1),
            "AdaBoost": AdaBoostClassifier(),
        }

        # hyperparameter grids for GridSearchCV, commented ones skipped to save time
        params = {
            "Decision Tree": {
                'criterion': ['gini', 'entropy', 'log_loss'],
                # 'splitter':['best','random'],
                # 'max_features':['sqrt','log2'],
            },
            "Random Forest": {
                # 'criterion':['gini', 'entropy', 'log_loss'],
                # 'max_features':['sqrt','log2',None],
                'n_estimators': [8, 16, 32, 128, 256]
            },
            "Gradient Boosting": {
                # 'loss':['log_loss', 'exponential'],
                'learning_rate': [.1, .01, .05, .001],
                'subsample': [0.6, 0.7, 0.75, 0.85, 0.9],
                # 'criterion':['squared_error', 'friedman_mse'],
                # 'max_features':['auto','sqrt','log2'],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            "Logistic Regression": {},  # no tuning, default settings
            "AdaBoost": {
                'learning_rate': [.1, .01, .001],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            }
        }

        # tune + train each model, returns {model_name: test_score}
        model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=x_test, y_test=y_test,
                                             models=models, param=params)

        # highest test score and the model name that got it
        best_model_score = max(sorted(model_report.values()))
        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]
        best_model = models[best_model_name]  # already fitted with best params inside evaluate_models

        # train metrics + log to mlflow (first run)
        y_train_pred = best_model.predict(X_train)
        classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
        self.track_mlflow(best_model, classification_train_metric)

        # test metrics + log to mlflow (second run)
        y_test_pred = best_model.predict(x_test)
        classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)
        self.track_mlflow(best_model, classification_test_metric)

        # load the fitted preprocessor from the transformation stage
        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok=True)

        # wrap preprocessor + model into one object and save in this run's Artifacts folder
        Network_Model = NetworkModel(preprocessor=preprocessor, model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path, obj=Network_Model)  # save the instance, not the class

        # model pusher: copy of the bare model used by app.py for prediction
        save_object("final_model/model.pkl", best_model)

        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=classification_train_metric,
            test_metric_artifact=classification_test_metric
        )
        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        '''Load train/test arrays, split into X and y, and run training.'''
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            # last column is the target, everything before it is features
            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            model_trainer_artifact = self.train_model(x_train, y_train, x_test, y_test)
            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)