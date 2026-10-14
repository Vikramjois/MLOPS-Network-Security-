'''
Helper functions shared across pipeline components:
YAML read/write, numpy save/load, pickle save/load, and model evaluation.
'''

import yaml
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
import os, sys
import numpy as np
# import dill
import pickle

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def read_yaml_file(file_path: str) -> dict:
    '''Read a YAML file (e.g. schema.yaml) and return its contents as a dict.'''
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)  # safe_load won't run any code hidden in the file
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    '''
    Write content (e.g. the drift report dict) to a YAML file.
    If replace is True, any existing file is deleted first.
    '''
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # make sure the folder exists
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def save_numpy_array_data(file_path: str, array: np.array):
    '''Save a numpy array to a .npy file (e.g. transformed train/test data).'''
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:  # "wb" = write binary
            np.save(file_obj, array)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


def save_object(file_path: str, obj: object) -> None:
    '''Save any Python object with pickle (e.g. preprocessor, trained model).'''
    try:
        logging.info("Entered the save_object method of MainUtils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


def load_object(file_path: str) -> object:
    '''Load a pickled object (e.g. preprocessing.pkl, model.pkl). Fails if the file is missing.'''
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:  # "rb" = read binary
            print(file_obj)  # debug print, safe to remove
            return pickle.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.array:
    '''Load a numpy array from a .npy file.'''
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    '''
    Tune each model with GridSearchCV, retrain with the best params,
    and return {model_name: test_score} so the trainer can pick the best one.
    models: dict of name -> model, param: dict of name -> param grid
    '''
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]  # param grid for this model

            gs = GridSearchCV(model, para, cv=3)  # try every combination, 3-fold CV
            gs.fit(X_train, y_train)

            # retrain the original model with the best params found
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # r2_score is a regression metric; for classification, accuracy or f1 is a better fit
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise NetworkSecurityException(e, sys)