'''
Helper functions shared across pipeline components.
Right now: reading and writing YAML files (schema, drift report).
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