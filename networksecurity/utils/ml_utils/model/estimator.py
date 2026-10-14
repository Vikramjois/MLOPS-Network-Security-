'''
NetworkModel: wraps the fitted preprocessor and the trained model into one object.
Saved as model.pkl, so prediction only needs one file and one predict() call.
'''

from networksecurity.constants.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME  # not used yet, for saving the final model later

import os
import sys

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging


class NetworkModel:
    '''Holds preprocessor + model so new data is always transformed the same way before predicting.'''

    def __init__(self, preprocessor, model):
        try:
            self.preprocessor = preprocessor  # fitted pipeline (KNN imputer), from preprocessing.pkl
            self.model = model  # best trained model from the model trainer
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def predict(self, x):
        '''Transform raw input with the saved preprocessor, then predict (0 = phishing, 1 = legitimate).'''
        try:
            x_transform = self.preprocessor.transform(x)  # transform only, never fit again
            y_hat = self.model.predict(x_transform)
            return y_hat
        except Exception as e:
            raise NetworkSecurityException(e, sys)