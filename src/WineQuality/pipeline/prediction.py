import joblib 
import numpy as np 
import pandas as pd 
from pathlib import Path


class PredictionPipeline:
    def __init__(self):
        """
        Initializes the PredictionPipeline class by loading the trained model from a file.
        """
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))
        
    def predict(self, data):
        """
        Makes a prediction based on the input data using the trained model.
        
        Parameters:
            data (array-like): The input data to make predictions on.
        
        Returns:
            prediction (array-like): The predicted values.
        """
        prediction = self.model.predict(data)
        
        return prediction