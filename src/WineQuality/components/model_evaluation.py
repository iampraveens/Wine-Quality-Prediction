import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from WineQuality.utils.common import save_json
from urllib.parse import urlparse
import numpy as np
import joblib
from WineQuality.entity.config_entity import ModelEvaluationConfig
from pathlib import Path

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        """
        This is the constructor method for the ModelEvaluation class.
        
        It initializes the object with a ModelEvaluationConfig object.
        
        Parameters:
            config (ModelEvaluationConfig): The configuration for model evaluation.
        """
        self.config = config

    def eval_metrics(self,actual, pred):
        """
        Evaluates the performance of a model by calculating the root mean squared error (RMSE), 
        mean absolute error (MAE), and R-squared (R2) score.
        Parameters:
            actual (array-like): The actual values of the target variable.
            pred (array-like): The predicted values of the target variable.
        Returns:
            tuple: A tuple containing the RMSE, MAE, and R2 score of the model.
        """
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2
    
    def save_results(self):
        """
    	Saves the results of model evaluation by calculating the performance metrics 
    	such as root mean squared error (RMSE), mean absolute error (MAE), and R-squared (R2) score.
    	
    	Parameters:
    		No parameters are required as the function uses the configuration provided 
    		during the instantiation of the ModelEvaluation class.
    	"""
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        X_test = test_data.drop([self.config.target_column], axis=1)
        y_test = test_data[[self.config.target_column]]
        
        predicted = model.predict(X_test)

        (rmse, mae, r2) = self.eval_metrics(y_test, predicted)
        
        # Saving metrics as local
        scores = {"rmse": rmse, "mae": mae, "r2": r2}
        save_json(path=Path(self.config.metric_file_name), data=scores)