import pandas as pd 
import os
import joblib
from sklearn.linear_model import ElasticNet
from WineQuality import logger
from WineQuality.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        """
        Initializes the ModelTrainer class with a ModelTrainerConfig object.
        
        Args:
            config (ModelTrainerConfig): The configuration object for the model trainer.
        """
        self.config = config
        
    def train(self):
        """
        Trains a linear regression model using the ElasticNet algorithm and saves the trained model to a file.
        This function reads the training and test data from the specified paths in the configuration object. It then 
        extracts the features and target variable from the data and splits the features into training and test sets.
        The ElasticNet algorithm is used to train the model on the training data, and the trained model is saved 
        to a file using the joblib library.
        Parameters:
            self (ModelTrainer): The instance of the ModelTrainer class.
        """
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)
        
        X_train = train_data.drop([self.config.target_column], axis=1)
        X_test = test_data.drop([self.config.target_column], axis=1)
        y_train = train_data[[self.config.target_column]]
        y_test = train_data[[self.config.target_column]]
        
        lr = ElasticNet(alpha=self.config.alpha, l1_ratio=self.config.l1_ratio, random_state=42)
        lr.fit(X_train, y_train)
        
        joblib.dump(lr, os.path.join(self.config.root_dir, self.config.model_name))

