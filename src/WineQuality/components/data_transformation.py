import os
import pandas as pd
from WineQuality import logger
from sklearn.model_selection import train_test_split
from WineQuality.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        """
        Initializes the DataTransformation class with a DataTransformationConfig object.
        
        Args:
            config (DataTransformationConfig): The configuration object for data transformation.
        """
        self.config = config
        
    def train_test_spliting(self):
        """
        Splits the data into training and test sets.
        This function reads the data from the specified data path and drops the 'Id' column.
        It then splits the data into training and test sets using the `train_test_split` function
        from the `sklearn.model_selection` module. The split is done with a ratio of 0.75 for the
        training set and 0.25 for the test set.
        The training and test sets are saved as CSV files in the specified root directory.
        Parameters:
            self (DataTransformation): The instance of the `DataTransformation` class.
        """
        data = pd.read_csv(self.config.data_path)
        data = data.drop(columns="Id", axis=1)

        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(data)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)
        