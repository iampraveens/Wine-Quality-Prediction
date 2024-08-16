from WineQuality.config.configuration import ConfigurationManager
from WineQuality.components.data_transformation import DataTransformation
from WineQuality import logger
from pathlib import Path



STAGE_NAME = "Data Transformation stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        """
        Initializes an instance of the DataTransformationTrainingPipeline class.
        
        This is a special method that Python calls when an object is instantiated from the class.
        """
        pass

    def main(self):
        """
        Initiates the data transformation pipeline by checking the data validation status.
        
        If the data validation status is True, it retrieves the data transformation configuration, 
        creates a DataTransformation instance, and performs the train-test splitting.
        
        If the data validation status is False, it raises an exception indicating that the data schema is not valid.
        """
        try:
            with open(Path("artifacts/data_validation/status.txt"), "r") as f:
                status = f.read().split(" ")[-1]

            if status == "True":
                config = ConfigurationManager()
                data_transformation_config = config.get_data_transformation_config()
                data_transformation = DataTransformation(config=data_transformation_config)
                data_transformation.train_test_spliting()

            else:
                raise Exception("You data schema is not valid")

        except Exception as e:
            print(e)