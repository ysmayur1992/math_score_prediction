import os
import sys
import pandas as pd
from src.logg import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.componants.data_transformation import Data_Transformation_Config
from src.componants.data_transformation import Data_Transformation
from src.componants.model_training import Model_Trainer

@dataclass
class data_ingestion_config:
    raw_data_path:str = os.path.join('artifacts',"dataFrame.csv")
    train_data_path:str = os.path.join('artifacts',"train.csv")
    test_data_path:str = os.path.join('artifacts',"test.csv")


class Data_Ingestion:
    def __init__(self) -> None:
        self.config = data_ingestion_config()

    def get_data_ingestion(self):
        try:
            logging.info("data ingestion just started")

            df = pd.read_csv('data\stud.csv')
            os.makedirs(os.path.dirname(self.config.raw_data_path),exist_ok=True)
            logging.info("directory made now converting to csv")
            df.to_csv(self.config.raw_data_path,header=True,index=False)
            logging.info("splitting with train_test_split")
            train,test = train_test_split(df,test_size=0.25,random_state=42)
            logging.info("converting train and test to csv")
            train.to_csv(self.config.train_data_path,header=True,index=False)
            test.to_csv(self.config.test_data_path,header=True,index=False)
            logging.info("Ingestion done now returning the paths")

            return (
                self.config.train_data_path,
                self.config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        

if __name__=="__main__":
    ingest = Data_Ingestion()
    train_path,test_path = ingest.get_data_ingestion()

    transformation = Data_Transformation()
    train_set,test_set,_ = transformation.Initiate_Transformation(train_path,test_path)

    model_obj = Model_Trainer()
    _,name,score = model_obj.Initiate_Model_Training(train_set,test_set)
    message = "The best model is {0} and its score is {1}".format(name,score)
    print(message)