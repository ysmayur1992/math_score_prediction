import os
import sys
import pandas as pd
import numpy as np
from src.logg import logging
from src.exception import CustomException
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from src.utils import save_object

@dataclass
class Data_Transformation_Config:
    data_preprocessor = os.path.join('artifacts',"preprocessed_object.pkl")

class Data_Transformation:
    def __init__(self) -> None:
        self.data_trans_config = Data_Transformation_Config()

    def get_data_transformation(self):
        try:
            numerical = ['reading_score','writing_score']
            categorical = ['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']

            num_pipeline = Pipeline(
                steps=[
                    ('impute',SimpleImputer(strategy='median')),
                    ('scaling',StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('impute',SimpleImputer(strategy='most_frequent')),
                    ('encoding',OneHotEncoder()),
                    ('scaling',StandardScaler(with_mean=False))
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ('Numerical Features',num_pipeline,numerical),
                    ('Categorical Features',cat_pipeline,categorical)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        

    def Initiate_Transformation(self,train_path,test_path):
        try:
            logging.info("Data Transformation is started")
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)

            target = 'math_score'
            indep_train_features = df_train.drop(target,axis=1)
            indep_test_features = df_test.drop(target,axis=1)

            object = self.get_data_transformation()
            logging.info("preprocessor object is created now conducting preprocessing of data")
            indep_train_features_arr = object.fit_transform(indep_train_features)
            indep_test_features_arr = object.transform(indep_test_features)

            train_set = np.c_[indep_train_features_arr,np.array(df_train[target])]
            test_set = np.c_[indep_test_features_arr,np.array(df_test[target])]
            logging.info("concatenation done now saving and returning")
            save_object(
                obj = object,
                file_path = self.data_trans_config.data_preprocessor
            )

            return (
                train_set,
                test_set,
                self.data_trans_config.data_preprocessor
            )

        except Exception as e:
            raise CustomException(e,sys)