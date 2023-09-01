import os
import sys
from src.logg import logging
from src.exception import CustomException
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from dataclasses import dataclass
from src.utils import save_object,Evaluate_Models

@dataclass
class model_training_config:
    model_config = os.path.join('artifacts',"regression_model.pkl")

class Model_Trainer:
    def __init__(self) -> None:
        self.config = model_training_config()

    def Initiate_Model_Training(self,df_train,df_test):
        try:
            logging.info("Model training started now splitting into X_train,X_test,y_train,y_test")

            X_train,X_test,y_train,y_test = (
                df_train[:,:-1],
                df_test[:,:-1],
                df_train[:,-1],
                df_test[:,-1]
            )

            model_dict = {
                "Linear Regression":LinearRegression(),
                "Decision Tree":DecisionTreeRegressor(),
                "SVR":SVR(),
                "XGBoost":XGBRegressor(),
                "CatBoost":CatBoostRegressor(),
                "KNN":KNeighborsRegressor(),
                "AdaBoost":AdaBoostRegressor(),
                "Gradient Boost":GradientBoostingRegressor(),
                "Random Forest":RandomForestRegressor()
            }
            logging.info("created dictionary of models now evaluating")
            scores_dict = Evaluate_Models(model_dict,X_train,X_test,y_train,y_test)
            logging.info("now sorting the best model up")
            sorted_score_dict = dict(sorted(scores_dict.items(),key=lambda x:x[1], reverse=True))

            best_model_name = list(sorted_score_dict.keys())[0]
            best_score = list(sorted_score_dict.values())[0]

            if best_score < 0.6:
                raise CustomException("No Best Model Found")
            
            logging.info("now saving and returning the score")
            
            save_object(
                obj = model_dict[best_model_name],
                file_path=self.config.model_config
            )

            return (
                self.config.model_config,
                best_model_name,
                best_score
            )
        
        except Exception as e:
            raise CustomException(e,sys)