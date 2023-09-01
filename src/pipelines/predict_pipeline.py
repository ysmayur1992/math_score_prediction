import os
import sys
import pandas as pd
from src.logg import logging
from src.exception import CustomException
from src.utils import load_object

class Prediction_Pipeline:
    def __init__(self) -> None:
        pass

    def prediction(self,df):
        try:
            preprocessor_obj = load_object(os.path.join('artifacts',"preprocessed_object.pkl"))
            model_obj = load_object(os.path.join('artifacts',"regression_model.pkl"))
            scaled_values = preprocessor_obj.transform(df)
            y_pred = model_obj.predict(scaled_values)
            return y_pred[0]
            
        except Exception as e:
            raise CustomException(e,sys)



class Custom_Object:
    def __init__(self,
                 gender:str,
                 ethnicity:str,
                 education:str,
                 lunch:str,
                 courses:str,
                 reading:int,
                 writing:int
                ) -> None:
        self.gender = gender
        self.ethnicity = ethnicity
        self.education = education
        self.lunch = lunch
        self.courses = courses
        self.reading = reading
        self.writing = writing

    def get_DataFrame(self):
        try:
            feature_dict = {
                "gender":[self.gender],
                "race_ethnicity":[self.ethnicity],
                "parental_level_of_education":[self.education],
                "lunch":[self.lunch],
                "test_preparation_course":[self.courses],
                "reading_score":[self.reading],
                "writing_score":[self.writing]
            }

            return pd.DataFrame(feature_dict)
        except Exception as e:
            raise CustomException(e,sys)
    