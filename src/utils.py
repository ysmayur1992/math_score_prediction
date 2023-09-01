import os
import sys
from src.logg import logging
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score

def save_object(file_path,obj):
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path,exist_ok=True)

    with open(file_path,"wb") as file_obj:
        dill.dump(obj,file_obj)


def Evaluate_Models(model_dict:dict,X_train,X_test,y_train,y_test):
    score_list = dict()

    for name,model in model_dict.items():
        logging.info("Evaluating the model: {0}".format(name))
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        score = r2_score(y_pred,y_test)
        score_list.update({name:score})

    return score_list


def load_object(file_path):
    
    with open(file_path,"rb") as file_obj:
        return dill.load(file_obj)