from src.logg import logging
import sys

def get_error_message(error,error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    frame = exc_tb.tb_frame.f_code.co_filename
    message = "The Error occured in file: [{0}] at line no: {1} and the error is: {2}".format(frame,exc_tb.tb_lineno,str(error))
    return message

class CustomException(Exception):
    def __init__(self, error,error_detail:sys) -> None:
        super().__init__(error)
        self.error_message = get_error_message(error,error_detail=error_detail)

    def __str__(self) -> str:
        return super().__str__()
    
