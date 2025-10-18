import os
import sys

def error_msg_details(error, error_details):
    try:
        _, _, exc_tb = error_details.exc_info()
        if exc_tb is None: return f"Error Occured: {str(error)}"
        
        file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        error_message = f"Error Occure Python Script: [{file_name}, Line Number: [{exc_tb.tb_lineno}], Error Message: [{str(error)}]"
        return error_message
    
    except Exception as e:
        return f"Error Occured: {str(e)}. Additionally error occured in expection handling: {str(e)}"
    

class ForestExpection(Exception):
    def __init__(self, error_msg, error_details=sys):
        super().__init__(error_msg)

        self.error_msg = error_msg_details(error_msg, error_details)

    def __str__(self):
        return self.error_msg
