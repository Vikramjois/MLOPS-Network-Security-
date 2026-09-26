import sys
from networksecurity.logging import logger

class NetworkSecurityException(Exception):
    def __init__(self,error_message,error_details:sys):
        self.error_message = error_message #save raw error
        _,_,exc_tb = error_details.exc_info() #dig out traceback
        
        self.lineno=exc_tb.tb_lineno #save line number
        self.file_name=exc_tb.tb_frame.f_code.co_filename #save file name
    
    def __str__(self):
        return "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        self.file_name, self.lineno, str(self.error_message)) #return formatted string using the 3 saved fields
                                                              #controls what print(exception) shows
        
if __name__=='__main__':
    try:
        logger.logging.info("Enter the try block") #is calling the info severity function from logger python file
        a=1/0
        print("This will not be printed",a)
    except Exception as e:
           raise NetworkSecurityException(e,sys)