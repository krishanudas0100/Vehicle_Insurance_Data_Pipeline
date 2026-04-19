import sys
import logging

def error_message_detail(error: Exception, error_detail: sys) -> str:
    _, _, exc_tb = error_detail.exc_info()

    # Add this null check
    if exc_tb is None:
        return f"Error: {str(error)}"

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = f"Error occurred in python script: [{file_name}] at line number [{line_number}]: {str(error)}"
    
    logging.error(error_message)
    return error_message


class MyException(Exception):
    def __init__(self, error_message: str, error_detail: sys):
        super().__init__(error_message)

        # ✅ Pass 'error_message' correctly (was already fine, but ensure sys is passed)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self) -> str:
        return self.error_message