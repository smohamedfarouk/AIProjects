import sys

class CustomException(Exception):
    def __init__(self, message, errors=None):
        super().__init__(message)
        self.errors = errors
        _, _, tb = sys.exc_info()
        self.line_number = tb.tb_lineno if tb else None


    @staticmethod
    def get_detailed_error_message():
        _, _, tb = sys.exc_info()
        filename = tb.tb_frame.f_code.co_filename if tb else None
        line_number = tb.tb_lineno if tb else None
        if tb:
            return f"Error occurred in file {filename} at line {line_number }"
        return "No detailed error information available"        


    def __str__(self):
        return self.errors if self.errors else super().__str__()
