import logging
import os
from datetime import datetime

LOGS_DIR = "logs"
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)
LOG_FILE = os.path.join(LOGS_DIR, f"{datetime.now().strftime('%Y-%m-%d')}.log")

logging.basicConfig(level=logging.INFO, filename=LOG_FILE, filemode='a', format='%(asctime)s - %(levelname)s - %(message)s' )


def get_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    return logger