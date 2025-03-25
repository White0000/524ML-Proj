import logging
import os

logger = logging.getLogger("Utils")
logger.setLevel(logging.INFO)

def create_directory_if_not_exists(path: str):
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def log_message(content: str):
    logger.info(content)
