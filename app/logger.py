import logging
import sys
from app.config import settings


LOG_FORMAT = "{asctime} - {name} - {levelname} - {message}"

def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, style="{"))
        logger.addHandler(handler)
    

    return logger





logger = setup_logger(settings.logger_name)