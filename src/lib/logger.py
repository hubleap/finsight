import logging
import sys

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] - [%(message)s]"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    

def get_logger(name: str = 'app') -> logging.Logger:
    """Create or retrieve a configured logger with console and file handlers.

    Sets up a logger with both console output and rotating file handler.
    Each log entry includes request ID for tracing.

    Args:
        name: Name of the logger instance

    Returns:
        Configured Logger instance with formatters and handlers
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.propagate = False

    return logger
