import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logger(log_path: str, log_level: str = "DEBUG") -> logging.Logger:
    """
    Configure and return a reusable logger.

    Args:
        log_path (str): Path to the log file.
        log_level (str): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').

    Returns:
        logging.Logger: Configured logger instance.

    Raises:
        ValueError: If log_path is not provided or empty.
    """

    if not log_path:
        raise ValueError("'log_path' is required to initialize the logger.")

    # Create the log directory if it doesn't exist
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Create a logger instance
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level.upper(), logging.DEBUG))
    logger.propagate = False  # Prevent duplicate logs if multiple handlers exist

    # Define a readable log format
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler (rotates when file reaches 5MB, keeps 3 backups)
    file_handler = RotatingFileHandler(
        log_path, maxBytes=5 * 1024 * 1024, backupCount=10, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Avoid adding multiple handlers if already initialized
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    logger.debug("Logger initialized successfully.")
    return logger
