"""
Centralized logging system for Knowledge Graph project.

This module provides a unified logging configuration with:
- File logging with rotation
- Console logging with colors
- Structured log format
- Per-module loggers
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }

    def format(self, record):
        """Format log record with colors."""
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"

        # Format the message
        result = super().format(record)

        # Reset levelname for subsequent formatters
        record.levelname = levelname

        return result


class LoggerManager:
    """
    Manager for creating and configuring loggers.

    Features:
    - Singleton pattern to ensure consistent configuration
    - Rotating file handlers to prevent log files from growing too large
    - Separate logs for different modules
    - Console and file output
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the logger manager (only once)."""
        if not LoggerManager._initialized:
            self.logs_dir = Path("logs")
            self.logs_dir.mkdir(exist_ok=True)
            self.loggers = {}
            LoggerManager._initialized = True

    def get_logger(
        self,
        name: str,
        level: str = "INFO",
        log_to_file: bool = True,
        log_to_console: bool = True
    ) -> logging.Logger:
        """
        Get or create a logger with specified configuration.

        Args:
            name: Logger name (typically module name)
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_to_file: Whether to log to file
            log_to_console: Whether to log to console

        Returns:
            Configured logger instance

        Example:
            >>> logger = LoggerManager().get_logger("graph_builder", "DEBUG")
            >>> logger.info("Starting graph construction")
        """
        # Return existing logger if already configured
        if name in self.loggers:
            return self.loggers[name]

        # Create new logger
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        logger.propagate = False

        # Remove existing handlers to avoid duplicates
        logger.handlers.clear()

        # Create formatters
        file_formatter = logging.Formatter(
            fmt='%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        console_formatter = ColoredFormatter(
            fmt='%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )

        # Add file handler if requested
        if log_to_file:
            # Create main log file
            log_file = self.logs_dir / f"{name}.log"
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

            # Create daily log file
            today = datetime.now().strftime("%Y-%m-%d")
            daily_log_file = self.logs_dir / f"{name}_{today}.log"
            daily_handler = RotatingFileHandler(
                daily_log_file,
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=3,
                encoding='utf-8'
            )
            daily_handler.setLevel(logging.DEBUG)
            daily_handler.setFormatter(file_formatter)
            logger.addHandler(daily_handler)

        # Add console handler if requested
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, level.upper()))
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        # Store logger
        self.loggers[name] = logger

        return logger

    def set_level(self, name: str, level: str):
        """
        Change logging level for an existing logger.

        Args:
            name: Logger name
            level: New logging level
        """
        if name in self.loggers:
            self.loggers[name].setLevel(getattr(logging, level.upper()))
            for handler in self.loggers[name].handlers:
                if isinstance(handler, logging.StreamHandler):
                    handler.setLevel(getattr(logging, level.upper()))

    def close_all(self):
        """Close all logger handlers."""
        for logger in self.loggers.values():
            for handler in logger.handlers:
                handler.close()
                logger.removeHandler(handler)


def get_logger(
    name: str,
    level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True
) -> logging.Logger:
    """
    Convenience function to get a logger.

    Args:
        name: Logger name (typically __name__)
        level: Logging level
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console

    Returns:
        Configured logger instance

    Example:
        >>> from fact_check_kg.core.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
    """
    manager = LoggerManager()
    return manager.get_logger(name, level, log_to_file, log_to_console)


def log_section(logger: logging.Logger, title: str, width: int = 60):
    """
    Log a formatted section header.

    Args:
        logger: Logger instance
        title: Section title
        width: Width of the separator line

    Example:
        >>> log_section(logger, "STEP 1: Data Loading")
    """
    logger.info("=" * width)
    logger.info(title.center(width))
    logger.info("=" * width)


def log_stats(logger: logging.Logger, stats: dict, title: Optional[str] = None):
    """
    Log statistics in a formatted way.

    Args:
        logger: Logger instance
        stats: Dictionary of statistics
        title: Optional title for the stats block

    Example:
        >>> log_stats(logger, {"nodes": 1000, "edges": 5000}, "Graph Statistics")
    """
    if title:
        logger.info(f"\n{title}:")

    for key, value in stats.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")
