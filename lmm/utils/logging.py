"""
Centralized logging configuration for the ML Markdown project.

This module provides a standardized way to configure and use Python's 
logging module across the entire project. It ensures consistent log 
formatting, appropriate log levels, and centralized configuration.

Usage:
    from library.lm_logging import get_logger, ConsoleLogger, 
        FileLogger, ExceptionConsoleLogger

    # Use the abstract interface implementations
    console_logger = ConsoleLogger(__name__)
    file_logger = FileLogger(__name__, "app.log")
    exception_logger = ExceptionConsoleLogger(__name__)

    # Or use the traditional logger
    logger = get_logger(__name__)
"""

import logging
import sys
from pathlib import Path
from abc import ABC, abstractmethod


class ILogger(ABC):
    """
    Abstract interface for logging functionality.
    """

    @abstractmethod
    def set_level(self, level: int) -> None:
        """Set the logging level for the logger."""
        pass

    @abstractmethod
    def info(self, msg: str) -> None:
        """Log an informational message."""
        pass

    @abstractmethod
    def error(self, msg: str) -> None:
        """Log an error message."""
        pass

    @abstractmethod
    def warning(self, msg: str) -> None:
        """Log a warning message."""
        pass

    @abstractmethod
    def critical(self, msg: str) -> None:
        """Log a critical message."""
        pass


class ConsoleLogger(ILogger):
    """
    A console logger implementation that uses logging.Logger as a 
    delegate. Logs messages to the console using Python's built-in 
    logging module.
    """

    def __init__(self, name: str = "") -> None:
        """
        Initialize the ConsoleLogger with a specific logger name.

        Args:
            name: The name of the logger, typically __name__ to use 
            the module name
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # Ensure we have a console handler if none exists
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def set_level(self, level: int) -> None:
        """Set the logging level for the logger."""
        self.logger.setLevel(level)

    def info(self, msg: str) -> None:
        """Log an informational message."""
        self.logger.info(msg)

    def error(self, msg: str) -> None:
        """Log an error message."""
        self.logger.error(msg, exc_info=True)

    def warning(self, msg: str) -> None:
        """Log a warning message."""
        self.logger.warning(msg)

    def critical(self, msg: str) -> None:
        """Log a critical message."""
        self.logger.critical(msg, exc_info=True, stack_info=True)


class FileLogger(ILogger):
    """
    A file logger implementation that uses logging.Logger as a 
    delegate. Logs messages to a specified file using Python's 
    built-in logging module.
    """

    def __init__(
        self, name: str = "", log_file: str | Path = "app.log"
    ) -> None:
        """
        Initialize the FileLogger with a specific logger name and 
        file path.

        Args:
            name: The name of the logger, typically __name__ to use 
                the module name
            log_file: Path to the log file where messages will be 
                written
        """
        self.logger = logging.getLogger(f"{name}_file")
        self.logger.setLevel(logging.INFO)

        # Clear any existing handlers to avoid duplicates
        self.logger.handlers.clear()

        # Add file handler
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # Prevent propagation to avoid duplicate logs
        self.logger.propagate = False

    def set_level(self, level: int) -> None:
        """Set the logging level for the logger."""
        self.logger.setLevel(level)

    def info(self, msg: str) -> None:
        """Log an informational message."""
        self.logger.info(msg)

    def error(self, msg: str) -> None:
        """Log an error message."""
        self.logger.error(msg, exc_info=True)

    def warning(self, msg: str) -> None:
        """Log a warning message."""
        self.logger.warning(msg)

    def critical(self, msg: str) -> None:
        """Log a critical message."""
        self.logger.critical(msg, exc_info=True, stack_info=True)


class FileConsoleLogger(ILogger):
    """
    A file logger implementation that uses logging.Logger as a 
    delegate. Logs messages to a specified file using Python's 
    built-in logging module, and relays the messages to the console 
    as well.
    """

    console_logger: ILogger

    def __init__(
        self, name: str = "", log_file: str | Path = "app.log"
    ) -> None:
        """
        Initialize the FileLogger with a specific logger name and 
        file path.

        Args:
            name: The name of the logger, typically __name__ to use 
                the module name
            log_file: Path to the log file where messages will be 
                written
        """
        self.logger = logging.getLogger(f"{name}_file")
        self.logger.setLevel(logging.INFO)

        # Clear any existing handlers to avoid duplicates
        self.logger.handlers.clear()

        # Add file handler
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # Prevent propagation to avoid duplicate logs
        self.logger.propagate = False

        # Delegate for console
        self.console_logger = ConsoleLogger(name)

    def set_level(self, level: int) -> None:
        """Set the logging level for the logger."""
        self.logger.setLevel(level)
        self.console_logger.set_level(level)

    def info(self, msg: str) -> None:
        """Log an informational message."""
        self.logger.info(msg)
        self.console_logger.info(msg)

    def error(self, msg: str) -> None:
        """Log an error message."""
        self.logger.error(msg, exc_info=True)
        self.console_logger.error(msg)

    def warning(self, msg: str) -> None:
        """Log a warning message."""
        self.logger.warning(msg)
        self.console_logger.warning(msg)

    def critical(self, msg: str) -> None:
        """Log a critical message."""
        self.logger.critical(msg, exc_info=True, stack_info=True)
        self.console_logger.critical(msg)


class ExceptionConsoleLogger(ILogger):
    """
    A console logger implementation that raises exceptions on error 
    and critical calls.

    This logger behaves like ConsoleLogger for info, warning, and 
    set_level methods, but raises exceptions when error() or 
    critical() methods are called.
    The message is still logged before the exception is raised.
    """

    def __init__(self, name: str = "") -> None:
        """
        Initialize the ExceptionConsoleLogger with a specific logger 
        name.

        Args:
            name: The name of the logger, typically __name__ to use 
                the module name
        """
        self.logger = logging.getLogger(f"{name}_exception")
        self.logger.setLevel(logging.INFO)

        # Ensure we have a console handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def set_level(self, level: int) -> None:
        """Set the logging level for the logger."""
        self.logger.setLevel(level)

    def info(self, msg: str) -> None:
        """Log an informational message."""
        self.logger.info(msg)

    def error(self, msg: str) -> None:
        """Log an error message and raise an exception."""
        self.logger.error(msg)
        raise RuntimeError(f"Error: {msg}")

    def warning(self, msg: str) -> None:
        """Log a warning message."""
        self.logger.warning(msg)

    def critical(self, msg: str) -> None:
        """Log a critical message and raise an exception."""
        self.logger.critical(msg)
        raise RuntimeError(f"Critical error: {msg}")


def get_logger(name: str) -> ILogger:
    """
    Get a logger with the specified name.

    Args:
        name: The name of the logger, typically __name__ to use the 
            module name

    Returns:
        A configured logger instance
    """
    logger = ConsoleLogger(name)
    return logger


# Legacy utility functions for backward compatibility
LOG_FORMAT = '%(levelname)s - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    stream=sys.stdout,
)


def get_logging_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: The name of the logger, typically __name__ to use the 
            module name

    Returns:
        A configured logger instance
    """
    logger = logging.Logger(name)
    return logger


def set_log_level(level: int) -> None:
    """
    Set the log level for all loggers.

    Args:
        level: The logging level (e.g., logging.DEBUG, logging.INFO)
    """
    logging.getLogger().setLevel(level)


def add_file_handler(log_file: str | Path) -> None:
    """
    Add a file handler to the root logger to write logs to a file.

    Args:
        log_file: Path to the log file
    """
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    )
    logging.getLogger().addHandler(file_handler)
