"""Logging utilities for Gaussia."""

import logging


class VerboseLogger:
    """
    Custom logger class that handles verbose logging.
    Only logs messages when verbose mode is enabled.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.setLevel(logging.DEBUG)

    def info(self, message: str):
        """Log info message if verbose is enabled"""
        if self.verbose:
            self.logger.info(message)

    def debug(self, message: str):
        """Log debug message if verbose is enabled"""
        if self.verbose:
            self.logger.debug(message)

    def warning(self, message: str):
        """Log warning message if verbose is enabled"""
        if self.verbose:
            self.logger.warning(message)

    def error(self, message: str):
        """Log error message if verbose is enabled"""
        if self.verbose:
            self.logger.error(message)

    def exception(self, message: str):
        """Log exception message with traceback if verbose is enabled"""
        if self.verbose:
            self.logger.exception(message)


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
