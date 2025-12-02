"""
Logging Utilities for AuroraAI Router

Configurable logging setup for the router package.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


# Global logger cache
_loggers = {}


def setup_logger(
    name: str = "auroraai_router",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with console and optional file output.
    
    Args:
        name: Logger name
        level: Logging level (default: INFO)
        log_file: Optional path to log file
        format_string: Custom format string
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if name in _loggers:
        return _loggers[name]
    
    logger.setLevel(level)
    
    # Default format
    if format_string is None:
        format_string = "[%(asctime)s] %(levelname)s - %(name)s - %(message)s"
    
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    _loggers[name] = logger
    return logger


def get_logger(name: str = "auroraai_router") -> logging.Logger:
    """
    Get an existing logger or create a default one.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    if name in _loggers:
        return _loggers[name]
    
    return setup_logger(name)


class LogContext:
    """
    Context manager for temporary logging configuration.
    
    Example:
        with LogContext(level=logging.DEBUG):
            # Detailed logging here
            process_data()
    """
    
    def __init__(
        self,
        name: str = "auroraai_router",
        level: Optional[int] = None
    ):
        self.name = name
        self.level = level
        self.original_level = None
    
    def __enter__(self):
        logger = get_logger(self.name)
        self.original_level = logger.level
        
        if self.level is not None:
            logger.setLevel(self.level)
        
        return logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logger = get_logger(self.name)
        logger.setLevel(self.original_level)
        return False


def log_progress(
    current: int,
    total: int,
    prefix: str = "Progress",
    logger: Optional[logging.Logger] = None,
    log_every: int = 100
):
    """
    Log progress updates periodically.
    
    Args:
        current: Current step
        total: Total steps
        prefix: Log message prefix
        logger: Logger to use (defaults to main logger)
        log_every: Log every N steps
    """
    if current % log_every == 0 or current == total:
        if logger is None:
            logger = get_logger()
        
        percent = (current / total) * 100 if total > 0 else 0
        logger.info(f"{prefix}: {current}/{total} ({percent:.1f}%)")
