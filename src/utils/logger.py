"""
Logging utilities for NeuroUrban application.
"""

import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True
) -> None:
    """
    Setup logging configuration for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        console_output: Whether to output logs to console
    """

    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Set up formatters
    formatter = logging.Formatter(log_format, date_format)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler with UTF-8 encoding
    if console_output:
        # Try to set UTF-8 encoding for Windows console
        if sys.platform.startswith('win'):
            try:
                # Set console to UTF-8 mode
                os.system('chcp 65001 >nul 2>&1')
                console_handler = logging.StreamHandler(sys.stdout)
            except:
                # Fallback to stderr which usually handles Unicode better
                console_handler = logging.StreamHandler(sys.stderr)
        else:
            console_handler = logging.StreamHandler(sys.stdout)

        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler with UTF-8 encoding
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"neurourban_{timestamp}.log"

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Log the setup
    logger = logging.getLogger(__name__)
    logger.info(f"Logging setup complete - Level: {log_level}")
    logger.info(f"Log file: {log_file}")

def safe_emoji_text(emoji_text: str, fallback_text: str = None) -> str:
    """
    Return emoji text if supported, otherwise fallback text.

    Args:
        emoji_text: Text with emoji characters
        fallback_text: Fallback text without emojis

    Returns:
        Safe text for current platform
    """
    if fallback_text is None:
        # Remove emojis and clean up text
        import re
        fallback_text = re.sub(r'[^\w\s\-\.\,\!\?\:\;]', '', emoji_text).strip()
        # Clean up multiple spaces
        fallback_text = re.sub(r'\s+', ' ', fallback_text)

    # Check if we're on Windows with encoding issues
    if sys.platform.startswith('win'):
        try:
            # Test if we can encode the emoji text
            emoji_text.encode('cp1252')
            return emoji_text
        except UnicodeEncodeError:
            return fallback_text

    return emoji_text

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
