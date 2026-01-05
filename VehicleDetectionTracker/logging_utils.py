"""Logging utilities."""

import os
from datetime import datetime
from pathlib import Path


def _ensure_log_dir():
    """Ensure logs directory exists."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    return log_dir


def _write_to_log_file(message, log_category="app"):
    """Write log message to appropriate log file."""
    try:
        log_dir = _ensure_log_dir()
        log_file = log_dir / f"{log_category}_{datetime.now().strftime('%Y-%m-%d')}.log"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")
    except Exception as e:
        # Don't fail if logging fails, just print error
        print(f"Error writing to log file: {e}")


def log(message, category="app"):
    """
    Print log message with datetime timestamp and save to file.
    
    Args:
        message (str): Log message
        category (str): Log category (app, config, telegram, ocr, etc.)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    
    # Print to console
    print(log_message)
    
    # Write to file
    _write_to_log_file(log_message, category)
