"""Logging utilities."""

from datetime import datetime


def log(message):
    """Print log message with datetime timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
