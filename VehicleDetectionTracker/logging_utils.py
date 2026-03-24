"""Logging utilities."""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional


def _ensure_log_dir() -> Path:
    """Ensure logs directory exists."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    return log_dir


def _write_to_log_file(message: str, log_category: str = "app") -> None:
    """Write log message to single daily log file (all messages in one file per day)."""
    try:
        log_dir = _ensure_log_dir()
        # All logs go to single file per day: {date}.log
        log_file = log_dir / f"{datetime.now().strftime('%Y-%m-%d')}.log"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")
    except Exception as e:
        # Don't fail if logging fails, just print error
        print(f"Error writing to log file: {e}")


def log(message: str, category: str = "app") -> None:
    """
    Print log message with datetime timestamp and save to single daily log file.

    Args:
        message (str): Log message
        category (str): Log category (ignored - all go to same file)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"

    # Print to console
    print(log_message)

    # Write to file (category ignored, all in same file)
    _write_to_log_file(log_message)


def log_plate(track_id: int, message: str, category: str = "plate") -> None:
    """
    Print plate processing log message with track_id for easy identification.

    Args:
        track_id: Vehicle track ID (int or str, including versioned IDs like "2_v2")
        message (str): Log message (without track_id prefix)
        category (str): Log category (ignored - all go to same file)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Format: [timestamp] [TRACK:id] message
    log_message = f"[{timestamp}] [TRACK:{track_id}] {message}"

    # Print to console
    print(log_message)

    # Write to file (category ignored, all in same file)
    _write_to_log_file(log_message)
