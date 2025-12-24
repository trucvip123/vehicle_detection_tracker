"""
Time scheduler module for managing operating hours of vehicle detection system.
Supports scheduling based on start/end hours with timezone support.
"""

from datetime import datetime
from VehicleDetectionTracker.config_loader import get_config


def get_time_info():
    """
    Get current time information including the formatted time string.

    Returns:
        dict: Dictionary containing:
            - 'current_time': Formatted time string (HH:MM:SS)
            - 'hour': Current hour (0-23)
            - 'minute': Current minute (0-59)
            - 'second': Current second (0-59)
            - 'datetime': datetime object
    """
    config = get_config()
    operating_hours_config = config.get("operating_hours", {})
    timezone_offset = operating_hours_config.get("timezone_offset", 0)

    # Get current time (you can adjust this to use timezone_offset if needed)
    current_datetime = datetime.now()

    return {
        "current_time": current_datetime.strftime("%H:%M:%S"),
        "hour": current_datetime.hour,
        "minute": current_datetime.minute,
        "second": current_datetime.second,
        "datetime": current_datetime,
        "timezone_offset": timezone_offset,
    }


def is_outside_operating_hours():
    """
    Check if current time is outside the configured operating hours.

    Operating hours are defined in config.yaml under 'operating_hours' section:
    - enabled: Whether operating hours scheduling is active
    - start_hour: Hour when processing should start (0-23)
    - end_hour: Hour when processing should end (0-23)

    Special cases:
    - If start_hour < end_hour: Process from start_hour to end_hour same day
    - If start_hour > end_hour: Process from start_hour to 23:59 and 0:00 to end_hour (overnight)
    - If start_hour == end_hour: Always process (24/7)

    Returns:
        bool: True if currently outside operating hours, False if within operating hours
    """
    config = get_config()
    operating_hours_config = config.get("operating_hours", {})

    # Check if operating hours scheduling is enabled
    enabled = operating_hours_config.get("enabled", False)
    if not enabled:
        # Operating hours scheduling is disabled, always within operating hours
        return False

    start_hour = operating_hours_config.get("start_hour", 6)
    end_hour = operating_hours_config.get("end_hour", 20)

    time_info = get_time_info()
    current_hour = time_info["hour"]

    # Edge case: if start_hour == end_hour, always process (24/7)
    if start_hour == end_hour:
        return False

    # Case 1: start_hour < end_hour (normal operating hours within same day)
    # Example: 6 AM to 8 PM
    if start_hour < end_hour:
        if start_hour <= current_hour < end_hour:
            return False  # Within operating hours
        else:
            return True  # Outside operating hours

    # Case 2: start_hour > end_hour (overnight operating hours)
    # Example: 8 PM (20) to 6 AM (6) - process from 20:00 to 23:59 and 0:00 to 6:00
    else:
        if current_hour >= start_hour or current_hour < end_hour:
            return False  # Within operating hours
        else:
            return True  # Outside operating hours


def get_operating_hours_info():
    """
    Get detailed information about the configured operating hours.

    Returns:
        dict: Dictionary containing:
            - 'enabled': Whether operating hours scheduling is enabled
            - 'start_hour': Start hour (0-23)
            - 'end_hour': End hour (0-23)
            - 'current_hour': Current hour
            - 'is_outside': Whether currently outside operating hours
            - 'schedule_type': 'daily' or 'overnight' or '24/7'
            - 'description': Human-readable description of operating hours
    """
    config = get_config()
    operating_hours_config = config.get("operating_hours", {})

    enabled = operating_hours_config.get("enabled", False)
    start_hour = operating_hours_config.get("start_hour", 6)
    end_hour = operating_hours_config.get("end_hour", 20)

    time_info = get_time_info()
    current_hour = time_info["hour"]
    is_outside = is_outside_operating_hours()

    # Determine schedule type
    if not enabled:
        schedule_type = "disabled"
        description = "Operating hours scheduling is disabled (24/7 operation)"
    elif start_hour == end_hour:
        schedule_type = "24/7"
        description = "24/7 operation enabled"
    elif start_hour < end_hour:
        schedule_type = "daily"
        description = f"Daily operation: {start_hour:02d}:00 to {end_hour:02d}:00"
    else:
        schedule_type = "overnight"
        description = f"Overnight operation: {start_hour:02d}:00 to {end_hour:02d}:00 (crosses midnight)"

    return {
        "enabled": enabled,
        "start_hour": start_hour,
        "end_hour": end_hour,
        "current_hour": current_hour,
        "is_outside": is_outside,
        "schedule_type": schedule_type,
        "description": description,
    }
