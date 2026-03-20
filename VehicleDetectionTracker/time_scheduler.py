"""
Time scheduler module for managing operating hours of vehicle detection system.
Supports scheduling based on start/end hours with timezone support.
"""

from datetime import datetime, timedelta
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
    Check if current time is outside the configured operating hours and return the next end_time for notification.
    Checks both hours AND minutes for accurate time-based scheduling.

    Returns:
        tuple: (is_outside: bool, next_end_time: datetime)
    """
    config = get_config()
    operating_hours_config = config.get("operating_hours", {})

    enabled = operating_hours_config.get("enabled", False)
    start_hour = operating_hours_config.get("start_hour", 6)
    end_hour = operating_hours_config.get("end_hour", 20)

    time_info = get_time_info()
    now = time_info["datetime"]

    # Edge case: if start_hour == end_hour, always process (24/7)
    if not enabled or start_hour == end_hour:
        return (False, None)

    # Create start and end times with current date at HH:00:00
    start_time = now.replace(hour=start_hour, minute=0, second=0, microsecond=0)
    end_time = now.replace(hour=end_hour, minute=0, second=0, microsecond=0)

    # Calculate next end_time (when operating hours end next)
    if start_hour < end_hour:
        # Normal case: same day (e.g., 6:00 AM to 6:00 PM)
        if now >= end_time:
            # Already past end_hour today, next end_time is tomorrow
            end_time = end_time + timedelta(days=1)
        is_outside = not (start_time <= now < end_time)
    else:
        # Overnight case (e.g., 10:00 PM to 6:00 AM)
        if now >= start_time:
            # After start_hour, end_time is tomorrow
            end_time = (now + timedelta(days=1)).replace(
                hour=end_hour, minute=0, second=0, microsecond=0
            )
            is_outside = False
        elif now < end_time:
            # Before end_hour (still in night hours)
            end_time = now.replace(hour=end_hour, minute=0, second=0, microsecond=0)
            is_outside = False
        else:
            # Outside operating hours (between end_hour and start_hour)
            end_time = now.replace(hour=end_hour, minute=0, second=0, microsecond=0)
            is_outside = True

    return (is_outside, end_time)


def get_operating_hours_info():
    """
    Get detailed information about the configured operating hours.

    Returns:
        dict: Dictionary containing:
            - 'enabled': Whether operating hours scheduling is enabled
            - 'start_hour': Start hour (0-23)
            - 'end_hour': End hour (0-23)
            - 'current_time': Current time in HH:MM:SS format
            - 'is_outside': Whether currently outside operating hours
            - 'schedule_type': 'daily' or 'overnight' or '24/7' or 'disabled'
            - 'description': Human-readable description of operating hours
    """
    config = get_config()
    operating_hours_config = config.get("operating_hours", {})

    enabled = operating_hours_config.get("enabled", False)
    start_hour = operating_hours_config.get("start_hour", 6)
    end_hour = operating_hours_config.get("end_hour", 20)

    time_info = get_time_info()
    current_time = time_info["current_time"]
    is_outside, _ = is_outside_operating_hours()

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
        "current_time": current_time,
        "is_outside": is_outside,
        "status": "Outside operating hours ❌" if is_outside else "Within operating hours ✓",
        "schedule_type": schedule_type,
        "description": description,
    }
