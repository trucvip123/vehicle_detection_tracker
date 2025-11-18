"""
Utilities for vehicle tracking, speed calculation and direction estimation.
"""
import math

def map_direction_to_label(direction):
    """Map direction angle to human-readable label."""
    direction_ranges = {
        (-math.pi / 8, math.pi / 8): "Right",
        (math.pi / 8, 3 * math.pi / 8): "Bottom Right",
        (3 * math.pi / 8, 5 * math.pi / 8): "Bottom",
        (5 * math.pi / 8, 7 * math.pi / 8): "Bottom Left",
        (7 * math.pi / 8, -7 * math.pi / 8): "Left",
        (-7 * math.pi / 8, -5 * math.pi / 8): "Top Left",
        (-5 * math.pi / 8, -3 * math.pi / 8): "Top",
        (-3 * math.pi / 8, -math.pi / 8): "Top Right",
    }
    for angle_range, label in direction_ranges.items():
        if angle_range[0] <= direction <= angle_range[1]:
            return label
    return "Unknown"

def calculate_speed_and_direction(timestamps, positions):
    """
    Calculate speed and direction from position history.
    
    Args:
        timestamps: List of timestamps
        positions: List of (x,y) positions
        
    Returns:
        dict: Speed and direction information
    """
    if len(timestamps) < 2:
        return {
            "kph": None,
            "reliability": 0.0,
            "direction_label": None,
            "direction": None
        }

    # Calculate speeds
    delta_t_list = []
    distance_list = []
    
    for i in range(1, len(timestamps)):
        t1, t2 = timestamps[i - 1], timestamps[i]
        delta_t = t2.timestamp() - t1.timestamp()
        if delta_t > 0:
            x1, y1 = positions[i - 1]
            x2, y2 = positions[i]
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            delta_t_list.append(delta_t)
            distance_list.append(distance)

    speeds = [
        distance / delta_t
        for distance, delta_t in zip(distance_list, delta_t_list)
    ]
    
    # Calculate average speed
    if len(speeds) > 0:
        avg_speed_mps = sum(speeds) / len(speeds)
        speed_kph = convert_mps_to_kmph(avg_speed_mps)
    else:
        speed_kph = None

    # Calculate direction
    initial_x, initial_y = positions[0]
    final_x, final_y = positions[-1]
    direction = math.atan2(final_y - initial_y, final_x - initial_x)
    direction_label = map_direction_to_label(direction)

    # Calculate reliability
    if len(timestamps) < 5:
        reliability = 0.5  # Low reliability
    elif len(timestamps) < 10:
        reliability = 0.7  # Moderate reliability
    else:
        reliability = 1.0  # High reliability

    return {
        "kph": speed_kph,
        "reliability": reliability,
        "direction_label": direction_label,
        "direction": direction
    }

def convert_mps_to_kmph(meters_per_second):
    """Convert speed from m/s to km/h."""
    return meters_per_second * 3.6

def update_tracking_history(track_history, track_id, x, y, max_history=30):
    """
    Update tracking history for a vehicle.
    
    Args:
        track_history: Dictionary of tracking histories
        track_id: Vehicle ID
        x, y: Current position
        max_history: Maximum number of positions to keep
    """
    if track_id not in track_history:
        track_history[track_id] = []
    
    track = track_history[track_id]
    track.append((float(x), float(y)))
    
    if len(track) > max_history:
        track.pop(0)
    
    return track