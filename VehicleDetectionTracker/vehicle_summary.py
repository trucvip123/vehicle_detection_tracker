"""Vehicle summary and reporting utilities."""

from datetime import datetime
from typing import Optional, Dict, Any, Callable, Union


def levenshtein_distance(s1: Optional[str], s2: Optional[str]) -> Union[int, float]:
    """
    Calculate the Levenshtein distance between two strings.
    Used to identify similar license plates (differ by 1-2 characters).
    
    Args:
        s1, s2: Strings to compare
        
    Returns:
        int: Number of character differences
    """
    # Handle None values gracefully
    if s1 is None or s2 is None:
        return float('inf')  # Consider None as completely different
    
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # j+1 instead of j since previous_row and current_row are one character longer
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def normalize_plate_parts(plate: str) -> Optional[Dict[str, str]]:
    """
    Parse Vietnamese license plate into components.
    
    Expected format: XX-YYY.ZZ or XXX-YYY.ZZ or XXXX-YYY.ZZ
    where:
    - XX/XXX/XXXX: province code (mixed alphanumeric)
    - YYY: serial number (3 digits)
    - ZZ: check digits (2 digits)
    
    Returns:
        Dict with 'province', 'serial', 'check' keys, or None if invalid format
    """
    if not plate or not isinstance(plate, str):
        return None
    
    plate = plate.strip()
    try:
        # Split by '-' to separate province from the rest
        parts = plate.split('-')
        if len(parts) != 2:
            return None
        
        province = parts[0].strip()
        rest = parts[1].strip()  # e.g., "151.58"
        
        # Split rest by '.' to get serial and check digits
        rest_parts = rest.split('.')
        if len(rest_parts) != 2:
            return None
        
        serial = rest_parts[0].strip()
        check = rest_parts[1].strip()
        
        # Validate: serial/check should be digits
        if not serial.isdigit() or not check.isdigit():
            return None
        
        return {
            'province': province,
            'serial': serial,
            'check': check
        }
    except Exception:
        return None


def merge_similar_plates(plate_summary: Dict[str, int], log_func: Optional[Callable[[str], None]] = None) -> Dict[str, int]:
    THRESHOLD = 2  # Merge plates that differ by up to 2 characters
    plates = list(plate_summary.keys())
    
    if not plates:
        return {}

    # union-find
    parent = {p: p for p in plates}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(b)] = find(a)

    # Merge logic: different rules for different cases
    for i in range(len(plates)):
        for j in range(i + 1, len(plates)):
            plate1 = plates[i]
            plate2 = plates[j]
            
            # Rule 1: Check if only province code differs (serial+check match)
            parts1 = normalize_plate_parts(plate1)
            parts2 = normalize_plate_parts(plate2)
            
            if parts1 and parts2:
                # If serial + check are the same but province differs → merge
                if (parts1['serial'] == parts2['serial'] and 
                    parts1['check'] == parts2['check'] and
                    parts1['province'] != parts2['province']):
                    if log_func:
                        log_func(f"[MERGE] Province diff only: {plate1} + {plate2}")
                    union(plate1, plate2)
                    continue
            
            # Rule 2: Traditional Levenshtein distance check
            if levenshtein_distance(plate1, plate2) <= THRESHOLD:
                union(plate1, plate2)

    # Group merged plates
    groups = {}
    for p in plates:
        root = find(p)
        groups.setdefault(root, []).append(p)

    # Build result: use plate with highest count as representative
    result = {}
    for group in groups.values():
        represent = max(group, key=lambda p: plate_summary[p])
        total = sum(plate_summary[p] for p in group)
        result[represent] = total

    return result


def get_today_vehicles_summary(vehicle_last_seen: Dict[int, datetime], vehicle_directions: Dict[int, str], vehicle_plates: Dict[int, str], log_func: Optional[Callable[[str], None]] = None) -> list:
    """
    Get summary of vehicles (plates and counts) from today's tracking data.
    
    Args:
        vehicle_last_seen: Dict of {track_id: timestamp}
        vehicle_directions: Dict of {track_id: direction_label}
        vehicle_plates: Dict of {track_id: plate_text}
        log_func: Optional logging function
        
    Returns:
        list: List of tuples (plate_text, count) sorted by count (descending)
    """
    try:
        today_str = datetime.now().strftime("%Y%m%d")
        
        # Get vehicles that entered today (direction contains "bottom")
        vehicles_today = [
            tid
            for tid, ts in vehicle_last_seen.items()
            if hasattr(ts, "strftime")
            and ts.strftime("%Y%m%d") == today_str
            and "bottom" in vehicle_directions.get(tid, "").lower()
        ]
        
        # Build plate summary
        plate_summary = {}
        for track_id in vehicles_today:
            plate_text = vehicle_plates.get(track_id, "?")
            if plate_text not in plate_summary:
                plate_summary[plate_text] = 0
            plate_summary[plate_text] += 1
        
        # Merge similar plates (differ by 1-2 characters)
        plate_summary = merge_similar_plates(plate_summary, log_func)
        
        # Sort by count (descending)
        sorted_plates = sorted(
            plate_summary.items(),
            key=lambda x: (-x[1], x[0])
        )
        
        return sorted_plates
    except Exception as e:
        if log_func:
            log_func(f"[ERROR] Error getting today's vehicles summary: {e}")
        return []


def save_daily_vehicle_summary(
    vehicle_last_seen: Dict[int, datetime],
    vehicle_directions: Dict[int, str],
    vehicle_plates: Dict[int, str],
    log_func: Callable[[str], None],
    send_telegram_func: Callable[[str, Optional[str]], None],
    date_str: Optional[str] = None,
    vehicle_plate_counts: Optional[Dict[int, Dict[str, int]]] = None,
) -> None:
    """
    Send Telegram notification with summary of vehicles that entered today with license plate details.
    
    Args:
        vehicle_last_seen: Dict of {track_id: timestamp}
        vehicle_directions: Dict of {track_id: direction_label}
        vehicle_plates: Dict of {track_id: plate_text}
        log_func: Logging function
        send_telegram_func: Function to send Telegram notification
        date_str: Date in YYYYMMDD format. If None, use today.
        vehicle_plate_counts: Dict of {track_id: {plate_text: count}} for detection counts
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y%m%d")
    
    # Initialize vehicle_plate_counts if not provided
    if vehicle_plate_counts is None:
        vehicle_plate_counts = {}
    
    # Debug log for tracking
    log_func(f"[DEBUG] vehicle_last_seen: {vehicle_last_seen}")
    log_func(f"[DEBUG] vehicle_directions: {vehicle_directions}")
    for tid, direction in vehicle_directions.items():
        log_func(f"[SUMMARY] vehicle_id={tid} direction={direction}")
    for tid, ts in vehicle_last_seen.items():
        date_match = (
            ts.strftime("%Y%m%d") == date_str if hasattr(ts, "strftime") else False
        )
        log_func(f"[SUMMARY] vehicle_id={tid} last_seen={ts} today={date_match}")
    
    # Only count vehicles with direction_label indicating entry (e.g., 'bottom')
    vehicles_today = [
        tid
        for tid, ts in vehicle_last_seen.items()
        if hasattr(ts, "strftime")
        and ts.strftime("%Y%m%d") == date_str
        and "bottom" in vehicle_directions.get(tid, "").lower()
    ]
    log_func(f"[DEBUG] vehicles_today: {vehicles_today}")
    
    try:
        # Group vehicles by plate number for detailed summary
        # Use detection counts from vehicle_plate_counts if available
        plate_summary = {}
        for track_id in vehicles_today:
            plate_text = vehicle_plates.get(track_id, "unknown")
            if plate_text not in plate_summary:
                plate_summary[plate_text] = 0
            # Get detection count for this vehicle (default to 1 if not found)
            if track_id in vehicle_plate_counts and plate_text in vehicle_plate_counts[track_id]:
                detection_count = vehicle_plate_counts[track_id][plate_text]
            else:
                detection_count = 1
            plate_summary[plate_text] += detection_count
        
        # Merge similar plates (differ by 1-2 characters)
        plate_summary = merge_similar_plates(plate_summary, log_func)
        
        # Filter out unknown/None plates from the summary
        plate_summary_filtered = {
            plate: count for plate, count in plate_summary.items()
            if plate and str(plate).strip().lower() != "unknown"
        }
        
        # Count vehicles with identified plates (for summary)
        vehicles_with_plates = sum(plate_summary_filtered.values()) if plate_summary_filtered else 0
        
        # Format date for readable display: YYYYMMDD -> YYYY-MM-DD
        formatted_date = f"{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]}"
        
        # Build detailed message (only include vehicles with identified plates)
        msg = f"Tổng hợp xe vào ngày {formatted_date}: {vehicles_with_plates} xe vào khu vực mỏ\n"
        msg += "━" * 30 + "\n"
        
        # Sort plates by count (highest first) then alphabetically
        sorted_plates = sorted(
            plate_summary_filtered.items(), 
            key=lambda x: (-x[1], x[0])
        )
        
        for plate_text, count in sorted_plates:
            msg += f"📍 Biển số {plate_text}: {count} xe\n"
        
        msg += "━" * 30
        
        send_telegram_func(msg)
        log_func(f"Telegram notification sent for daily summary: {msg}")
    except Exception as e:
        log_func(f"Failed to send Telegram summary notification: {e}")
