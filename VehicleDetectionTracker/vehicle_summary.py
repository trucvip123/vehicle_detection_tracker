"""Vehicle summary and reporting utilities."""

from datetime import datetime


def levenshtein_distance(s1, s2):
    """
    Calculate the Levenshtein distance between two strings.
    Used to identify similar license plates (differ by 1-2 characters).
    
    Args:
        s1, s2: Strings to compare
        
    Returns:
        int: Number of character differences
    """
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


def merge_similar_plates(plate_summary, log_func=None):
    THRESHOLD = 2  # Merge plates that differ by up to 2 characters
    plates = list(plate_summary.keys())

    # union-find
    parent = {p: p for p in plates}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(b)] = find(a)

    # nối các biển giống nhau
    for i in range(len(plates)):
        for j in range(i + 1, len(plates)):
            if levenshtein_distance(plates[i], plates[j]) <= THRESHOLD:
                union(plates[i], plates[j])

    # gom nhóm
    groups = {}
    for p in plates:
        root = find(p)
        groups.setdefault(root, []).append(p)

    # build result
    result = {}
    for group in groups.values():
        represent = max(group, key=lambda p: plate_summary[p])
        total = sum(plate_summary[p] for p in group)
        result[represent] = total

    return result


def get_today_vehicles_summary(vehicle_last_seen, vehicle_directions, vehicle_plates, log_func=None):
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
    vehicle_last_seen,
    vehicle_directions,
    vehicle_plates,
    log_func,
    send_telegram_func,
    date_str=None,
):
    """
    Send Telegram notification with summary of vehicles that entered today with license plate details.
    
    Args:
        vehicle_last_seen: Dict of {track_id: timestamp}
        vehicle_directions: Dict of {track_id: direction_label}
        vehicle_plates: Dict of {track_id: plate_text}
        log_func: Logging function
        send_telegram_func: Function to send Telegram notification
        date_str: Date in YYYYMMDD format. If None, use today.
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y%m%d")
    
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
        
        # Format date for readable display: YYYYMMDD -> YYYY-MM-DD
        formatted_date = f"{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]}"
        
        # Build detailed message
        msg = f"Tổng hợp xe vào ngày {formatted_date}: {len(vehicles_today)} xe vào khu vực mỏ\n"
        msg += "━" * 40 + "\n"
        
        # Sort plates by count (highest first) then alphabetically
        sorted_plates = sorted(
            plate_summary.items(), 
            key=lambda x: (-x[1], x[0])
        )
        
        for plate_text, count in sorted_plates:
            msg += f"📍 Biển số {plate_text}: {count} xe\n"
        
        msg += "━" * 40
        
        send_telegram_func(msg)
        log_func(f"Telegram notification sent for daily summary: {msg}")
    except Exception as e:
        log_func(f"Failed to send Telegram summary notification: {e}")
