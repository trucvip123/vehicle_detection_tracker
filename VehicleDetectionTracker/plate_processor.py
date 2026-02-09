"""License plate detection and processing utilities."""

from datetime import datetime
import threading
import cv2
import json
import os
from pathlib import Path
from VehicleDetectionTracker.plate_utils import detect_license_plate_sync
from VehicleDetectionTracker.utils.send_bot import send_notify_to_telegram
from VehicleDetectionTracker.logging_utils import log_plate


# Global sets to track vehicle notifications
_vehicle_telegram_sent_with_plate = set()  # Vehicles that sent notification with plate
_vehicle_telegram_sent_without_plate = set()  # Vehicles that sent notification without plate
_vehicle_telegram_sent_lock = threading.Lock()


def reset_telegram_sent():
    """Reset the telegram sent tracking set."""
    global _vehicle_telegram_sent_with_plate, _vehicle_telegram_sent_without_plate
    _vehicle_telegram_sent_with_plate.clear()
    _vehicle_telegram_sent_without_plate.clear()


def reset_daily_tracking():
    """Reset daily tracking data (to be called at midnight or start of new day)."""
    reset_telegram_sent()


class PlateProcessor:
    """Handles license plate detection and tracking."""

    def __init__(self, plate_model, ocr_reader, executor, log_func):
        self.plate_model = plate_model
        self.ocr_reader = ocr_reader
        self.executor = executor
        self.log = log_func

        # Track detected plates per vehicle
        self.vehicle_plates = {}  # {track_id: plate_text}
        self.vehicle_plate_counts = {}  # {track_id: {plate_text: count}}
        self.vehicle_directions = {}  # {track_id: direction_label}
        self.vehicle_last_seen = {}  # {track_id: timestamp}
        self.vehicle_missing_frames = {}  # {track_id: missing_frame_count}

        self._model_lock = threading.Lock()

        # Persistence settings
        self.state_dir = "vehicle_state"  # Directory to store state files
        self._state_lock = threading.Lock()
        self._last_reset_date = None  # Track last reset date for daily reset
        
        # Create state directory if it doesn't exist
        os.makedirs(self.state_dir, exist_ok=True)

        # Load persisted state if it exists
        self._load_state()

    def save_daily_vehicle_summary(self, date_str=None):
        """
        Gửi thông báo Telegram tổng hợp số lượng xe đi vào trong ngày với chi tiết biển số.
        Args:
            date_str (str): Date in YYYYMMDD format. If None, use today.
        """
        from VehicleDetectionTracker.utils.send_bot import send_warning_to_telegram

        if date_str is None:
            date_str = datetime.now().strftime("%Y%m%d")
        
        # Debug log for tracking
        self.log(f"[DEBUG] vehicle_last_seen: {self.vehicle_last_seen}")
        self.log(f"[DEBUG] vehicle_directions: {self.vehicle_directions}")
        for tid, direction in self.vehicle_directions.items():
            self.log(f"[SUMMARY] vehicle_id={tid} direction={direction}")
        for tid, ts in self.vehicle_last_seen.items():
            date_match = (
                ts.strftime("%Y%m%d") == date_str if hasattr(ts, "strftime") else False
            )
            self.log(f"[SUMMARY] vehicle_id={tid} last_seen={ts} today={date_match}")
        
        # Only count vehicles with direction_label indicating entry (e.g., 'bottom')
        vehicles_today = [
            tid
            for tid, ts in self.vehicle_last_seen.items()
            if hasattr(ts, "strftime")
            and ts.strftime("%Y%m%d") == date_str
            and "bottom" in self.vehicle_directions.get(tid, "").lower()
        ]
        self.log(f"[DEBUG] vehicles_today: {vehicles_today}")
        
        try:
            # Group vehicles by plate number for detailed summary
            plate_summary = {}
            for track_id in vehicles_today:
                plate_text = self.vehicle_plates.get(track_id, "unknown")
                if plate_text not in plate_summary:
                    plate_summary[plate_text] = 0
                plate_summary[plate_text] += 1
            
            # Merge similar plates (differ by 1-2 characters)
            plate_summary = self._merge_similar_plates(plate_summary)
            
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
            
            send_warning_to_telegram(msg)
            self.log(f"Telegram notification sent for daily summary: {msg}")
        except Exception as e:
            self.log(f"Failed to send Telegram summary notification: {e}")

    def _get_first_vehicle_image(self, vehicle_dir, track_id):
        """
        Get the first image file from the vehicle directory.
        
        Args:
            vehicle_dir: Path to vehicle screenshots directory
            track_id: Vehicle track ID for logging
            
        Returns:
            str: Path to first image file or None if not found
        """
        try:
            if os.path.exists(vehicle_dir):
                # Find all PNG files in the directory
                image_files = [
                    f for f in os.listdir(vehicle_dir) 
                    if f.endswith('.png')
                ]
                
                if image_files:
                    # Sort to get the first (earliest) image
                    image_files.sort()
                    image_path = os.path.join(vehicle_dir, image_files[-1])
                    self.log(
                        f"[PLATE] vehicle_id={track_id} found first image: {image_files[-1]}"
                    )
                    return image_path
                else:
                    self.log(
                        f"[PLATE] vehicle_id={track_id} no images found in {vehicle_dir}"
                    )
                    return None
            else:
                self.log(
                    f"[PLATE] vehicle_id={track_id} directory not found: {vehicle_dir}"
                )
                return None
        except Exception as e:
            self.log(
                f"[PLATE] vehicle_id={track_id} error finding image: {e}"
            )
            return None

    def get_most_detected_plate(self, track_id):
        """
        Get the license plate with highest detection count for a vehicle.

        Args:
            track_id: Vehicle track ID

        Returns:
            tuple: (plate_text, count) or (None, 0) if no plates detected
        """
        if (
            track_id not in self.vehicle_plate_counts
            or not self.vehicle_plate_counts[track_id]
        ):
            return None, 0

        plate_counts = self.vehicle_plate_counts[track_id]
        # Find plate with maximum count
        most_detected_plate = max(plate_counts.items(), key=lambda x: x[1])
        return most_detected_plate

    def _sanitize_filename(self, filename):
        """
        Sanitize filename by removing/replacing invalid characters.
        
        Args:
            filename: Original filename string
            
        Returns:
            str: Sanitized filename safe for filesystem
        """
        import re
        # Replace invalid filename characters with underscore
        # Invalid chars: < > : " / \ | ? *
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Remove leading/trailing spaces and dots
        sanitized = sanitized.strip('. ')
        return sanitized if sanitized else "unknown"

    def process_plate_background_sync(
        self,
        track_id,
        vehicle_frame,
        direction_label=None,
        frame_timestamp=None,
        timestamp_str=None,
        vehicle_dir="screenshots",
    ):
        """
        Sync wrapper for background plate processing using ThreadPoolExecutor.
        Tracks license plate detection counts.

        Args:
            track_id: Vehicle track ID
            vehicle_frame: Cropped vehicle frame
            direction_label: Vehicle direction label (if available)
            frame_timestamp: Actual detection datetime object
            timestamp_str: Formatted timestamp string for logging/files
            vehicle_dir: Vehicle directory for saving files
        """
        self.log(f"[PLATE] vehicle_id={track_id} Processing plate background")
        try:
            # Update last_seen timestamp if available
            if frame_timestamp:
                self.vehicle_last_seen[track_id] = frame_timestamp
            
            # Always update direction if available (not just when plate is detected)
            if direction_label:
                self.vehicle_directions[track_id] = direction_label
                log_plate(track_id, f"direction_set={direction_label}")
            elif track_id not in self.vehicle_directions:
                self.vehicle_directions[track_id] = "Unknown"

            # Detect license plate
            license_plate_info = detect_license_plate_sync(
                self.plate_model,
                vehicle_frame,
                self.ocr_reader,
                self._model_lock,
                timestamp_str,
                vehicle_dir=vehicle_dir,
            )
            plate_text = license_plate_info.get("text") if license_plate_info else None
            count_detections = license_plate_info.get("count") if license_plate_info else 0
            # Ensure count_detections is always an integer (never None)
            if count_detections is None:
                count_detections = 0
            log_plate(track_id, f"detected_plate={plate_text}")
            
            # Send Telegram notification logic
            global _vehicle_telegram_sent_with_plate, _vehicle_telegram_sent_without_plate, _vehicle_telegram_sent_lock
            
            # Only send notifications for vehicles entering (not exiting)
            # Skip if direction contains "top" (exit/ra khỏi) or is "Unknown"
            is_entering = direction_label and "top" not in direction_label.lower()
            
            if not is_entering:
                log_plate(track_id, f"Skipping notification - vehicle is exiting (direction={direction_label})")
                return
            
            # Track effective_track_id for use across both cases
            effective_track_id = track_id  # Default: use original track_id
            
            with _vehicle_telegram_sent_lock:
                # Case 1: Plate detected - send immediately if not already sent
                if plate_text and plate_text != "unknown":
                    if track_id not in _vehicle_telegram_sent_with_plate:
                        # Check if this is first detection for this track_id
                        is_first_detection = track_id not in self.vehicle_plate_counts
                        
                        # If track_id has plates already, check if it's reused (different vehicle)
                        if not is_first_detection:
                            existing_plates = self.vehicle_plate_counts[track_id]
                            all_existing_plates = list(existing_plates.keys())
                            
                            # Check if new plate is significantly different (not just OCR variation)
                            is_different = True
                            for existing_plate in all_existing_plates:
                                dist = self._levenshtein_distance(plate_text, existing_plate)
                                if dist <= 2:  # Similar plate (OCR variation)
                                    is_different = False
                                    break
                            
                            # If completely different plate, track_id was reused
                            # Create a versioned ID to keep both vehicles' data separate
                            if is_different and len(all_existing_plates) > 0:
                                # Find next version number
                                version = 2
                                while f"{track_id}_v{version}" in self.vehicle_plate_counts:
                                    version += 1
                                
                                effective_track_id = f"{track_id}_v{version}"
                                log_plate(track_id, f"Track ID reused! Creating version ID: {effective_track_id}")
                                log_plate(track_id, f"Keeping old data: {all_existing_plates}")
                                log_plate(effective_track_id, f"Storing new vehicle data: {plate_text}")
                                is_first_detection = True  # Treat versioned ID as first detection
                        
                        # Increment detection count for this plate
                        if effective_track_id not in self.vehicle_plate_counts:
                            self.vehicle_plate_counts[effective_track_id] = {}
                        self.vehicle_plate_counts[effective_track_id][plate_text] = (
                            self.vehicle_plate_counts[effective_track_id].get(plate_text, 0) + 1
                        )
                        
                        # Select primary plate: highest count, then alphabetically (stable)
                        plate_counts = self.vehicle_plate_counts[effective_track_id]
                        best_plate = sorted(
                            plate_counts.items(),
                            key=lambda x: (-x[1], x[0])
                        )[0][0]
                        
                        # Update vehicle_plates with best plate
                        self.vehicle_plates[effective_track_id] = best_plate
                        
                        # Log if multiple plates detected for this vehicle
                        if len(plate_counts) > 1:
                            log_plate(effective_track_id, f"Multiple plates detected: {plate_counts} -> Primary: {best_plate}")
                        
                        # Save state after updating plates
                        self._save_state()
                        
                        # Sanitize plate text for filename
                        sanitized_plate = self._sanitize_filename(plate_text)
                        image_path = os.path.join(vehicle_dir, f"vehicle_{sanitized_plate}.png")
                        cv2.imwrite(image_path, vehicle_frame)
                        
                        log_plate(effective_track_id, f"Sending Telegram notification with plate={plate_text}")
                        send_notify_to_telegram(
                            plate_text,
                            direction_label,
                            frame_timestamp,
                            image_path=image_path,
                        )
                        _vehicle_telegram_sent_with_plate.add(effective_track_id)
                        # Remove from without_plate list if it was there
                        _vehicle_telegram_sent_without_plate.discard(effective_track_id)
                
                # Case 2: Plate is None or unknown - only send if vehicle has been missing for a while
                # (indicating it has finished passing through all frames)
                elif track_id not in _vehicle_telegram_sent_with_plate:
                    missing_frames = self.vehicle_missing_frames.get(track_id, 0)
                    # Ensure missing_frames is always an int (not None)
                    if missing_frames is None:
                        missing_frames = 0
                    
                    log_plate(track_id, f"Plate not detected, count_detections={count_detections}, missing_frames={missing_frames}")
                    # Only send "unknown" notification if vehicle has been missing for a significant time
                    # This ensures we waited for all frames to pass before confirming no plate found
                    MISSING_THRESHOLD = 50  # frames threshold before sending unknown notification
                    
                    # Send notification if: (a) had detections but too blurry to read, OR (b) been missing too long
                    should_send_unknown = (
                        (count_detections > 0 and missing_frames >= MISSING_THRESHOLD) or
                        (missing_frames >= MISSING_THRESHOLD * 2)  # Very old vehicle with no activity
                    )
                    
                    if should_send_unknown and track_id not in _vehicle_telegram_sent_without_plate:
                        # Find first image in vehicle_dir
                        image_path = self._get_first_vehicle_image(vehicle_dir, track_id)
                        self.log(
                                f"[PLATE] vehicle_id={track_id} image_path:{image_path}"
                            )
                        # Send Telegram notification
                        if image_path and os.path.exists(image_path):
                            self.log(
                                f"[PLATE] vehicle_id={track_id} Sending Telegram notification without plate (missing_frames={missing_frames})"
                            )
                            send_notify_to_telegram(
                                "không xác định",
                                direction_label,
                                frame_timestamp,
                                image_path=image_path,
                            )
                        else:
                            self.log(
                                f"[PLATE] vehicle_id={track_id} No image found, sending notification without image (missing_frames={missing_frames})"
                            )
                            send_notify_to_telegram(
                                "không xác định",
                                direction_label,
                                frame_timestamp,
                                image_path=None,
                            )
                        
                        _vehicle_telegram_sent_without_plate.add(track_id)
        except Exception as e:
            self.log(f"[PLATE] vehicle_id={track_id} detection_error: {e}")

    def submit_plate_processing(
        self,
        track_id,
        vehicle_frame,
        direction_label,
        frame_timestamp,
        timestamp_str,
        vehicle_dir,
    ):
        """Submit plate processing to background executor."""
        if vehicle_frame.size > 0:
            self.executor.submit(
                self.process_plate_background_sync,
                track_id,
                vehicle_frame.copy(),
                direction_label,
                frame_timestamp,
                timestamp_str,
                vehicle_dir=vehicle_dir,
            )

    def _levenshtein_distance(self, s1, s2):
        """
        Calculate the Levenshtein distance between two strings.
        Used to identify similar license plates (differ by 1-2 characters).
        
        Args:
            s1, s2: Strings to compare
            
        Returns:
            int: Number of character differences
        """
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
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

    def _merge_similar_plates(self, plate_summary):
        """
        Merge similar license plates (differ by 1-2 characters).
        Select the plate with highest count as representative.
        
        Args:
            plate_summary: Dict of {plate_text: count}
            
        Returns:
            Dict: Merged plate summary
        """
        MAX_DISTANCE = 3  # Merge plates that differ by up to 2 characters
        
        merged = {}
        processed = set()
        
        # Sort plates by count (highest first) for representative selection
        sorted_plates = sorted(
            plate_summary.items(), 
            key=lambda x: -x[1]
        )
        
        for plate_text, count in sorted_plates:
            if plate_text in processed:
                continue
            
            # Find all similar plates
            similar_group = {plate_text: count}
            processed.add(plate_text)
            
            for other_plate, other_count in sorted_plates:
                if other_plate in processed:
                    continue
                
                # Check if similar (distance <= MAX_DISTANCE)
                distance = self._levenshtein_distance(plate_text, other_plate)
                if distance <= MAX_DISTANCE and distance > 0:
                    similar_group[other_plate] = other_count
                    processed.add(other_plate)
            
            # Use the first (highest count) as representative
            representative = plate_text
            total_count = sum(similar_group.values())
            
            # # Log if we merged multiple plates
            # if len(similar_group) > 1:
            #     self.log(
            #         f"[MERGE] Merged similar plates: {similar_group} -> Representative: {representative} ({total_count} xe)"
            #     )
            
            merged[representative] = total_count
        
        return merged

    def _get_state_file_path(self, date_str=None):
        """
        Get the state file path for a specific date.
        
        Args:
            date_str: Date in YYYYMMDD format. If None, use today.
            
        Returns:
            str: Path to state file for the date
        """
        if date_str is None:
            date_str = datetime.now().strftime("%Y%m%d")
        return os.path.join(self.state_dir, f"vehicle_state_{date_str}.json")

    def _load_state(self):
        """Load vehicle state from persisted JSON file for today only."""
        try:
            state_file = self._get_state_file_path()
            today_str = datetime.now().strftime("%Y%m%d")
            
            if os.path.exists(state_file):
                with open(state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)

                # Restore vehicle_last_seen with datetime objects (TODAY ONLY)
                if "vehicle_last_seen" in state:
                    for track_id, timestamp_str in state["vehicle_last_seen"].items():
                        try:
                            # Convert string back to datetime
                            dt = datetime.fromisoformat(timestamp_str)
                            # Only load vehicles from today
                            if dt.strftime("%Y%m%d") == today_str:
                                self.vehicle_last_seen[int(track_id)] = dt
                        except (ValueError, TypeError):
                            pass

                # Get list of valid track_ids from today
                valid_track_ids = set(self.vehicle_last_seen.keys())

                # Restore vehicle_directions (only for valid track_ids)
                if "vehicle_directions" in state:
                    for track_id_str, direction in state["vehicle_directions"].items():
                        track_id = int(track_id_str)
                        if track_id in valid_track_ids:
                            self.vehicle_directions[track_id] = direction

                # Restore vehicle_plates (only for valid track_ids)
                if "vehicle_plates" in state:
                    for track_id_str, plate_text in state["vehicle_plates"].items():
                        track_id = int(track_id_str)
                        if track_id in valid_track_ids:
                            self.vehicle_plates[track_id] = plate_text

                # Restore vehicle_plate_counts (only for valid track_ids)
                if "vehicle_plate_counts" in state:
                    for track_id_str, plate_counts in state["vehicle_plate_counts"].items():
                        track_id = int(track_id_str)
                        if track_id in valid_track_ids:
                            self.vehicle_plate_counts[track_id] = plate_counts

                self.log(
                    f"[PERSIST] Loaded state: {len(self.vehicle_last_seen)} vehicles from today ({today_str}) from {state_file}"
                )
        except Exception as e:
            self.log(f"[PERSIST] Failed to load state: {e}")

    def check_and_reset_daily_tracking(self):
        """Check if it's a new day and reset tracking data if needed."""
        today_str = datetime.now().strftime("%Y%m%d")
        
        # If last reset date is different from today, do daily reset
        if self._last_reset_date != today_str:
            self._last_reset_date = today_str
            self.log(f"[DAILY_RESET] Resetting daily tracking for {today_str}")
            
            # Clear session-specific tracking data (but keep persistent data from JSON)
            self.vehicle_missing_frames.clear()  # Reset missing frame counts
            reset_daily_tracking()  # Reset telegram notification tracking
            
            self.log(f"[DAILY_RESET] ✓ Daily tracking reset completed")

    def _save_state(self):
        """Save vehicle state to JSON file for persistence (one file per day, today only)."""
        try:
            with self._state_lock:
                today_str = datetime.now().strftime("%Y%m%d")
                state_file = self._get_state_file_path()
                
                # Filter: only include vehicles from today
                today_vehicles_last_seen = {}
                for track_id, ts in self.vehicle_last_seen.items():
                    if hasattr(ts, "strftime") and ts.strftime("%Y%m%d") == today_str:
                        today_vehicles_last_seen[str(track_id)] = (
                            ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
                        )
                
                # Get valid track_ids from today
                valid_track_ids = set(int(tid) for tid in today_vehicles_last_seen.keys())
                
                # Filter other dictionaries to only include today's vehicles
                today_directions = {
                    str(tid): direction
                    for tid, direction in self.vehicle_directions.items()
                    if tid in valid_track_ids
                }
                
                today_plates = {
                    str(tid): plate
                    for tid, plate in self.vehicle_plates.items()
                    if tid in valid_track_ids
                }
                
                today_plate_counts = {
                    str(tid): counts
                    for tid, counts in self.vehicle_plate_counts.items()
                    if tid in valid_track_ids
                }
                
                state = {
                    "vehicle_last_seen": today_vehicles_last_seen,
                    "vehicle_directions": today_directions,
                    "vehicle_plates": today_plates,
                    "vehicle_plate_counts": today_plate_counts,
                    "timestamp": datetime.now().isoformat(),
                }

                with open(state_file, "w", encoding="utf-8") as f:
                    json.dump(state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.log(f"[PERSIST] Failed to save state: {e}")

    def get_today_vehicles_summary(self):
        """
        Get summary of vehicles (plates and counts) from today's tracking data.
        
        Returns:
            list: List of tuples (plate_text, count) sorted by count (descending)
        """
        try:
            today_str = datetime.now().strftime("%Y%m%d")
            
            # Get vehicles that entered today (direction contains "bottom")
            vehicles_today = [
                tid
                for tid, ts in self.vehicle_last_seen.items()
                if hasattr(ts, "strftime")
                and ts.strftime("%Y%m%d") == today_str
                and "bottom" in self.vehicle_directions.get(tid, "").lower()
            ]
            
            # Build plate summary
            plate_summary = {}
            for track_id in vehicles_today:
                plate_text = self.vehicle_plates.get(track_id, "?")
                if plate_text not in plate_summary:
                    plate_summary[plate_text] = 0
                plate_summary[plate_text] += 1
            
            # Merge similar plates (differ by 1-2 characters)
            plate_summary = self._merge_similar_plates(plate_summary)
            
            # Sort by count (descending)
            sorted_plates = sorted(
                plate_summary.items(),
                key=lambda x: (-x[1], x[0])
            )
            
            return sorted_plates
        except Exception as e:
            self.log(f"[ERROR] Error getting today's vehicles summary: {e}")
            return []


