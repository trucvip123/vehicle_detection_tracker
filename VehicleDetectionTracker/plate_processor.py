"""License plate detection and processing utilities."""

from datetime import datetime
import threading
import cv2
import json
import os
from pathlib import Path
from VehicleDetectionTracker.plate_utils import detect_license_plate_sync
from VehicleDetectionTracker.utils.send_bot import send_notify_to_telegram


# Global set to track vehicles that have already sent Telegram notifications
_vehicle_telegram_sent = set()
_vehicle_telegram_sent_lock = threading.Lock()


def reset_telegram_sent():
    """Reset the telegram sent tracking set."""
    global _vehicle_telegram_sent
    _vehicle_telegram_sent.clear()


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
        self.state_file = "vehicle_state.json"
        self._state_lock = threading.Lock()

        # Load persisted state if it exists
        self._load_state()

    def save_daily_vehicle_summary(self, date_str=None):
        """
        Gửi thông báo Telegram tổng hợp số lượng xe đi vào trong ngày, không xuất file CSV.
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
        # Only count vehicles with direction_label indicating entry (e.g., 'IN')
        vehicles_today = [
            tid
            for tid, ts in self.vehicle_last_seen.items()
            if hasattr(ts, "strftime")
            and ts.strftime("%Y%m%d") == date_str
            and "top" in self.vehicle_directions.get(tid, "").lower()
        ]
        self.log(f"[DEBUG] vehicles_today: {vehicles_today}")
        try:
            msg = f"Tổng hợp xe vào ngày {date_str}: {len(vehicles_today)} xe vào khu vực mỏ."
            send_warning_to_telegram(msg)
            self.log(f"Telegram notification sent for daily summary: {msg}")
        except Exception as e:
            self.log(f"Failed to send Telegram summary notification: {e}")

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
            # Use sync version for simplicity in streaming mode
            license_plate_info = detect_license_plate_sync(
                self.plate_model,
                vehicle_frame,
                self.ocr_reader,
                self._model_lock,
                timestamp_str,
                vehicle_dir=vehicle_dir,
            )
            plate_text = license_plate_info.get("text") if license_plate_info else None
            self.log(f"[PLATE] vehicle_id={track_id} detected_plate={plate_text}")
            if plate_text and plate_text != "unknown":
                # Update most recent plate for display
                self.vehicle_plates[track_id] = plate_text

                # Increment detection count for this plate
                if track_id not in self.vehicle_plate_counts:
                    self.vehicle_plate_counts[track_id] = {}
                self.vehicle_plate_counts[track_id][plate_text] = (
                    self.vehicle_plate_counts[track_id].get(plate_text, 0) + 1
                )

                # Get direction if available
                if direction_label:
                    self.vehicle_directions[track_id] = direction_label
                    self.log(
                        f"[PLATE] vehicle_id={track_id} direction_set={direction_label}"
                    )
                elif track_id not in self.vehicle_directions:
                    self.vehicle_directions[track_id] = "Unknown"

                # Update last seen timestamp
                if frame_timestamp:
                    self.vehicle_last_seen[track_id] = frame_timestamp
                    self.log(
                        f"[PLATE] vehicle_id={track_id} last_seen_updated={frame_timestamp}"
                    )
                    # Save state after updating last_seen
                    self._save_state()

                # Send Telegram notification only once per vehicle ID (thread-safe)
                global _vehicle_telegram_sent, _vehicle_telegram_sent_lock
                with _vehicle_telegram_sent_lock:
                    if track_id not in _vehicle_telegram_sent:
                        filename = f"{vehicle_dir}/vehicle_{plate_text}.png"
                        cv2.imwrite(filename, vehicle_frame)
                        self.log(
                            f"[PLATE] vehicle_id={track_id} Sending Telegram notification"
                        )
                        send_notify_to_telegram(
                            plate_text,
                            direction_label,
                            frame_timestamp,
                            image_path=filename,
                        )
                        _vehicle_telegram_sent.add(track_id)
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

    def _load_state(self):
        """Load vehicle state from persisted JSON file."""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)

                # Restore vehicle_last_seen with datetime objects
                if "vehicle_last_seen" in state:
                    for track_id, timestamp_str in state["vehicle_last_seen"].items():
                        try:
                            # Convert string back to datetime
                            self.vehicle_last_seen[int(track_id)] = (
                                datetime.fromisoformat(timestamp_str)
                            )
                        except (ValueError, TypeError):
                            pass

                # Restore vehicle_directions
                if "vehicle_directions" in state:
                    for track_id, direction in state["vehicle_directions"].items():
                        self.vehicle_directions[int(track_id)] = direction

                self.log(
                    f"[PERSIST] Loaded state: {len(self.vehicle_last_seen)} vehicles from {self.state_file}"
                )
        except Exception as e:
            self.log(f"[PERSIST] Failed to load state: {e}")

    def _save_state(self):
        """Save vehicle state to JSON file for persistence."""
        try:
            with self._state_lock:
                state = {
                    "vehicle_last_seen": {
                        str(track_id): (
                            ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
                        )
                        for track_id, ts in self.vehicle_last_seen.items()
                    },
                    "vehicle_directions": self.vehicle_directions.copy(),
                    "timestamp": datetime.now().isoformat(),
                }

                with open(self.state_file, "w", encoding="utf-8") as f:
                    json.dump(state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.log(f"[PERSIST] Failed to save state: {e}")
