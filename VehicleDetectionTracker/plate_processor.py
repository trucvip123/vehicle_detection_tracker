"""License plate detection and processing utilities."""

import threading
import cv2
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

    def save_daily_vehicle_summary(self, date_str=None):
        """
        Gửi thông báo Telegram tổng hợp số lượng xe đi vào trong ngày, không xuất file CSV.
        Args:
            date_str (str): Date in YYYYMMDD format. If None, use today.
        """
        from datetime import datetime
        from VehicleDetectionTracker.utils.send_bot import send_notify_to_telegram
        if date_str is None:
            date_str = datetime.now().strftime("%Y%m%d")
        # Only count vehicles with direction_label indicating entry (e.g., 'IN')
        vehicles_today = [tid for tid, ts in self.vehicle_last_seen.items()
                 if ts.strftime("%Y%m%d") == date_str and "top" in self.vehicle_directions.get(tid, '').lower()]
        try:
            msg = f"Tổng hợp xe vào ngày {date_str}: {len(vehicles_today)} xe vào khu vực mỏ."
            send_notify_to_telegram(msg)
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
        self, track_id, vehicle_frame, direction_label=None, timestamp=None
    ):
        """
        Sync wrapper for background plate processing using ThreadPoolExecutor.
        Tracks license plate detection counts.

        Args:
            track_id: Vehicle track ID
            vehicle_frame: Cropped vehicle frame
            direction_label: Vehicle direction label (if available)
            timestamp: Detection timestamp
        """
        self.log(f"Processing plate background for vehicle {track_id}")
        try:
            # Use sync version for simplicity in streaming mode
            license_plate_info = detect_license_plate_sync(
                self.plate_model,
                vehicle_frame,
                self.ocr_reader,
                self._model_lock,
                timestamp,
                track_id=track_id
            )
            plate_text = license_plate_info.get("text") if license_plate_info else None
            self.log(f"Vehicle {track_id} detected plate: {plate_text}")
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
                elif track_id not in self.vehicle_directions:
                    self.vehicle_directions[track_id] = "Unknown"

                # Update last seen timestamp
                if timestamp:
                    self.vehicle_last_seen[track_id] = timestamp

                # Send Telegram notification only once per vehicle ID (thread-safe)
                global _vehicle_telegram_sent, _vehicle_telegram_sent_lock
                with _vehicle_telegram_sent_lock:
                    if track_id not in _vehicle_telegram_sent:
                        filename = f"screenshots/vehicle_{plate_text}.png"
                        cv2.imwrite(filename, vehicle_frame)
                        self.log(
                            f"Sending Telegram notification for vehicle {track_id}..."
                        )
                        send_notify_to_telegram(
                            plate_text, direction_label, timestamp, image_path=filename
                        )
                        _vehicle_telegram_sent.add(track_id)
        except Exception as e:
            self.log(f"Background plate detection error for vehicle {track_id}: {e}")

    def submit_plate_processing(
        self, track_id, vehicle_frame, direction_label, timestamp_str
    ):
        """Submit plate processing to background executor."""
        if vehicle_frame.size > 0:
            self.executor.submit(
                self.process_plate_background_sync,
                track_id,
                vehicle_frame.copy(),
                direction_label,
                timestamp_str,
            )
