"""License plate detection and processing utilities."""

from datetime import datetime
import threading
import cv2
import json
import os
from pathlib import Path
from concurrent.futures import wait, ALL_COMPLETED
from VehicleDetectionTracker.plate_utils import detect_license_plate_sync
from VehicleDetectionTracker.utils.send_bot import send_notify_to_telegram
from VehicleDetectionTracker.logging_utils import log_plate
from VehicleDetectionTracker.vehicle_summary import (
    save_daily_vehicle_summary,
    levenshtein_distance,
    merge_similar_plates,
)


# Global sets to track vehicle notifications
_vehicle_telegram_sent_with_plate = set()  # Vehicles that sent notification with plate
_vehicle_telegram_sent_without_plate = set()  # Vehicles that sent notification without plate
_vehicle_telegram_sent_lock = threading.RLock()  # RLock to allow reentrant locking


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
        self.vehicle_plate_counts = {}  # {track_id: {plate_text: count}} # Count of distinct vehicles with each plate (for summary, not frame count)
        self.vehicle_plate_counts_each_frame = {}  # {track_id: {plate_text: detection_count}} # Count of how many times each plate is detected (incremented on each detection)
        self.vehicle_directions = {}  # {track_id: direction_label}
        self.vehicle_last_seen = {}  # {track_id: timestamp}
        self._vehicles_without_plate_logged = set()  # Track which vehicles we've already logged as missing plate
        self.vehicle_pending_futures = {}  # {track_id: [future1, future2, ...]} - track background tasks
        self.vehicle_detected_plate_images = {}  # {track_id: path_to_image_with_detected_plate}
        self.vehicle_pending_task_count = {}  # {track_id: count of pending background tasks}
        self._task_count_lock = threading.Lock()  # Lock for task count updates

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

        save_daily_vehicle_summary(
            self.vehicle_last_seen,
            self.vehicle_directions,
            self.vehicle_plates,
            self.log,
            send_warning_to_telegram,
            date_str,
            self.vehicle_plate_counts,
        )

    def _get_best_vehicle_image_by_plate(self, vehicle_dir, plate_text, track_id):
        """
        Get the best vehicle image that matches the detected plate.
        Priority:
        1. Image with detected_plate_XX (actual detected plate image)
        2. Image matching plate name (vehicle_PLATE.png)
        3. Largest image in directory
        
        Args:
            vehicle_dir: Path to vehicle screenshots directory
            plate_text: License plate text (e.g., '77C-166.34')
            track_id: Vehicle track ID for logging
            
        Returns:
            str: Path to best matching image file or None if not found
        """
        try:
            if not os.path.exists(vehicle_dir):
                self.log(f"[IMAGE_SELECT] vehicle_id={track_id} directory not found: {vehicle_dir}")
                return None
            
            # Strategy 1: Try to find detected_plate_XX.png (highest priority - actual detected image)
            # These files have the plate info in their name
            detected_plate_images = [
                f for f in os.listdir(vehicle_dir)
                if f.startswith('detected_plate_') and f.endswith('.png')
            ]
            
            if detected_plate_images:
                # Sort by number in filename (e.g., detected_plate_05_77C-166.34.png)
                # Pick the one with highest detection count
                detected_plate_images.sort(
                    key=lambda x: int(x.split('_')[2]) if len(x.split('_')) > 2 and x.split('_')[2].isdigit() else 0,
                    reverse=True
                )
                
                # Prefer high-confidence images with detection count >= 2
                # Multiple detections confirm the plate accuracy, avoid single-detection OCR errors
                high_confidence_images = [
                    img for img in detected_plate_images
                    if int(img.split('_')[2]) >= 2
                ]
                
                best_detected = high_confidence_images[0] if high_confidence_images else detected_plate_images[0]
                image_path = os.path.join(vehicle_dir, best_detected)
                if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
                    detection_count = int(best_detected.split('_')[2]) if len(best_detected.split('_')) > 2 and best_detected.split('_')[2].isdigit() else 0
                    confidence = "HIGH (count >= 2)" if detection_count >= 2 else "LOW (single detection)"
                    self.log(f"[IMAGE_SELECT] vehicle_id={track_id} Using detected plate image: {best_detected} [{confidence}]")
                    return image_path
            
            # Strategy 2: Try to find image with matching plate name
            sanitized_plate = self._sanitize_filename(plate_text)
            target_filename = f"vehicle_{sanitized_plate}.png"
            
            target_path = os.path.join(vehicle_dir, target_filename)
            if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
                self.log(f"[IMAGE_SELECT] vehicle_id={track_id} Found image matching plate '{plate_text}': {target_filename}")
                return target_path
            
            # Strategy 3: Pick largest image (best quality/coverage)
            image_sizes = []
            for img_file in os.listdir(vehicle_dir):
                if img_file.endswith('.png'):
                    try:
                        img_path = os.path.join(vehicle_dir, img_file)
                        size = os.path.getsize(img_path)
                        if size > 0:  # Only consider non-empty files
                            image_sizes.append((img_file, size))
                    except:
                        pass
            
            if image_sizes:
                # Sort by size descending and pick largest
                image_sizes.sort(key=lambda x: x[1], reverse=True)
                best_image = image_sizes[0][0]
                image_path = os.path.join(vehicle_dir, best_image)
                self.log(f"[IMAGE_SELECT] vehicle_id={track_id} Using largest image by size: {best_image} ({image_sizes[0][1]} bytes)")
                return image_path
            
            self.log(f"[IMAGE_SELECT] vehicle_id={track_id} No valid images found in {vehicle_dir}")
            return None
            
        except Exception as e:
            self.log(f"[IMAGE_SELECT] vehicle_id={track_id} error finding image: {e}")
            return None

    def _get_first_vehicle_image(self, vehicle_dir, track_id):
        """
        Get the first image file from the vehicle directory.
        Used as fallback when no plates detected.
        
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
                    # Sort to get the largest image (best quality)
                    image_sizes = []
                    for img_file in image_files:
                        try:
                            img_path = os.path.join(vehicle_dir, img_file)
                            size = os.path.getsize(img_path)
                            image_sizes.append((img_file, size))
                        except:
                            pass
                    
                    if image_sizes:
                        # Pick largest image
                        image_sizes.sort(key=lambda x: x[1], reverse=True)
                        best_image = image_sizes[0][0]
                        image_path = os.path.join(vehicle_dir, best_image)
                        self.log(f"[PLATE] vehicle_id={track_id} found largest image: {best_image}")
                        return image_path
                else:
                    self.log(f"[PLATE] vehicle_id={track_id} no images found in {vehicle_dir}")
                    return None
            else:
                self.log(f"[PLATE] vehicle_id={track_id} directory not found: {vehicle_dir}")
                return None
        except Exception as e:
            self.log(f"[PLATE] vehicle_id={track_id} error finding image: {e}")
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
            track_id not in self.vehicle_plate_counts_each_frame
            or not self.vehicle_plate_counts_each_frame[track_id]
        ):
            return None, 0

        plate_counts = self.vehicle_plate_counts_each_frame[track_id]
        # Find plate with maximum count
        most_detected_plate = max(reversed(plate_counts.items()), key=lambda x: x[1])  # In case of tie, get the most recently added plate
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
        # Check if this vehicle_id has already sent notification with plate
        # If so, skip processing to avoid duplicate notifications
        global _vehicle_telegram_sent_with_plate, _vehicle_telegram_sent_without_plate, _vehicle_telegram_sent_lock
        
        with _vehicle_telegram_sent_lock:
            if track_id in _vehicle_telegram_sent_with_plate:
                self.log(f"[PLATE] vehicle_id={track_id} Already sent notification with plate, skipping background processing")
                return
        
        self.log(f"[PLATE] vehicle_id={track_id} Processing plate background")
        try:
            self.log(f"[PLATE] vehicle_id={track_id} ► Starting background plate detection")
            # EARLY CHECK: Skip if vehicle already notified (to avoid stale background tasks)
            with _vehicle_telegram_sent_lock:
                if track_id in _vehicle_telegram_sent_with_plate or track_id in _vehicle_telegram_sent_without_plate:
                    self.log(f"[PLATE] vehicle_id={track_id} Vehicle already has notification sent, skipping background processing (stale task)")
                    return
            
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
            self.log(f"[PLATE] vehicle_id={track_id} Calling detect_license_plate_sync...")
            license_plate_info = detect_license_plate_sync(
                self.plate_model,
                vehicle_frame,
                self.ocr_reader,
                self._model_lock,
                timestamp_str,
                vehicle_dir=vehicle_dir,
                track_id=track_id,
            )
            plate_text = license_plate_info.get("text") if license_plate_info else None
            count_detections = license_plate_info.get("count") if license_plate_info else 0
            # Ensure count_detections is always an integer (never None)
            if count_detections is None:
                count_detections = 0
            self.log(f"[PLATE] vehicle_id={track_id} Detection result: plate={plate_text}, count={count_detections}")
            log_plate(track_id, f"detected_plate={plate_text}")
            
            # Only send notifications for vehicles entering (not exiting)
            # Must contain "bottom" to be confirmed as entering
            # Skip if direction contains "top" (exit/ra khỏi) or is "Unknown"
            is_entering = direction_label and "bottom" in direction_label.lower()
            
            self.log(f"[PLATE] vehicle_id={track_id} ► Direction check: direction_label={direction_label}, is_entering={is_entering}")
            
            if not is_entering:
                self.log(f"[PLATE] vehicle_id={track_id} ✗ Skipping - vehicle is exiting or direction unknown (direction={direction_label})")
                log_plate(track_id, f"Skipping notification - vehicle is exiting or direction unknown (direction={direction_label})")
                return
            
            # Track effective_track_id for use across both cases
            effective_track_id = track_id  # Default: use original track_id
            
            # CHANGED: Only accumulate plate data, don't send notification immediately
            # Notifications will be sent when vehicle disappears (in frame_processor.py)
            with _vehicle_telegram_sent_lock:
                # SECOND CHECK: Skip if already notified (double-check before accumulating)
                if track_id in _vehicle_telegram_sent_with_plate or track_id in _vehicle_telegram_sent_without_plate:
                    self.log(f"[PLATE] vehicle_id={track_id} Vehicle notification sent while detecting, skipping accumulation")
                    return
                
                # Only accumulate plate detections (no immediate notification)
                if plate_text and plate_text != "unknown":
                    self.log(f"[PLATE] vehicle_id={track_id} ► Processing valid plate detection: {plate_text}")
                    # Check if this is first detection for this track_id
                    is_first_detection = track_id not in self.vehicle_plate_counts_each_frame
                    
                    # Track effective_track_id for plate versioning
                    effective_track_id = track_id
                    
                    # If track_id has plates already, check if it's reused (different vehicle)
                    if not is_first_detection:
                        existing_plates = self.vehicle_plate_counts_each_frame[track_id]
                        all_existing_plates = list(existing_plates.keys())
                        
                        # Check if new plate is significantly different (not just OCR variation)
                        is_different = True
                        for existing_plate in all_existing_plates:
                            dist = levenshtein_distance(plate_text, existing_plate)
                            if dist <= 2:  # Similar plate (OCR variation)
                                is_different = False
                                break
                        
                        # Check time gap: only consider it a different vehicle if time > 60 seconds
                        time_gap_exceeded = False
                        if is_different and track_id in self.vehicle_last_seen:
                            last_seen_time = self.vehicle_last_seen[track_id]
                            if frame_timestamp and hasattr(last_seen_time, 'timestamp'):
                                time_gap = frame_timestamp.timestamp() - last_seen_time.timestamp()
                                time_gap_exceeded = time_gap > 60  # More than 60 seconds
                                log_plate(track_id, f"Time gap check: {time_gap:.1f}s (threshold: 60s), gap_exceeded={time_gap_exceeded}")
                        
                        # If completely different plate AND time gap > 60s, track_id was reused
                        # Create a versioned ID to keep both vehicles' data separate
                        if is_different and time_gap_exceeded and len(all_existing_plates) > 0:
                            # Find next version number
                            version = 2
                            while f"{track_id}_v{version}" in self.vehicle_plate_counts_each_frame:
                                version += 1
                            
                            effective_track_id = f"{track_id}_v{version}"
                            log_plate(track_id, f"Track ID reused! Creating version ID: {effective_track_id}")
                            log_plate(track_id, f"Keeping old data: {all_existing_plates}")
                            log_plate(effective_track_id, f"Storing new vehicle data: {plate_text}")
                            is_first_detection = True  # Treat versioned ID as first detection
                            
                            # Update vehicle_last_seen and vehicle_directions with versioned ID
                            if frame_timestamp:
                                self.vehicle_last_seen[effective_track_id] = frame_timestamp
                            if direction_label:
                                self.vehicle_directions[effective_track_id] = direction_label
                            log_plate(effective_track_id, f"Updated vehicle_last_seen and vehicle_directions with versioned ID")
                    
                    # Track this vehicle-plate pair with detection count
                    # vehicle_plate_counts_each_frame represents: how many times this plate was detected for this vehicle
                    if effective_track_id not in self.vehicle_plate_counts_each_frame:
                        self.vehicle_plate_counts_each_frame[effective_track_id] = {}
                    
                    # Initialize to 0 if not already set, then increment
                    # Each detection increments the count by 1
                    if plate_text not in self.vehicle_plate_counts_each_frame[effective_track_id]:
                        self.vehicle_plate_counts_each_frame[effective_track_id][plate_text] = 0
                    self.vehicle_plate_counts_each_frame[effective_track_id][plate_text] += 1
                    
                    # NOTE: Do NOT use vehicle_plate_counts for intermediate decisions
                    # vehicle_plate_counts is only for end-of-day summary
                    # Use vehicle_plates for notifications and tracking
                    
                    # Just accumulate the plate detection for summary purposes
                    plate_counts = self.vehicle_plate_counts_each_frame[effective_track_id]
                    log_plate(effective_track_id, f"Accumulated plate detection: plate={plate_text}, plate_counts={plate_counts}")
                    
                    # Set best plate in vehicle_plates if not already set
                    # First valid plate detected becomes the primary plate for this vehicle
                    if effective_track_id not in self.vehicle_plates or not self.vehicle_plates[effective_track_id]:
                        self.vehicle_plates[effective_track_id] = plate_text
                        self.log(f"[PLATE] vehicle_id={effective_track_id} Set primary plate: {plate_text}")
                        
                        # Also track in vehicle_plate_counts for summary (1 vehicle = 1 count)
                        if effective_track_id not in self.vehicle_plate_counts:
                            self.vehicle_plate_counts[effective_track_id] = {}
                        if plate_text not in self.vehicle_plate_counts[effective_track_id]:
                            self.vehicle_plate_counts[effective_track_id][plate_text] = 1
                            self.log(f"[PLATE] vehicle_id={effective_track_id} Added to plate_counts: {plate_text}")
                    
                    # Save vehicle frame for later use in final notification
                    # Store detected frame with plate info in filename
                    os.makedirs(vehicle_dir, exist_ok=True)
                    sanitized_plate = self._sanitize_filename(plate_text)
                    
                    # Save as detected_plate_XX.png to track images that actually contain detected plates
                    # Use count of this specific plate as the number in filename
                    detection_count = plate_counts.get(plate_text, 0)
                    detected_image_name = f"detected_plate_{detection_count:02d}_{sanitized_plate}.png"
                    detected_image_path = os.path.join(vehicle_dir, detected_image_name)
                    cv2.imwrite(detected_image_path, vehicle_frame)
                    
                    # Track this image as having a successfully detected plate
                    self.vehicle_detected_plate_images[effective_track_id] = detected_image_path
                    
                    log_plate(effective_track_id, f"Saved detected plate image: {detected_image_name}")
                    self.log(f"[PLATE] vehicle_id={effective_track_id} ✓ Accumulated plate detection: {plate_text}, images saved")
                    
                    # vehicle_plate_counts is only used for end-of-day summary calculations
                    # Notification will be sent using vehicle_plates when all background tasks complete (via _on_plate_task_complete callback)
        except Exception as e:
            self.log(f"[PLATE] vehicle_id={track_id} ✗ Detection error: {e}")
            import traceback
            self.log(f"[PLATE] vehicle_id={track_id} Traceback: {traceback.format_exc()}")

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
        # Check if already notified before submitting to avoid unnecessary processing
        global _vehicle_telegram_sent_with_plate, _vehicle_telegram_sent_lock
        
        with _vehicle_telegram_sent_lock:
            if track_id in _vehicle_telegram_sent_with_plate:
                self.log(f"[PLATE] vehicle_id={track_id} Already sent notification, skipping submission to executor")
                return
        
        if vehicle_frame.size > 0:
            self.log(f"[PLATE] vehicle_id={track_id} Submitting to background executor")
            future = self.executor.submit(
                self.process_plate_background_sync,
                track_id,
                vehicle_frame.copy(),
                direction_label,
                frame_timestamp,
                timestamp_str,
                vehicle_dir=vehicle_dir,
            )
            # Track the pending future
            if track_id not in self.vehicle_pending_futures:
                self.vehicle_pending_futures[track_id] = []
            self.vehicle_pending_futures[track_id].append(future)
            
            # Track pending task count
            with self._task_count_lock:
                if track_id not in self.vehicle_pending_task_count:
                    self.vehicle_pending_task_count[track_id] = 0
                self.vehicle_pending_task_count[track_id] += 1
            
            # Add callback to be called when task completes
            future.add_done_callback(lambda f: self._on_plate_task_complete(track_id, vehicle_dir))

    def _on_plate_task_complete(self, track_id, vehicle_dir):
        """
        Callback called when a background plate detection task completes.
        Decrements task count when all tasks for this vehicle are done.
        Sends notification immediately when all background tasks complete.
        
        Note: vehicle_plates is already set during process_plate_background_sync.
        vehicle_plate_counts is only used for end-of-day summary, not for intermediate decisions.
        
        Args:
            track_id: Vehicle track ID
            vehicle_dir: Directory for vehicle screenshots
        """
        try:
            with self._task_count_lock:
                if track_id in self.vehicle_pending_task_count:
                    self.vehicle_pending_task_count[track_id] -= 1
                    remaining_tasks = self.vehicle_pending_task_count[track_id]
                    
                    self.log(f"[TASK_COMPLETE] vehicle_id={track_id} Background task completed. Remaining tasks: {remaining_tasks}")
                    
                    # If all tasks for this vehicle are done
                    if remaining_tasks == 0:
                        self.log(f"[TASK_COMPLETE] vehicle_id={track_id} ✓ All background tasks completed")
                        
                        # Best plate is already set from process_plate_background_sync
                        current_best_plate = self.get_most_detected_plate(track_id)[0]
                        self.log(f"[TASK_COMPLETE] vehicle_id={track_id} Best plate: {current_best_plate}")
                        
                        # Note: State will be saved only after Telegram API succeeds
                        self.log(f"[TASK_COMPLETE] vehicle_id={track_id} ✓ All detection tasks done, ready to send notification now")
            
            # Send notification OUTSIDE the lock to avoid blocking other operations
            # Only send if all tasks are complete (remaining_tasks == 0)
            with self._task_count_lock:
                remaining_tasks = self.vehicle_pending_task_count.get(track_id, 0)
            
            if remaining_tasks == 0:
                self.log(f"[TASK_COMPLETE] vehicle_id={track_id} ► Sending notification now (all tasks complete)...")
                result = self.send_final_vehicle_notification(track_id, vehicle_dir=vehicle_dir)
                if result:
                    self.log(f"[TASK_COMPLETE] vehicle_id={track_id} ✓ Notification sent successfully")
                else:
                    self.log(f"[TASK_COMPLETE] vehicle_id={track_id} ⚠ Notification sending failed or skipped")
                
        except Exception as e:
            self.log(f"[TASK_COMPLETE] vehicle_id={track_id} ERROR in _on_plate_task_complete: {e}")
            import traceback
            self.log(f"[TASK_COMPLETE] Traceback: {traceback.format_exc()}")
        finally:
            # Clean up pending task count for this vehicle
            with self._task_count_lock:
                if track_id in self.vehicle_pending_task_count and self.vehicle_pending_task_count[track_id] == 0:
                    self.vehicle_pending_task_count.pop(track_id, None)
                    self.log(f"[TASK_COMPLETE] vehicle_id={track_id} Cleaned up pending_task_count")

    def wait_pending_tasks_for_vehicle(self, track_id, timeout=30):
        """
        Wait for all pending background tasks (futures) for a vehicle to complete.
        
        Args:
            track_id: Vehicle track ID
            timeout: Maximum time to wait in seconds (default 30s)
            
        Returns:
            True if all tasks completed, False if timeout
        """
        if track_id not in self.vehicle_pending_futures:
            return True
        
        futures = self.vehicle_pending_futures[track_id]
        if not futures:
            return True
        
        self.log(f"[WAIT_TASKS] vehicle_id={track_id} Waiting for {len(futures)} pending background tasks (timeout={timeout}s)...")
        
        try:
            # Wait for all futures to complete
            done, not_done = wait(futures, timeout=timeout, return_when=ALL_COMPLETED)
            
            if not_done:
                self.log(f"[WAIT_TASKS] vehicle_id={track_id} ⚠ {len(not_done)}/{len(futures)} tasks did NOT complete within {timeout}s")
                return False
            else:
                self.log(f"[WAIT_TASKS] vehicle_id={track_id} ✓ All {len(futures)} pending tasks completed")
                
                # Clear futures after completion
                self.vehicle_pending_futures.pop(track_id, None)
                return True
                
        except Exception as e:
            self.log(f"[WAIT_TASKS] vehicle_id={track_id} Error waiting for tasks: {e}")
            return False

    def wait_all_background_tasks(self, timeout=60):
        """
        Wait for ALL pending background tasks for all vehicles to complete.
        This is called during cleanup to ensure all notifications are sent.
        
        Args:
            timeout: Maximum time to wait in seconds (default 60s)
            
        Returns:
            bool: True if all tasks completed, False if timeout occurred
        """
        import time
        
        start_time = time.time()
        
        # Get list of tracked vehicles with pending tasks
        with self._task_count_lock:
            pending_vehicles = list(self.vehicle_pending_task_count.keys())
            pending_count = len(pending_vehicles)
        
        if pending_count == 0:
            self.log(f"[WAIT_ALL_TASKS] No pending background tasks")
            return True
        
        self.log(f"[WAIT_ALL_TASKS] Waiting for {pending_count} vehicles with pending tasks (timeout={timeout}s)...")
        
        # Wait for all vehicles
        for track_id in pending_vehicles:
            elapsed = time.time() - start_time
            remaining_timeout = timeout - elapsed
            
            if remaining_timeout <= 0:
                self.log(f"[WAIT_ALL_TASKS] ⚠ Timeout reached before waiting for all vehicles")
                return False
            
            with self._task_count_lock:
                remaining = self.vehicle_pending_task_count.get(track_id, 0)
            
            if remaining > 0:
                self.log(f"[WAIT_ALL_TASKS] Waiting for vehicle_id={track_id} ({remaining} pending tasks, {remaining_timeout:.1f}s remaining)...")
                self.wait_pending_tasks_for_vehicle(track_id, timeout=remaining_timeout)
        
        # Final check - wait a bit for callbacks to complete (includes the daemon thread delay)
        time.sleep(0.5)
        
        # Verify all are done
        with self._task_count_lock:
            final_pending = list(self.vehicle_pending_task_count.keys())
        
        if final_pending:
            self.log(f"[WAIT_ALL_TASKS] ⚠ Some vehicles still have pending tasks: {final_pending}")
            return False
        else:
            self.log(f"[WAIT_ALL_TASKS] ✓ All background tasks completed successfully")
            return True

    def send_notifications_for_completed_vehicles(self):
        """
        Send notifications for any remaining vehicles that haven't been notified yet.
        This is called during cleanup when video ends (fallback for vehicles with pending tasks).
        Most notifications should already be sent via _on_plate_task_complete callbacks.
        """
        import time
        global _vehicle_telegram_sent_with_plate, _vehicle_telegram_sent_without_plate, _vehicle_telegram_sent_lock
        
        self.log(f"[CLEANUP_NOTIFY] Starting cleanup notifications (fallback for remaining vehicles)...")
        
        # Get all vehicles that have ever been tracked
        all_vehicle_ids = set(self.vehicle_plates.keys()) | set(self.vehicle_directions.keys()) | set(self.vehicle_last_seen.keys())
        
        # First pass: check for vehicles with pending tasks
        vehicles_with_pending = []
        for track_id in all_vehicle_ids:
            with self._task_count_lock:
                remaining_tasks = self.vehicle_pending_task_count.get(track_id, 0)
            if remaining_tasks > 0:
                vehicles_with_pending.append((track_id, remaining_tasks))
        
        # If some tasks are pending, wait a bit more for them to complete
        if vehicles_with_pending:
            self.log(f"[CLEANUP_NOTIFY] Found {len(vehicles_with_pending)} vehicles with pending tasks: {[v[0] for v in vehicles_with_pending]}")
            self.log(f"[CLEANUP_NOTIFY] Waiting additional 15s for pending tasks to complete...")
            
            start_wait = time.time()
            max_wait_time = 15
            last_check_time = start_wait
            
            while time.time() - start_wait < max_wait_time:
                still_pending = []
                current_time = time.time()
                
                for track_id, _ in vehicles_with_pending:
                    with self._task_count_lock:
                        remaining = self.vehicle_pending_task_count.get(track_id, 0)
                    if remaining > 0:
                        still_pending.append(track_id)
                
                if not still_pending:
                    self.log(f"[CLEANUP_NOTIFY] ✓ All pending tasks completed during wait")
                    break
                
                # Log progress every 5 seconds
                if current_time - last_check_time >= 5:
                    elapsed = current_time - start_wait
                    remaining_time = max_wait_time - elapsed
                    self.log(f"[CLEANUP_NOTIFY] Still waiting... {elapsed:.1f}/{max_wait_time}s, pending vehicles: {still_pending}")
                    last_check_time = current_time
                
                time.sleep(0.5)
            else:
                # Timeout reached
                if still_pending:
                    self.log(f"[CLEANUP_NOTIFY] ⚠ {len(still_pending)} vehicles STILL have pending tasks after {max_wait_time}s: {still_pending}")
        
        # Second pass: send notifications for any remaining vehicles (not yet notified)
        not_notified_count = 0
        for track_id in all_vehicle_ids:
            # Check if already sent
            with _vehicle_telegram_sent_lock:
                already_sent = (track_id in _vehicle_telegram_sent_with_plate or 
                              track_id in _vehicle_telegram_sent_without_plate)
            
            if not already_sent:
                not_notified_count += 1
                with self._task_count_lock:
                    remaining_tasks = self.vehicle_pending_task_count.get(track_id, 0)
                
                # Check if vehicle has plate data (from completed detection tasks)
                has_plate_data = track_id in self.vehicle_plates and self.vehicle_plates[track_id]
                
                if remaining_tasks == 0:
                    status = "(all tasks completed via callback)"
                elif has_plate_data:
                    status = f"({remaining_tasks} tasks still pending but plate detected, sending with available data)"
                else:
                    status = f"({remaining_tasks} tasks still pending, plate data incomplete)"
                
                self.log(f"[CLEANUP_NOTIFY] vehicle_id={track_id} Sending cleanup notification {status}")
                
                # Sync vehicle_plates from vehicle_plates (already should be set if tasks completed)
                # Only use vehicle_plates for notification - vehicle_plate_counts is only for end-of-day summary
                if track_id not in self.vehicle_plates or not self.vehicle_plates[track_id]:
                    self.log(f"[CLEANUP_NOTIFY] vehicle_id={track_id} Warning: vehicle_plates not set, no plate to send in notification")
                
                # Find vehicle_dir from vehicle_last_seen
                vehicle_dir = None
                # Try to find the directory pattern: screenshots/YYYYMMDD/HHMM_track_id
                import glob
                if track_id in self.vehicle_last_seen:
                    date_str = self.vehicle_last_seen[track_id].strftime("%Y%m%d")
                    pattern = f"screenshots/{date_str}/*_{track_id}"
                    matching_dirs = glob.glob(pattern)
                    if matching_dirs:
                        vehicle_dir = matching_dirs[0]
                
                result = self.send_final_vehicle_notification(track_id, vehicle_dir=vehicle_dir)
                if result:
                    self.log(f"[CLEANUP_NOTIFY] vehicle_id={track_id} ✓ Cleanup notification sent")
                else:
                    self.log(f"[CLEANUP_NOTIFY] vehicle_id={track_id} ⚠ Cleanup notification skipped")
            else:
                self.log(f"[CLEANUP_NOTIFY] vehicle_id={track_id} Already notified, skipping")
        
        if not_notified_count == 0:
            self.log(f"[CLEANUP_NOTIFY] ✓ No remaining vehicles to notify (all already sent via callbacks)")
        else:
            self.log(f"[CLEANUP_NOTIFY] ✓ Cleanup notification complete ({not_notified_count} vehicles processed)")

    def send_final_vehicle_notification(self, track_id, vehicle_dir=None):
        """
        Send final notification for a vehicle after all detections are aggregated.
        This is called when vehicle disappears (missing_frames exceeds threshold).
        
        Uses vehicle_plates which is already set during plate detection.
        Note: vehicle_plate_counts is only used for end-of-day summary, not for notification decisions.
        
        Args:
            track_id: Vehicle track ID
            vehicle_dir: Directory containing vehicle screenshots
            
        Returns:
            bool: True if notification was sent, False otherwise
        """
        global _vehicle_telegram_sent_with_plate, _vehicle_telegram_sent_without_plate, _vehicle_telegram_sent_lock
        
        # Skip if already sent
        with _vehicle_telegram_sent_lock:
            if track_id in _vehicle_telegram_sent_with_plate or track_id in _vehicle_telegram_sent_without_plate:
                self.log(f"[FINAL_NOTIFY] vehicle_id={track_id} Already sent notification, skipping final notification")
                return False
        
        try:
            direction_label = self.vehicle_directions.get(track_id, "Unknown")
            frame_timestamp = self.vehicle_last_seen.get(track_id)
            
            # Check if direction is entering (must contain "bottom")
            is_entering = direction_label and "bottom" in direction_label.lower()
            if not is_entering:
                self.log(f"[FINAL_NOTIFY] vehicle_id={track_id} Skipping final notification - vehicle is exiting or direction unknown (direction={direction_label})")
                return False
            
            plate_text = self.get_most_detected_plate(track_id)[0]  # Get the best plate detected for this vehicle
            notification_sent = False
            
            # Case 1: Plate detected
            if plate_text and plate_text != "unknown":
                self.log(f"[FINAL_NOTIFY] vehicle_id={track_id} Sending final notification WITH PLATE: plate={plate_text}")
                
                # Find best image matching the detected plate
                image_path = None
                if vehicle_dir:
                    image_path = self._get_best_vehicle_image_by_plate(vehicle_dir, plate_text, track_id)
                
                # Send final notification with best plate (OUTSIDE of lock)
                self.log(f"[FINAL_NOTIFY] vehicle_id={track_id} ► Calling Telegram API...")
                try:
                    telegram_response = send_notify_to_telegram(
                        plate_text,
                        direction_label,
                        frame_timestamp,
                        image_path=image_path,
                    )
                    self.log(f"[FINAL_NOTIFY] vehicle_id={track_id} ✓ Telegram API call completed (success={telegram_response})")
                except Exception as e:
                    self.log(f"[FINAL_NOTIFY] vehicle_id={track_id} ✗ Telegram API error: {e}")
                    telegram_response = {"ok": False, "error": str(e)}
                
                # Only mark as sent if Telegram API succeeded (check 'ok' field)
                telegram_success = telegram_response.get("ok", False)
                if telegram_success:
                    # Now update tracking status inside lock
                    with _vehicle_telegram_sent_lock:
                        _vehicle_telegram_sent_with_plate.add(track_id)
                    
                    self.log(f"[FINAL_NOTIFY] vehicle_id={track_id} ✓ Final notification sent with plate={plate_text}")
                    notification_sent = True
                else:
                    self.log(f"[FINAL_NOTIFY] vehicle_id={track_id} ⚠ Telegram API failed, will retry later")
                    notification_sent = False
            
            if notification_sent:
                self.log(f"[FINAL_NOTIFY] vehicle_id={track_id} Saving state (Telegram API was successful)...")
                self._save_state()
                self.log(f"[FINAL_NOTIFY] vehicle_id={track_id} ✓ State saved")
                return True
            
            return False
                    
        except Exception as e:
            self.log(f"[FINAL_NOTIFY] vehicle_id={track_id} Error sending final notification: {e}")
            import traceback
            self.log(f"[FINAL_NOTIFY] vehicle_id={track_id} Traceback: {traceback.format_exc()}")
            return False

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

                # Load vehicle_plates (primary data source for valid track_ids)
                # Note: vehicle_directions and vehicle_last_seen are NOT persisted, only used in-memory during session
                valid_track_ids = set()
                if "vehicle_plates" in state:
                    for track_id_str, plate_text in state["vehicle_plates"].items():
                        track_id = int(track_id_str)
                        self.vehicle_plates[track_id] = plate_text
                        valid_track_ids.add(track_id)
                    self.log(f"[PERSIST] Restored vehicle_plates: {len(valid_track_ids)} vehicles")

                # Restore vehicle_plate_counts
                if "vehicle_plate_counts" in state:
                    for track_id_str, plate_counts in state["vehicle_plate_counts"].items():
                        track_id = int(track_id_str)
                        if track_id in valid_track_ids:
                            self.vehicle_plate_counts[track_id] = plate_counts

                # Restore notification sent status
                global _vehicle_telegram_sent_with_plate, _vehicle_telegram_sent_without_plate, _vehicle_telegram_sent_lock
                if "sent_with_plate" in state:
                    for track_id_str in state["sent_with_plate"]:
                        track_id = int(track_id_str)
                        if track_id in valid_track_ids:
                            with _vehicle_telegram_sent_lock:
                                _vehicle_telegram_sent_with_plate.add(track_id)
                                self.log(f"[PERSIST] Restored: vehicle_id={track_id} already sent with plate")
                
                if "sent_without_plate" in state:
                    for track_id_str in state["sent_without_plate"]:
                        track_id = int(track_id_str)
                        if track_id in valid_track_ids:
                            with _vehicle_telegram_sent_lock:
                                _vehicle_telegram_sent_without_plate.add(track_id)
                                self.log(f"[PERSIST] Restored: vehicle_id={track_id} already sent without plate")

                self.log(
                    f"[PERSIST] ✓ Loaded state: {len(valid_track_ids)} vehicles from {state_file}"
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
            
            # Clear all daily tracking data (fresh start for new day)
            self.vehicle_plates.clear()
            self.vehicle_plate_counts.clear()
            self.vehicle_plate_counts_each_frame.clear()
            self.vehicle_directions.clear()
            self.vehicle_last_seen.clear()
            self.vehicle_detected_plate_images.clear()
            self.vehicle_pending_futures.clear()
            self.vehicle_pending_task_count.clear()
            self._vehicles_without_plate_logged.clear()
            
            reset_daily_tracking()  # Reset telegram notification tracking
            
            self.log(f"[DAILY_RESET] ✓ All daily tracking data reset for new day")

    def _save_state(self):
        """Save vehicle state to JSON file for persistence (one file per day)."""
        try:
            self.log(f"[PERSIST] _save_state: Attempting to acquire lock...")
            with self._state_lock:
                self.log(f"[PERSIST] _save_state: Lock acquired")
                state_file = self._get_state_file_path()
                self.log(f"[PERSIST] _save_state: state_file={state_file}")
                
                # Convert all data to JSON-serializable format
                # No need to filter - data already contains only today's vehicles (reset at day start)
                self.log(f"[PERSIST] _save_state: Converting vehicle_plates...")
                today_plates = {
                    str(tid): plate
                    for tid, plate in self.vehicle_plates.items()
                }
                self.log(f"[PERSIST] _save_state: Converted plates: {len(today_plates)}")
                
                self.log(f"[PERSIST] _save_state: Converting vehicle_plate_counts...")
                today_plate_counts = {
                    str(tid): counts
                    for tid, counts in self.vehicle_plate_counts.items()
                }
                self.log(f"[PERSIST] _save_state: Converted plate_counts: {len(today_plate_counts)}")
                
                # Persist notification sent status
                self.log(f"[PERSIST] _save_state: Processing telegram sent status...")
                global _vehicle_telegram_sent_with_plate, _vehicle_telegram_sent_without_plate
                sent_with_plate_list = []
                sent_without_plate_list = []
                with _vehicle_telegram_sent_lock:
                    self.log(f"[PERSIST] _save_state: Telegram lock acquired")
                    for track_id in _vehicle_telegram_sent_with_plate:
                        sent_with_plate_list.append(str(track_id))
                    for track_id in _vehicle_telegram_sent_without_plate:
                        sent_without_plate_list.append(str(track_id))
                    self.log(f"[PERSIST] _save_state: Telegram lock released")
                
                self.log(f"[PERSIST] _save_state: Building state dict...")
                state = {
                    "vehicle_plates": today_plates,
                    "vehicle_plate_counts": today_plate_counts,
                    "sent_with_plate": sent_with_plate_list,
                    "sent_without_plate": sent_without_plate_list,
                    "timestamp": datetime.now().isoformat(),
                }
                self.log(f"[PERSIST] _save_state: State dict built, size={len(str(state))} chars")

                self.log(f"[PERSIST] _save_state: Writing to file {state_file}...")
                with open(state_file, "w", encoding="utf-8") as f:
                    json.dump(state, f, indent=2, ensure_ascii=False)
                self.log(f"[PERSIST] _save_state: ✓ File written successfully")
            self.log(f"[PERSIST] _save_state: Lock released")
        except Exception as e:
            self.log(f"[PERSIST] Failed to save state: {e}")
            import traceback
            self.log(f"[PERSIST] Traceback: {traceback.format_exc()}")

    def get_today_vehicles_summary(self):
        """
        Get summary of vehicles (plates and counts) from today's tracking data.
        
        Note: All data in memory is from today only (reset at day start).
        Returns:
            list: List of tuples (plate_text, count) sorted by count (descending)
        """
        try:
            # Get vehicles that entered (direction contains "bottom")
            # No need to filter by date - all data is from today
            vehicles_today = [
                tid
                for tid in self.vehicle_last_seen.keys()
                if "bottom" in self.vehicle_directions.get(tid, "").lower()
            ]
            
            # self.log(f"[SUMMARY] vehicles_today (entering/bottom): {sorted(vehicles_today)}")
            # self.log(f"[SUMMARY] vehicle_plates keys: {sorted(self.vehicle_plates.keys())}")
            # self.log(f"[SUMMARY] vehicle_plate_counts keys: {sorted(self.vehicle_plate_counts.keys())}")
            
            # Build plate summary using detection counts
            plate_summary = {}
            for track_id in vehicles_today:
                plate_text = self.vehicle_plates.get(track_id, "?")
                if plate_text == "?":
                    # Only log once per vehicle to avoid spam in logs
                    if track_id not in self._vehicles_without_plate_logged:
                        self.log(f"[SUMMARY] vehicle_id={track_id} missing from vehicle_plates (plate detection not complete yet)")
                        self._vehicles_without_plate_logged.add(track_id)
                    continue
                    
                if plate_text not in plate_summary:
                    plate_summary[plate_text] = 0
                # Get detection count for this vehicle (default to 1 if not found)
                if track_id in self.vehicle_plate_counts and plate_text in self.vehicle_plate_counts[track_id]:
                    detection_count = self.vehicle_plate_counts[track_id][plate_text]
                else:
                    detection_count = 1
                    
                # self.log(f"[SUMMARY] vehicle_id={track_id} plate={plate_text} count={detection_count}")
                plate_summary[plate_text] += detection_count
            
            # self.log(f"[SUMMARY] plate_summary before merge: {plate_summary}")
            
            # Merge similar plates (differ by 1-2 characters)
            plate_summary = merge_similar_plates(plate_summary, self.log)
            
            # self.log(f"[SUMMARY] plate_summary after merge: {plate_summary}")
            
            # Sort by count (descending)
            sorted_plates = sorted(
                plate_summary.items(),
                key=lambda x: (-x[1], x[0])
            )
            
            # self.log(f"[SUMMARY] Final sorted_plates: {sorted_plates}")
            
            return sorted_plates
        except Exception as e:
            self.log(f"[ERROR] Error getting today's vehicles summary: {e}")
            import traceback
            # self.log(f"[ERROR] Traceback: {traceback.format_exc()}")
            return []


