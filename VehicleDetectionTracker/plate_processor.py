"""License plate detection and processing utilities."""

from datetime import datetime
import threading
import cv2
import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from typing import Dict, Set, Optional, Tuple, Callable, Any, List
from difflib import SequenceMatcher

from VehicleDetectionTracker.plate_utils import (
    detect_license_plate_sync,
    submit_plate_detection_async,
    initialize_inference_queue,
    batch_detect_license_plates,
    initialize_batch_accumulator,
    get_batch_accumulator,
)
from VehicleDetectionTracker.utils.send_bot import send_notify_to_telegram
from VehicleDetectionTracker.logging_utils import log_plate
from VehicleDetectionTracker.vehicle_summary import (
    save_daily_vehicle_summary,
    levenshtein_distance,
    merge_similar_plates,
)
from VehicleDetectionTracker.metrics import get_metrics_collector


# Global sets to track vehicle notifications using UUIDs (not track_ids)
_vehicle_telegram_sent_with_plate = set()  # Vehicle UUIDs that sent notification with plate
_vehicle_telegram_sent_without_plate = set()  # Vehicle UUIDs that sent notification without plate
_vehicle_telegram_sent_lock = threading.RLock()  # RLock to allow reentrant locking

# UUID Mapping: Maps detector track_id (int) → system UUID (str)
_track_id_to_uuid = {}  # {track_id: uuid_string}
_uuid_mapping_lock = threading.RLock()  # Lock for UUID mapping updates


def reset_telegram_sent() -> None:
    """Reset the telegram sent tracking sets."""
    global _vehicle_telegram_sent_with_plate, _vehicle_telegram_sent_without_plate
    _vehicle_telegram_sent_with_plate.clear()
    _vehicle_telegram_sent_without_plate.clear()


def reset_daily_tracking() -> None:
    """Reset daily tracking data (to be called at midnight or start of new day)."""
    global _track_id_to_uuid, _uuid_mapping_lock
    reset_telegram_sent()
    # Reset UUID mapping only at new day (not on system restart)
    with _uuid_mapping_lock:
        _track_id_to_uuid.clear()


class PlateProcessor:
    """Handles license plate detection and tracking."""

    def __init__(
        self, 
        plate_model: Any, 
        ocr_reader: Any, 
        executor: ThreadPoolExecutor, 
        log_func: Callable[[str], None]
    ) -> None:
        self.plate_model = plate_model
        self.ocr_reader = ocr_reader
        self.executor = executor
        self.log = log_func
        self.metrics = get_metrics_collector()  # Get metrics collector instance
        
        # Load detection config for batch processing
        from VehicleDetectionTracker.config_loader import get_plate_detection_config
        self.detection_config = get_plate_detection_config()

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
        
        # Track pending inference queue tasks separately (for async queue detection)
        self.vehicle_pending_queue_tasks = {}  # {track_id: count of pending queue tasks}

        self._model_lock = threading.Lock()

        # Persistence settings
        self.state_dir = "vehicle_state"  # Directory to store state files
        # ===== THREAD SAFETY ARCHITECTURE =====
        # _state_lock (RLock): Protects ALL vehicle state dicts during read/write
        #   - vehicle_plates
        #   - vehicle_plate_counts
        #   - vehicle_plate_counts_each_frame
        #   - vehicle_directions
        #   - vehicle_last_seen
        #   - vehicle_detected_plate_images
        #   Use RLock to allow reentrant locking (same thread can acquire multiple times)
        self._state_lock = threading.RLock()
        self._last_reset_date = datetime.now().strftime("%Y%m%d")  # Initialize to today to prevent immediate reset
        
        # Create state directory if it doesn't exist
        os.makedirs(self.state_dir, exist_ok=True)

        # Initialize inference queue for async plate detection (reduces model_lock contention)
        initialize_inference_queue(num_workers=6)
        self.log("[PLATE_PROCESSOR] ✓ Initialized inference queue with 2 worker threads")

        # Initialize batch accumulator for async batch OCR (Approach 2)
        from VehicleDetectionTracker.config_loader import get_batch_inference_config
        batch_config = get_batch_inference_config()
        self.batch_enabled = batch_config.get("enabled", False)
        self.batch_accumulator = None
        if self.batch_enabled:
            self.batch_accumulator = initialize_batch_accumulator(
                batch_size=batch_config.get("batch_size", 8),
                time_threshold_ms=batch_config.get("time_threshold_ms", 500)
            )
            self.log(f"[PLATE_PROCESSOR] ✓ Initialized batch accumulator (batch_size={batch_config.get('batch_size')}, "
                    f"time_threshold={batch_config.get('time_threshold_ms')}ms)")
            
            # Start batch processor thread
            self.batch_processor_thread = threading.Thread(
                target=self._batch_processor_loop,
                daemon=True,
                name="BatchProcessor"
            )
            self.batch_processor_thread.start()
            self.batch_processor_running = True
        else:
            self.log("[PLATE_PROCESSOR] Batch accumulator disabled (using traditional per-vehicle processing)")
        
        # Initialize UUID tracking for restart resilience
        self._track_id_to_uuid_local = {}  # Temporary instance reference to global UUID mapping
        
        # Load persisted state if it exists
        self._load_state()

    # ===== UUID MAPPING METHODS (Restart Resilience) =====
    def get_or_create_uuid(self, track_id: int) -> str:
        """
        Get or create a UUID for a given track_id.
        This ensures each vehicle (even after detector restart) gets a unique UUID.
        
        Args:
            track_id: Detector's track ID (int)
            
        Returns:
            str: UUID string (unique per vehicle, persistent across restarts)
        """
        global _track_id_to_uuid, _uuid_mapping_lock
        
        with _uuid_mapping_lock:
            if track_id not in _track_id_to_uuid:
                # New vehicle detected - assign a UUID
                vehicle_uuid = str(uuid.uuid4())
                _track_id_to_uuid[track_id] = vehicle_uuid
                self.log(f"[UUID] New vehicle: track_id={track_id} → UUID={vehicle_uuid[:8]}")
                return vehicle_uuid
            else:
                # Vehicle already has a UUID
                return _track_id_to_uuid[track_id]

    # ===== THREAD-SAFE STATE ACCESS METHODS =====
    def update_vehicle_state(
        self, 
        track_id: int, 
        plate_text: Optional[str] = None, 
        direction: Optional[str] = None, 
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Thread-safe update of vehicle state (plate, direction, last_seen).
        All updates use _state_lock to prevent race conditions.
        
        Args:
            track_id: Vehicle track ID
            plate_text: License plate text (optional)
            direction: Direction label (optional)
            timestamp: Last seen timestamp (optional)
        """
        with self._state_lock:
            if plate_text is not None:
                self.vehicle_plates[track_id] = plate_text
            if direction is not None:
                self.vehicle_directions[track_id] = direction
            if timestamp is not None:
                self.vehicle_last_seen[track_id] = timestamp

    def get_vehicle_state(self, track_id: int) -> Tuple[Optional[str], Optional[str], Optional[datetime]]:
        """
        Thread-safe read of vehicle state.
        Returns: (plate_text, direction, last_seen_timestamp) or None for missing track_id
        """
        with self._state_lock:
            plate = self.vehicle_plates.get(track_id)
            direction = self.vehicle_directions.get(track_id)
            timestamp = self.vehicle_last_seen.get(track_id)
        return plate, direction, timestamp

    def get_all_vehicle_ids(self) -> Set[int]:
        """Thread-safe access to all tracked vehicle IDs."""
        with self._state_lock:
            return set(self.vehicle_plates.keys()) | set(self.vehicle_directions.keys()) | set(self.vehicle_last_seen.keys())

    def get_vehicle_plates_copy(self) -> Dict[int, str]:
        """Thread-safe copy of vehicle_plates dict for reading."""
        with self._state_lock:
            return self.vehicle_plates.copy()

    def get_vehicle_directions_copy(self) -> Dict[int, str]:
        """Thread-safe copy of vehicle_directions dict for reading."""
        with self._state_lock:
            return self.vehicle_directions.copy()

    def get_vehicle_last_seen_copy(self) -> Dict[int, datetime]:
        """Thread-safe copy of vehicle_last_seen dict for reading."""
        with self._state_lock:
            return self.vehicle_last_seen.copy()

    def save_daily_vehicle_summary(self, date_str: Optional[str] = None) -> None:
        """
        Gửi thông báo Telegram tổng hợp số lượng xe đi vào trong ngày với chi tiết biển số.
        Args:
            date_str (str): Date in YYYYMMDD format. If None, use today.
        """
        from VehicleDetectionTracker.utils.send_bot import send_warning_to_telegram

        # Use thread-safe copy methods to avoid holding lock during summary generation
        vehicle_plates_copy = self.get_vehicle_plates_copy()
        vehicle_directions_copy = self.get_vehicle_directions_copy()
        vehicle_last_seen_copy = self.get_vehicle_last_seen_copy()
        
        with self._state_lock:
            vehicle_plate_counts_copy = self.vehicle_plate_counts.copy()

        save_daily_vehicle_summary(
            vehicle_last_seen_copy,
            vehicle_directions_copy,
            vehicle_plates_copy,
            self.log,
            send_warning_to_telegram,
            date_str,
            vehicle_plate_counts_copy,
        )

    def _get_best_vehicle_image_by_plate(
        self, 
        vehicle_dir: str, 
        plate_text: str, 
        track_id: int
    ) -> Optional[str]:
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
                self.log(f"[IMAGE_SELECT] vehicle_id={track_id} ❌ directory NOT FOUND: {vehicle_dir}")
                return None
            
            # List all files in the directory for debugging
            all_files = os.listdir(vehicle_dir)
            png_files = [f for f in all_files if f.endswith('.png')]
            self.log(f"[IMAGE_SELECT] vehicle_id={track_id} Found {len(png_files)} PNG files in {vehicle_dir}: {png_files[:10]}")  # Show first 10
            
            # Strategy 1: Try to find detected_plate_XX.png (highest priority - actual detected image)
            # These files have the plate info in their name
            detected_plate_images = [
                f for f in all_files
                if f.startswith('detected_plate_') and f.endswith('.png')
            ]
            
            if detected_plate_images:
                self.log(f"[IMAGE_SELECT] vehicle_id={track_id} Found {len(detected_plate_images)} detected_plate images: {detected_plate_images}")
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
                    file_size = os.path.getsize(image_path)
                    self.log(f"[IMAGE_SELECT] vehicle_id={track_id} ✓ Selected detected_plate: {best_detected} (size={file_size}, {confidence})")
                    return image_path
                else:
                    self.log(f"[IMAGE_SELECT] vehicle_id={track_id} ⚠ Detected_plate file empty or not readable: {image_path}")
            else:
                self.log(f"[IMAGE_SELECT] vehicle_id={track_id} ℹ No detected_plate_XX images found (looking for pattern: detected_plate_*.png)")
            
            # Strategy 2: Try to find image with matching plate name
            sanitized_plate = self._sanitize_filename(plate_text)
            target_filename = f"vehicle_{sanitized_plate}.png"
            target_path = os.path.join(vehicle_dir, target_filename)
            
            self.log(f"[IMAGE_SELECT] vehicle_id={track_id} Strategy 2: Looking for plate-matching image: {target_filename}")
            if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
                file_size = os.path.getsize(target_path)
                self.log(f"[IMAGE_SELECT] vehicle_id={track_id} ✓ Found: {target_filename} (size={file_size})")
                return target_path
            else:
                self.log(f"[IMAGE_SELECT] vehicle_id={track_id} ℹ Not found: {target_filename}")
            
            # Strategy 3: Pick largest image (best quality/coverage)
            self.log(f"[IMAGE_SELECT] vehicle_id={track_id} Strategy 3: Looking for largest image in directory")
            image_sizes = []
            for img_file in all_files:
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
                file_size = image_sizes[0][1]
                self.log(f"[IMAGE_SELECT] vehicle_id={track_id} ✓ Found largest image: {best_image} (size={file_size})")
                return image_path
            
            self.log(f"[IMAGE_SELECT] vehicle_id={track_id} ❌ No valid images found in {vehicle_dir} (checked {len(png_files)} PNG files)")
            return None
            
        except Exception as e:
            self.log(f"[IMAGE_SELECT] vehicle_id={track_id} error finding image: {e}")
            return None

    def _get_first_vehicle_image(self, vehicle_dir: str, track_id: int) -> Optional[str]:
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

    @staticmethod
    def normalize_plate(plate: str) -> str:
        return plate.replace("-", "").replace(".", "").upper()

    @staticmethod
    def similarity(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    def get_most_detected_plate(self, track_id: int) -> Tuple[str, int]:
        """
        Get the license plate with highest detection count for a vehicle.
        If tie → choose plate most similar to others (consensus).
        
        Special handling for 2 candidates:
        If one candidate matches the confirmed plate in vehicle_plates, select that one.

        Args:
            track_id: Vehicle track ID

        Returns:
            tuple: (plate_text, count) or (None, 0)
        """
        if (
            track_id not in self.vehicle_plate_counts_each_frame
            or not self.vehicle_plate_counts_each_frame[track_id]
        ):
            return None, 0

        plate_counts = self.vehicle_plate_counts_each_frame[track_id]

        # 🔹 Step 1: tìm max count
        max_count = max(plate_counts.values())

        # 🔹 Step 2: lấy các plate có cùng max_count
        candidates = [p for p, c in plate_counts.items() if c == max_count]

        # Nếu chỉ có 1 thì return luôn
        if len(candidates) == 1:
            return candidates[0], max_count

        # 🔹 Step 2.5: Xử lý đặc biệt cho 2 candidates
        # Nếu 1 trong 2 biển số đã được xác nhận trong vehicle_plates, chọn nó
        if len(candidates) == 2:
            current_confirmed_plate = self.vehicle_plates.get(track_id)
            if current_confirmed_plate and current_confirmed_plate in candidates:
                self.log(f"[PLATE_SELECT] vehicle_id={track_id} ✓ Tie-break (2 candidates): Found confirmed plate in candidates: {current_confirmed_plate} (count={max_count})")
                return current_confirmed_plate, max_count
        
        # 🔹 Step 3: tie-break bằng similarity
        norm_map = {p: self.normalize_plate(p) for p in candidates}
    
        scores = {}
        for p1 in candidates:
            total_score = 0
            for p2 in candidates:
                if p1 == p2:
                    continue
                total_score += self.similarity(norm_map[p1], norm_map[p2])
            scores[p1] = total_score

        # 🔹 Step 4: chọn plate có similarity cao nhất
        best_plate = max(scores, key=scores.get)

        return best_plate, max_count

    def _sanitize_filename(self, filename: str) -> str:
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
        track_id: int,
        vehicle_frame: Any,
        direction_label: Optional[str] = None,
        frame_timestamp: Optional[datetime] = None,
        timestamp_str: Optional[str] = None,
        vehicle_dir: str = "screenshots",
    ) -> None:
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
        # Get or create UUID for this track_id (ensures uniqueness across restarts)
        vehicle_uuid = self.get_or_create_uuid(track_id)
        
        # Check if this vehicle (by UUID) has already sent notification
        # If so, skip processing to avoid duplicate notifications
        global _vehicle_telegram_sent_with_plate, _vehicle_telegram_sent_without_plate, _vehicle_telegram_sent_lock
        
        with _vehicle_telegram_sent_lock:
            if vehicle_uuid in _vehicle_telegram_sent_with_plate:
                self.log(f"[PLATE] vehicle_id={track_id} (uuid={vehicle_uuid[:8]}) Already sent notification with plate, skipping background processing")
                return
        
        self.log(f"[PLATE] vehicle_id={track_id} (uuid={vehicle_uuid[:8]}) Processing plate background")
        try:
            self.log(f"[PLATE] vehicle_id={track_id} (uuid={vehicle_uuid[:8]}) ► Starting background plate detection")
            # EARLY CHECK: Skip if vehicle already notified (to avoid stale background tasks)
            with _vehicle_telegram_sent_lock:
                if vehicle_uuid in _vehicle_telegram_sent_with_plate or vehicle_uuid in _vehicle_telegram_sent_without_plate:
                    self.log(f"[PLATE] vehicle_id={track_id} (uuid={vehicle_uuid[:8]}) Vehicle already has notification sent, skipping background processing (stale task)")
                    return
            
            # === TIME BLOCK: STATE UPDATE ===
            # with time_block(f"[STATE_UPDATE] vehicle_id={track_id}", self.log):
            # Update last_seen timestamp and direction (thread-safe)
            with self._state_lock:
                if frame_timestamp:
                    self.vehicle_last_seen[track_id] = frame_timestamp
                
                # Always update direction if available (not just when plate is detected)
                if direction_label:
                    self.vehicle_directions[track_id] = direction_label
                    log_plate(track_id, f"direction_set={direction_label}")
                elif track_id not in self.vehicle_directions:
                    self.vehicle_directions[track_id] = "Unknown"

            # === TIME BLOCK: PLATE DETECTION SUBMISSION ===
            # with time_block(f"[SUBMIT_PLATE] vehicle_id={track_id}", self.log):
            # Submit plate detection to inference queue asynchronously
            # This avoids model_lock bottleneck by queuing tasks
            self.log(f"[PLATE] vehicle_id={track_id} (uuid={vehicle_uuid[:8]}) Submitting plate detection to inference queue...")
            
            # Increment pending queue task count BEFORE submitting
            with self._task_count_lock:
                if track_id not in self.vehicle_pending_queue_tasks:
                    self.vehicle_pending_queue_tasks[track_id] = 0
                self.vehicle_pending_queue_tasks[track_id] += 1
                self.log(f"[PLATE] vehicle_id={track_id} Pending queue tasks: {self.vehicle_pending_queue_tasks[track_id]}")
            
            def _plate_detection_callback(license_plate_info):
                """Callback invoked when plate detection completes."""
                self._process_plate_result(
                    track_id,
                    license_plate_info,
                    direction_label,
                    frame_timestamp,
                    vehicle_frame,
                    vehicle_dir,
                )
            
            submit_plate_detection_async(
                self.plate_model,
                vehicle_frame,
                self.ocr_reader,
                self._model_lock,
                timestamp_str,
                callback=_plate_detection_callback,
                vehicle_dir=vehicle_dir,
                track_id=track_id,
            )
        
        except Exception as e:
            self.log(f"[PLATE] vehicle_id={track_id} ✗ Detection error: {e}")
            import traceback
            self.log(f"[PLATE] vehicle_id={track_id} Traceback: {traceback.format_exc()}")

    def _process_plate_result(
        self,
        track_id,
        license_plate_info,
        direction_label,
        frame_timestamp,
        vehicle_frame,
        vehicle_dir,
    ):
        """
        Process the result of plate detection asynchronously.
        This is called by the inference queue when detection completes.
        
        Args:
            track_id: Vehicle track ID
            license_plate_info: Result dict from detect_license_plate_sync
            direction_label: Vehicle direction label
            frame_timestamp: Actual detection datetime object
            vehicle_frame: The original vehicle frame
            vehicle_dir: Directory for saving images
        """
        try:
            import time as time_module
            plate_detection_start = time_module.time()
            
            self.log(f"[PLATE_RESULT] vehicle_id={track_id} Processing plate result (ENTRY)")
            
            # === TIME BLOCK: PARSE DETECTION RESULT ===
            # with time_block(f"[PARSE_RESULT] vehicle_id={track_id}", self.log):
            plate_text = license_plate_info.get("text") if license_plate_info else None
            count_detections = license_plate_info.get("count") if license_plate_info else 0
            confidence = license_plate_info.get("confidence", 0.0) if license_plate_info else 0.0
            # Ensure count_detections is always an integer (never None)
            if count_detections is None:
                count_detections = 0
            self.log(f"[PLATE_RESULT] vehicle_id={track_id} Detection result: plate={plate_text}, count={count_detections}, confidence={confidence:.3f}")
            log_plate(track_id, f"detected_plate={plate_text}")
            
            # === TIME BLOCK: DIRECTION CHECK ===
            # with time_block(f"[DIRECTION_CHECK] vehicle_id={track_id}", self.log):
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
            # === TIME BLOCK: ACCUMULATE PLATE DATA ===
            # with time_block(f"[ACCUMULATE] vehicle_id={track_id}", self.log):
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
                    
                    # Record plate detection in metrics
                    plate_detection_time = time_module.time() - plate_detection_start
                    self.metrics.record_plate_detection(
                        confidence=float(confidence),
                        processing_time=plate_detection_time,
                        ocr_attempted=True,
                        ocr_success=(plate_text is not None and plate_text != "unknown")
                    )
                    
                    # Just accumulate the plate detection for summary purposes
                    plate_counts = self.vehicle_plate_counts_each_frame[effective_track_id]
                    log_plate(effective_track_id, f"Accumulated plate detection: plate={plate_text}, plate_counts={plate_counts}")
                    
                    # Save vehicle frame for later use in final notification
                    # Store detected frame with plate info in filename
                    os.makedirs(vehicle_dir, exist_ok=True)
                    sanitized_plate = self._sanitize_filename(plate_text)
                    
                    # Save as detected_plate_XX.png to track images that actually contain detected plates
                    # Use count of this specific plate as the number in filename
                    detection_count = plate_counts.get(plate_text, 0)
                    detected_image_name = f"detected_plate_{detection_count:02d}_{sanitized_plate}.png"
                    detected_image_path = os.path.join(vehicle_dir, detected_image_name)
                    
                    # Save image and verify it was written
                    cv2.imwrite(detected_image_path, vehicle_frame)
                    if os.path.exists(detected_image_path) and os.path.getsize(detected_image_path) > 0:
                        file_size = os.path.getsize(detected_image_path)
                        self.log(f"[PLATE_IMAGE] vehicle_id={effective_track_id} ✓ Image saved: {detected_image_path} ({file_size} bytes)")
                    else:
                        self.log(f"[PLATE_IMAGE] vehicle_id={effective_track_id} ⚠ Image save FAILED or empty: {detected_image_path}")
                    
                    # Track this image as having a successfully detected plate
                    # Store by BOTH track_id and effective_track_id to ensure we can find it later
                    self.vehicle_detected_plate_images[effective_track_id] = detected_image_path
                    if track_id != effective_track_id:
                        self.vehicle_detected_plate_images[track_id] = detected_image_path
                        self.log(f"[PLATE_IMAGE] vehicle_id={track_id} Stored image path for versioned ID")
                    
                    log_plate(effective_track_id, f"Saved detected plate image: {detected_image_name}")
                    self.log(f"[PLATE] vehicle_id={effective_track_id} ✓ Accumulated plate detection: {plate_text}, images saved")
                    # vehicle_plate_counts is only used for end-of-day summary calculations
                    # Notification will be sent using vehicle_plates when all background tasks complete (via _on_plate_task_complete callback)
        except Exception as e:
            self.log(f"[PLATE] vehicle_id={track_id} ✗ Error processing plate result: {e}")
            import traceback
            self.log(f"[PLATE] vehicle_id={track_id} Traceback: {traceback.format_exc()}")
        finally:
            # Decrement queue task counter when callback completes
            with self._task_count_lock:
                if track_id in self.vehicle_pending_queue_tasks:
                    self.vehicle_pending_queue_tasks[track_id] -= 1
                    pending_queue_tasks = self.vehicle_pending_queue_tasks[track_id]
                    pending_executor_tasks = self.vehicle_pending_task_count.get(track_id, 0)
                    total_pending = pending_queue_tasks + pending_executor_tasks
                    
                    self.log(f"[PLATE_QUEUE] vehicle_id={track_id} Queue callback done. Queue: {pending_queue_tasks}, Executor: {pending_executor_tasks}, Total: {total_pending}")
                    
                    # If ALL tasks (executor + queue) are done, send notification from queue callback
                    # This way ensures we don't send twice (once from executor, once from queue)
                    if total_pending == 0:
                        self.log(f"[PLATE_QUEUE] vehicle_id={track_id} ✓✓ ALL tasks complete (queue + executor). Sending final notification...")
                        vehicle_dir = f"screenshots/{track_id}"
                        
                        # Send notification OUTSIDE the lock to avoid blocking
                        try:
                            result = self.send_final_vehicle_notification(track_id, vehicle_dir=vehicle_dir)
                            if result:
                                self.log(f"[PLATE_QUEUE] vehicle_id={track_id} ✓ Notification sent successfully")
                            else:
                                self.log(f"[PLATE_QUEUE] vehicle_id={track_id} ⚠ Notification sending failed or skipped")
                        except Exception as ex:
                            self.log(f"[PLATE_QUEUE] vehicle_id={track_id} ❌ Error sending notification: {ex}")
                        
                        # Clean up when completely done
                        self.vehicle_pending_queue_tasks.pop(track_id, None)
                        self.vehicle_pending_task_count.pop(track_id, None)
                        self.log(f"[PLATE_QUEUE] vehicle_id={track_id} Cleaned up all pending task counts")



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

    def submit_plate_processing_batch(self, frame_vehicles: Dict[int, dict]) -> None:
        """
        Submit batch of vehicle detections to accumulator for batch OCR processing (Approach 2).
        
        This method:
        1. Adds each vehicle detection to the batch accumulator
        2. Triggers batch processing when size or time threshold is reached
        3. Manages the batch processor thread
        
        Args:
            frame_vehicles: Dict of {track_id: {frame, direction, timestamp, timestamp_str, vehicle_dir}}
        """
        if not self.batch_enabled or not self.batch_accumulator:
            # Fall back to per-vehicle processing
            self.log(f"[BATCH_SUBMIT] Batch accumulator disabled, falling back to per-vehicle processing...")
            for track_id, vehicle_data in frame_vehicles.items():
                self.submit_plate_processing(
                    track_id,
                    vehicle_data['frame'],
                    vehicle_data['direction'],
                    vehicle_data['timestamp'],
                    vehicle_data['timestamp_str'],
                    vehicle_data['vehicle_dir']
                )
            return
        
        batch_size_before = len(frame_vehicles)
        self.log(f"[BATCH_SUBMIT] Adding {batch_size_before} vehicles to batch accumulator...")
        
        try:
            # Check if already notified before submitting to avoid unnecessary processing
            global _vehicle_telegram_sent_with_plate, _vehicle_telegram_sent_lock
            
            vehicles_to_add = {}
            with _vehicle_telegram_sent_lock:
                for track_id, vehicle_data in frame_vehicles.items():
                    if track_id not in _vehicle_telegram_sent_with_plate:
                        vehicles_to_add[track_id] = vehicle_data
            
            if not vehicles_to_add:
                self.log(f"[BATCH_SUBMIT] All vehicles already notified, skipping batch submission")
                return
            
            # Add each vehicle to the accumulator
            batch_triggered = False
            for track_id, vehicle_data in vehicles_to_add.items():
                triggered = self.batch_accumulator.add_detection(
                    track_id,
                    vehicle_data['frame'],
                    vehicle_data['timestamp_str'],
                    vehicle_data['vehicle_dir'],
                    vehicle_data['direction'],
                    vehicle_data['timestamp']
                )
                if triggered:
                    batch_triggered = True
            
            # Update pending task count for tracking
            with self._task_count_lock:
                for track_id in vehicles_to_add.keys():
                    if track_id not in self.vehicle_pending_task_count:
                        self.vehicle_pending_task_count[track_id] = 0
                    self.vehicle_pending_task_count[track_id] += 1
            
            if batch_triggered:
                self.log(f"[BATCH_SUBMIT] ✓ Batch trigger reached, processing will be initiated by batch processor thread")
            else:
                self.log(f"[BATCH_SUBMIT] Batch accumulating ({len(vehicles_to_add)} vehicles added)")
        
        except Exception as e:
            self.log(f"[BATCH_SUBMIT] ❌ Error adding to batch accumulator: {e}")
            import traceback
            self.log(traceback.format_exc())

    def _batch_processor_loop(self) -> None:
        """
        Main loop for batch processor thread.
        Continuously monitors batch accumulator and processes batches.
        """
        self.log(f"[BATCH_PROCESSOR] Thread started, monitoring batch accumulator...")
        
        while self.batch_processor_running:
            try:
                batch_stats = self.batch_accumulator.get_batch_stats()
                pending = batch_stats['pending_items']
                elapsed_ms = batch_stats['elapsed_ms']
                
                # Check if batch is ready or time threshold exceeded
                has_items = pending > 0
                time_expired = elapsed_ms > self.batch_accumulator.max_batch_wait_ms
                size_threshold = pending >= self.batch_accumulator.batch_size
                
                if size_threshold:
                    self.log(f"[BATCH_PROCESSOR] Size threshold reached ({pending}/{self.batch_accumulator.batch_size})")
                    batch = self.batch_accumulator.flush()
                    if batch:
                        self._process_batch(batch)
                
                elif time_expired and has_items:
                    self.log(f"[BATCH_PROCESSOR] Time threshold exceeded ({elapsed_ms:.0f}ms/{self.batch_accumulator.max_batch_wait_ms}ms)")
                    batch = self.batch_accumulator.flush()
                    if batch:
                        self._process_batch(batch)
                
                else:
                    # Sleep briefly before checking again
                    threading.Event().wait(0.1)
            
            except Exception as e:
                self.log(f"[BATCH_PROCESSOR] ❌ Error in batch processor loop: {e}")
                import traceback
                self.log(traceback.format_exc())
                threading.Event().wait(0.5)  # Back off on error
    
    def _process_batch(self, frame_vehicles: dict) -> None:
        """
        Process a batch of vehicle detections.
        
        Args:
            frame_vehicles: Dict of {track_id: vehicle_data}
        """
        if not frame_vehicles:
            return
        
        self.log(f"[BATCH_PROCESS] Processing batch of {len(frame_vehicles)} vehicles...")
        
        # Submit batch to executor for processing
        future = self.executor.submit(
            self._execute_batch_ocr,
            frame_vehicles
        )
        
        # Add callback for when batch completes
        future.add_done_callback(lambda f: self._on_batch_complete(frame_vehicles.keys()))
    
    def _execute_batch_ocr(self, frame_vehicles: dict) -> dict:
        """
        Execute batch OCR processing (run in executor thread).
        
        Args:
            frame_vehicles: Dict of {track_id: vehicle_data}
        
        Returns:
            Dict of {track_id: result}
        """
        vehicle_frames_dict = {
            tid: v['frame'] for tid, v in frame_vehicles.items()
        }
        
        try:
            # Run batch detection and OCR
            batch_results = batch_detect_license_plates(
                self.plate_model,
                vehicle_frames_dict,
                self.ocr_reader,
                self._model_lock,
                self.detection_config if hasattr(self, 'detection_config') else {}
            )
            
            # Update vehicle states with results
            for track_id, result in batch_results.items():
                try:
                    self._handle_batch_plate_result(
                        track_id,
                        result,
                        frame_vehicles[track_id]['direction'],
                        frame_vehicles[track_id]['timestamp'],
                        frame_vehicles[track_id]['vehicle_dir']
                    )
                except Exception as handle_err:
                    self.log(f"[BATCH_PROCESS] ❌ Error handling result for track_id={track_id}: {handle_err}")
            
            self.log(f"[BATCH_PROCESS] ✓ Batch OCR complete for {len(batch_results)} vehicles")
            return batch_results
        
        except Exception as e:
            self.log(f"[BATCH_PROCESS] ❌ Error in batch OCR execution: {e}")
            import traceback
            self.log(traceback.format_exc())
            return {}
    
    def _handle_batch_plate_result(self, track_id: int, result: dict, direction: str, 
                                   timestamp: datetime, vehicle_dir: str) -> None:
        """
        Handle a single plate detection result from batch processing.
        
        Args:
            track_id: Vehicle track ID
            result: Detection result dict {text, count, confidence}
            direction: Direction label
            timestamp: Frame timestamp
            vehicle_dir: Vehicle directory
        """
        try:
            plate_text = result.get('text')
            num_detections = result.get('count', 0)
            confidence = result.get('confidence', 0)
            
            self.log(f"[BATCH_RESULT] vehicle_id={track_id} Result: text='{plate_text}', count={num_detections}, conf={confidence:.3f}")
            
            # Update vehicle state
            if plate_text and plate_text != "unknown" and plate_text is not None:
                self.log(f"[BATCH_RESULT] vehicle_id={track_id} ✓ Setting primary plate: '{plate_text}'")
                self.update_vehicle_state(track_id, plate_text=plate_text, direction=direction, timestamp=timestamp)
            elif num_detections > 0:
                # OCR failed but we detected plates - use placeholder
                placeholder_text = f"DETECTED_{num_detections}x"
                self.log(f"[BATCH_RESULT] vehicle_id={track_id} ⚠ OCR failed, using placeholder: '{placeholder_text}'")
                self.update_vehicle_state(track_id, plate_text=placeholder_text, direction=direction, timestamp=timestamp)
        
        except Exception as e:
            self.log(f"[BATCH_RESULT] ❌ Error handling batch result for track_id={track_id}: {e}")
    
    def _on_batch_complete(self, track_ids) -> None:
        """
        Callback when batch processing completes.
        Decrements task counts and sends notifications if ready.
        
        Args:
            track_ids: Iterable of track IDs that were in the batch
        """
        try:
            with self._task_count_lock:
                for track_id in track_ids:
                    if track_id in self.vehicle_pending_task_count:
                        self.vehicle_pending_task_count[track_id] -= 1
                        remaining = self.vehicle_pending_task_count[track_id]
                        
                        if remaining == 0:
                            # All tasks complete, send notification
                            self.log(f"[BATCH_COMPLETE] vehicle_id={track_id} All batch tasks complete, sending notification...")
                            vehicle_dir = f"screenshots/{track_id}"
                            
                            # Send notification OUTSIDE the lock
                            try:
                                result = self.send_final_vehicle_notification(track_id, vehicle_dir=vehicle_dir)
                                if result:
                                    self.log(f"[BATCH_COMPLETE] vehicle_id={track_id} ✓ Notification sent")
                                else:
                                    self.log(f"[BATCH_COMPLETE] vehicle_id={track_id} ⚠ Notification failed")
                            except Exception as notify_err:
                                self.log(f"[BATCH_COMPLETE] vehicle_id={track_id} ❌ Notification error: {notify_err}")
        
        except Exception as e:
            self.log(f"[BATCH_COMPLETE] ❌ Error in batch complete callback: {e}")
            import traceback
            self.log(traceback.format_exc())

    def _on_plate_task_complete(self, track_id: int, vehicle_dir: str) -> None:
        """
        Callback called when a background plate detection task completes (executor task).
        Decrements executor task count when all tasks for this vehicle are done.
        
        NOTE: This is called when executor task completes, but inference queue tasks may still be running.
        We now check BOTH executor + queue tasks before sending notification.
        
        Args:
            track_id: Vehicle track ID
            vehicle_dir: Directory for vehicle screenshots
        """
        try:
            with self._task_count_lock:
                if track_id in self.vehicle_pending_task_count:
                    self.vehicle_pending_task_count[track_id] -= 1
                    remaining_executor_tasks = self.vehicle_pending_task_count[track_id]
                    remaining_queue_tasks = self.vehicle_pending_queue_tasks.get(track_id, 0)
                    total_remaining = remaining_executor_tasks + remaining_queue_tasks
                    
                    self.log(f"[TASK_COMPLETE] vehicle_id={track_id} Executor task completed. Executor: {remaining_executor_tasks}, Queue: {remaining_queue_tasks}, Total: {total_remaining}")
                    
                    # Check if we still have pending queue tasks
                    if remaining_queue_tasks > 0:
                        self.log(f"[TASK_COMPLETE] vehicle_id={track_id} Still pending queue tasks ({remaining_queue_tasks}), waiting for queue callback...")
                        return  # Don't send notification yet - queue callback will handle it
                    
                    # If all tasks for this vehicle are done AND no queue tasks pending
                    if total_remaining == 0:
                        self.log(f"[TASK_COMPLETE] vehicle_id={track_id} ✓ All background tasks completed (executor + queue)")
                        
                        # Best plate is already set from process_plate_background_sync
                        current_best_plate = self.get_most_detected_plate(track_id)[0]
                        self.log(f"[TASK_COMPLETE] vehicle_id={track_id} Best plate: {current_best_plate}")
                        
                        # Note: State will be saved only after Telegram API succeeds
                        self.log(f"[TASK_COMPLETE] vehicle_id={track_id} ✓ All detection tasks done, ready to send notification now")
            
            # Send notification OUTSIDE the lock to avoid blocking other operations
            # Only send if all tasks are complete (total_remaining == 0)
            with self._task_count_lock:
                remaining_executor_tasks = self.vehicle_pending_task_count.get(track_id, 0)
                remaining_queue_tasks = self.vehicle_pending_queue_tasks.get(track_id, 0)
                total_remaining = remaining_executor_tasks + remaining_queue_tasks
            
            if total_remaining == 0:
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
            # Clean up pending task counts for this vehicle if all done
            with self._task_count_lock:
                executor_tasks = self.vehicle_pending_task_count.get(track_id, 0)
                queue_tasks = self.vehicle_pending_queue_tasks.get(track_id, 0)
                if executor_tasks == 0 and queue_tasks == 0:
                    self.vehicle_pending_task_count.pop(track_id, None)
                    self.vehicle_pending_queue_tasks.pop(track_id, None)
                    self.log(f"[TASK_COMPLETE] vehicle_id={track_id} Cleaned up all pending task counts")

    def wait_pending_tasks_for_vehicle(self, track_id: int, timeout: int = 30) -> bool:
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

    def wait_all_background_tasks(self, timeout: int = 60) -> bool:
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
        
        # Get list of tracked vehicles with pending tasks (both executor and queue)
        with self._task_count_lock:
            pending_executor_vehicles = list(self.vehicle_pending_task_count.keys())
            pending_queue_vehicles = list(self.vehicle_pending_queue_tasks.keys())
            all_pending_vehicles = list(set(pending_executor_vehicles + pending_queue_vehicles))
            pending_count = len(all_pending_vehicles)
        
        if pending_count == 0:
            self.log(f"[WAIT_ALL_TASKS] No pending background tasks")
            return True
        
        self.log(f"[WAIT_ALL_TASKS] Waiting for {pending_count} vehicles with pending tasks (timeout={timeout}s)...")
        self.log(f"[WAIT_ALL_TASKS] Executor tasks: {pending_executor_vehicles}, Queue tasks: {pending_queue_vehicles}")
        
        # Wait for all vehicles
        for track_id in all_pending_vehicles:
            elapsed = time.time() - start_time
            remaining_timeout = timeout - elapsed
            
            if remaining_timeout <= 0:
                self.log(f"[WAIT_ALL_TASKS] ⚠ Timeout reached before waiting for all vehicles")
                with self._task_count_lock:
                    lingering = {
                        "executor": {k: self.vehicle_pending_task_count.get(k, 0) for k in all_pending_vehicles},
                        "queue": {k: self.vehicle_pending_queue_tasks.get(k, 0) for k in all_pending_vehicles}
                    }
                self.log(f"[WAIT_ALL_TASKS] Lingering tasks: {lingering}")
                return False
            
            with self._task_count_lock:
                remaining_executor = self.vehicle_pending_task_count.get(track_id, 0)
                remaining_queue = self.vehicle_pending_queue_tasks.get(track_id, 0)
                total_remaining = remaining_executor + remaining_queue
            
            if total_remaining > 0:
                self.log(f"[WAIT_ALL_TASKS] Waiting for vehicle_id={track_id} (Executor: {remaining_executor}, Queue: {remaining_queue}, {remaining_timeout:.1f}s remaining)...")
                self.wait_pending_tasks_for_vehicle(track_id, timeout=remaining_timeout)
        
        # Final check - wait a bit for callbacks to complete (includes the daemon thread delay)
        time.sleep(1.0)
        
        # Explicitly wait for inference queue to drain all remaining tasks
        self.log(f"[WAIT_ALL_TASKS] Waiting for inference queue to drain...")
        try:
            from VehicleDetectionTracker.plate_utils import get_inference_queue
            queue_instance = get_inference_queue()
            if queue_instance:
                queue_instance.wait_for_all_tasks()
                self.log(f"[WAIT_ALL_TASKS] ✓ Inference queue drained")
        except Exception as e:
            self.log(f"[WAIT_ALL_TASKS] ⚠ Error draining inference queue: {e}")
        
        # Verify all are done
        with self._task_count_lock:
            final_executor = list(self.vehicle_pending_task_count.keys())
            final_queue = list(self.vehicle_pending_queue_tasks.keys())
            final_pending = list(set(final_executor + final_queue))
        
        if final_pending:
            self.log(f"[WAIT_ALL_TASKS] ⚠ Some vehicles still have pending tasks after wait:")
            for track_id in final_pending:
                exec_remaining = self.vehicle_pending_task_count.get(track_id, 0)
                queue_remaining = self.vehicle_pending_queue_tasks.get(track_id, 0)
                self.log(f"[WAIT_ALL_TASKS]   vehicle_id={track_id}: Executor={exec_remaining}, Queue={queue_remaining}")
            return False
        else:
            self.log(f"[WAIT_ALL_TASKS] ✓ All background tasks completed successfully")
            return True

    def send_notifications_for_completed_vehicles(self) -> None:
        """
        Send notifications for any remaining vehicles that haven't been notified yet.
        This is called during cleanup when video ends (fallback for vehicles with pending tasks).
        Most notifications should already be sent via _on_plate_task_complete callbacks.
        """
        import time
        global _vehicle_telegram_sent_with_plate, _vehicle_telegram_sent_without_plate, _vehicle_telegram_sent_lock
        
        self.log(f"[CLEANUP_NOTIFY] Starting cleanup notifications (fallback for remaining vehicles)...")
        
        # Get all vehicles that have ever been tracked
        # Get all vehicle IDs (thread-safe)
        all_vehicle_ids = self.get_all_vehicle_ids()
        
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
            # Get UUID for this vehicle (don't create new one, just check if it exists)
            vehicle_uuid = None
            global _track_id_to_uuid, _uuid_mapping_lock
            with _uuid_mapping_lock:
                vehicle_uuid = _track_id_to_uuid.get(track_id)
            
            # Check if already sent (by UUID)
            with _vehicle_telegram_sent_lock:
                if vehicle_uuid:
                    already_sent = (vehicle_uuid in _vehicle_telegram_sent_with_plate or 
                                  vehicle_uuid in _vehicle_telegram_sent_without_plate)
                else:
                    # UUID not created yet means vehicle never went through plate processing
                    # So it definitely hasn't sent a notification yet
                    already_sent = False
            
            if not already_sent:
                not_notified_count += 1
                with self._task_count_lock:
                    remaining_tasks = self.vehicle_pending_task_count.get(track_id, 0)
                
                # Check if vehicle has plate data (from completed detection tasks - thread-safe)
                with self._state_lock:
                    has_plate_data = track_id in self.vehicle_plates and self.vehicle_plates[track_id]
                
                if remaining_tasks == 0:
                    status = "(all tasks completed via callback)"
                elif has_plate_data:
                    status = f"({remaining_tasks} tasks still pending but plate detected, sending with available data)"
                else:
                    status = f"({remaining_tasks} tasks still pending, plate data incomplete)"
                
                uuid_str = vehicle_uuid[:8] if vehicle_uuid else "new"
                self.log(f"[CLEANUP_NOTIFY] vehicle_id={track_id} (uuid={uuid_str}) Sending cleanup notification {status}")
                
                # Sync vehicle_plates from vehicle_plates (already should be set if tasks completed)
                # Only use vehicle_plates for notification - vehicle_plate_counts is only for end-of-day summary
                if track_id not in self.vehicle_plates or not self.vehicle_plates[track_id]:
                    self.log(f"[CLEANUP_NOTIFY] vehicle_id={track_id} (uuid={uuid_str}) Warning: vehicle_plates not set, no plate to send in notification")
                
                # Find vehicle_dir from vehicle_last_seen (thread-safe)
                vehicle_dir = None
                # Try to find the directory pattern: screenshots/YYYYMMDD/HHMM_track_id
                import glob
                vehicle_last_seen_copy = self.get_vehicle_last_seen_copy()
                if track_id in vehicle_last_seen_copy:
                    date_str = vehicle_last_seen_copy[track_id].strftime("%Y%m%d")
                    pattern = f"screenshots/{date_str}/*_{track_id}"
                    matching_dirs = glob.glob(pattern)
                    if matching_dirs:
                        vehicle_dir = matching_dirs[0]
                
                result = self.send_final_vehicle_notification(track_id, vehicle_dir=vehicle_dir)
                if result:
                    self.log(f"[CLEANUP_NOTIFY] vehicle_id={track_id} (uuid={uuid_str}) ✓ Cleanup notification sent")
                else:
                    self.log(f"[CLEANUP_NOTIFY] vehicle_id={track_id} (uuid={uuid_str}) ⚠ Cleanup notification skipped")
            else:
                uuid_str = vehicle_uuid[:8] if vehicle_uuid else "unknown"
                self.log(f"[CLEANUP_NOTIFY] vehicle_id={track_id} (uuid={uuid_str}) Already notified, skipping")
        
        if not_notified_count == 0:
            self.log(f"[CLEANUP_NOTIFY] ✓ No remaining vehicles to notify (all already sent via callbacks)")
        else:
            self.log(f"[CLEANUP_NOTIFY] ✓ Cleanup notification complete ({not_notified_count} vehicles processed)")

    def send_final_vehicle_notification(self, track_id: int, vehicle_dir: Optional[str] = None) -> bool:
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
        
        # Get or create UUID for this vehicle (uniqueness across restarts)
        vehicle_uuid = self.get_or_create_uuid(track_id)
        
        # Skip if already sent (check by UUID, not track_id)
        with _vehicle_telegram_sent_lock:
            if vehicle_uuid in _vehicle_telegram_sent_with_plate or vehicle_uuid in _vehicle_telegram_sent_without_plate:
                self.log(f"[FINAL_NOTIFY] vehicle_id={track_id} (uuid={vehicle_uuid[:8]}) Already sent notification, skipping final notification")
                return False
        
        try:
            # Get vehicle state (thread-safe)
            plate_text_current, direction_label, frame_timestamp = self.get_vehicle_state(track_id)
            if direction_label is None:
                direction_label = "Unknown"
            
            # Check if direction is entering (must contain "bottom")
            is_entering = direction_label and "bottom" in direction_label.lower()
            if not is_entering:
                self.log(f"[FINAL_NOTIFY] vehicle_id={track_id} (uuid={vehicle_uuid[:8]}) Skipping final notification - vehicle is exiting or direction unknown (direction={direction_label})")
                return False
            
            plate_text = self.get_most_detected_plate(track_id)[0]  # Get the best plate detected for this vehicle
            
            # Skip if no valid plate was detected (None means no detections)
            is_valid_plate = plate_text and str(plate_text).strip().lower() != "unknown"
            
            # Set best plate in vehicle_plates if not already set (thread-safe)
            # Only store if we have a valid plate (skip None and "unknown")
            with self._state_lock:
                if is_valid_plate:
                    if track_id not in self.vehicle_plates or not self.vehicle_plates[track_id]:
                        self.vehicle_plates[track_id] = plate_text
                        self.log(f"[PLATE] vehicle_id={track_id} Set primary plate: {plate_text}")
                        
                        # Also track in vehicle_plate_counts for summary (1 vehicle = 1 count)
                        if track_id not in self.vehicle_plate_counts:
                            self.vehicle_plate_counts[track_id] = {}
                        if plate_text not in self.vehicle_plate_counts[track_id]:
                            self.vehicle_plate_counts[track_id][plate_text] = 1
                            self.log(f"[PLATE] vehicle_id={track_id} Added to plate_counts: {plate_text}")
            
            
            notification_sent = False
            
            # Case 1: Plate detected (only send if valid plate)
            if is_valid_plate:
                self.log(f"[FINAL_NOTIFY] vehicle_id={track_id} (uuid={vehicle_uuid[:8]}) Sending final notification WITH PLATE: plate={plate_text}")
                
                # Find best image matching the detected plate
                image_path = None
                
                # First, try using the stored vehicle_detected_plate_images
                if track_id in self.vehicle_detected_plate_images:
                    stored_path = self.vehicle_detected_plate_images[track_id]
                    if os.path.exists(stored_path) and os.path.getsize(stored_path) > 0:
                        image_path = stored_path
                        self.log(f"[FINAL_NOTIFY] vehicle_id={track_id} (uuid={vehicle_uuid[:8]}) Using stored image: {stored_path}")
                    else:
                        self.log(f"[FINAL_NOTIFY] vehicle_id={track_id} (uuid={vehicle_uuid[:8]}) ⚠ Stored image not found or empty: {stored_path}")
                
                # If stored image not available, search vehicle_dir
                if not image_path and vehicle_dir:
                    image_path = self._get_best_vehicle_image_by_plate(vehicle_dir, plate_text, track_id)
                    if not image_path:
                        self.log(f"[FINAL_NOTIFY] vehicle_id={track_id} (uuid={vehicle_uuid[:8]}) ⚠ No image found in {vehicle_dir}")
                    else:
                        self.log(f"[FINAL_NOTIFY] vehicle_id={track_id} (uuid={vehicle_uuid[:8]}) ✓ Found image in directory: {image_path}")
                else:
                    if image_path:
                        self.log(f"[FINAL_NOTIFY] vehicle_id={track_id} (uuid={vehicle_uuid[:8]}) ✓ Using stored image path")
                    else:
                        self.log(f"[FINAL_NOTIFY] vehicle_id={track_id} (uuid={vehicle_uuid[:8]}) ⚠ No vehicle_dir provided")
                
                # Send final notification with best plate (OUTSIDE of lock)
                self.log(f"[FINAL_NOTIFY] vehicle_id={track_id} (uuid={vehicle_uuid[:8]}) ► Calling Telegram API with image_path={image_path}...")
                try:
                    # === TIME BLOCK: TELEGRAM SEND ===
                    # with time_block(f"[TELEGRAM_SEND] vehicle_id={track_id}", self.log):
                    telegram_response = send_notify_to_telegram(
                        plate_text,
                        direction_label,
                        frame_timestamp,
                        image_path=image_path,
                    )
                    self.log(f"[FINAL_NOTIFY] vehicle_id={track_id} (uuid={vehicle_uuid[:8]}) ✓ Telegram API call completed (success={telegram_response.get('ok', False)})")
                except Exception as e:
                    self.log(f"[FINAL_NOTIFY] vehicle_id={track_id} (uuid={vehicle_uuid[:8]}) ✗ Telegram API error: {e}")
                    telegram_response = {"ok": False, "error": str(e)}
                
                # Only mark as sent if Telegram API succeeded (check 'ok' field)
                telegram_success = telegram_response.get("ok", False)
                if telegram_success:
                    # Now update tracking status inside lock (store UUID, not track_id)
                    with _vehicle_telegram_sent_lock:
                        _vehicle_telegram_sent_with_plate.add(vehicle_uuid)
                    
                    self.log(f"[FINAL_NOTIFY] vehicle_id={track_id} (uuid={vehicle_uuid[:8]}) ✓ Final notification sent with plate={plate_text}")
                    # Record successful notification in metrics
                    self.metrics.record_notification_sent(success=True, api_call=True)
                    notification_sent = True
                else:
                    self.log(f"[FINAL_NOTIFY] vehicle_id={track_id} (uuid={vehicle_uuid[:8]}) ⚠ Telegram API failed, will retry later")
                    # Record failed notification in metrics
                    self.metrics.record_notification_sent(success=False, api_call=True)
                    notification_sent = False
            
            if notification_sent:
                # === TIME BLOCK: SAVE STATE ===
                # with time_block(f"[SAVE_STATE] vehicle_id={track_id}", self.log):
                self.log(f"[FINAL_NOTIFY] vehicle_id={track_id} (uuid={vehicle_uuid[:8]}) Saving state (Telegram API was successful)...")
                self._save_state()
                self.log(f"[FINAL_NOTIFY] vehicle_id={track_id} (uuid={vehicle_uuid[:8]}) ✓ State saved")
                return True
            else:
                # Record failed notification attempt if no plate was sent
                if not notification_sent:
                    self.metrics.record_notification_sent(success=False, api_call=True)
            
            return False
                    
        except Exception as e:
            self.log(f"[FINAL_NOTIFY] vehicle_id={track_id} Error sending final notification: {e}")
            import traceback
            self.log(f"[FINAL_NOTIFY] vehicle_id={track_id} Traceback: {traceback.format_exc()}")
            # Record error in notification metrics
            self.metrics.record_notification_sent(success=False, api_call=False)
            return False

    def _get_state_file_path(self, date_str: Optional[str] = None) -> str:
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

    def _load_state(self) -> None:
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

                # Restore vehicle_directions
                if "vehicle_directions" in state:
                    for track_id_str, direction in state["vehicle_directions"].items():
                        track_id = int(track_id_str)
                        if track_id in valid_track_ids:
                            self.vehicle_directions[track_id] = direction
                    self.log(f"[PERSIST] Restored vehicle_directions: {len(self.vehicle_directions)} vehicles")
                else:
                    # BACKWARD COMPATIBILITY: Old state files don't have vehicle_directions
                    # Set default direction for all vehicles (assume they entered)
                    self.log(f"[PERSIST] ⚠ Old state file format (no vehicle_directions). Setting default direction for all vehicles...")
                    for track_id in valid_track_ids:
                        self.vehicle_directions[track_id] = "bottom"  # Assume all entered vehicles came from bottom
                    self.log(f"[PERSIST] Set default direction 'bottom' for {len(valid_track_ids)} vehicles from old state file")

                # Restore vehicle_last_seen with timestamp from state file
                # All vehicles in persisted state are from today, so use state's timestamp
                if "timestamp" in state and valid_track_ids:
                    try:
                        state_timestamp = datetime.fromisoformat(state["timestamp"])
                        for track_id in valid_track_ids:
                            self.vehicle_last_seen[track_id] = state_timestamp
                        self.log(f"[PERSIST] Set vehicle_last_seen for {len(valid_track_ids)} vehicles to {state_timestamp}")
                    except Exception as ts_err:
                        self.log(f"[PERSIST] ⚠ Could not parse state timestamp: {ts_err}")

                # Restore notification sent status (now using UUIDs instead of track_ids)
                global _vehicle_telegram_sent_with_plate, _vehicle_telegram_sent_without_plate, _vehicle_telegram_sent_lock, _track_id_to_uuid, _uuid_mapping_lock
                
                # IMPORTANT: Never restore notification status on restart!
                # Why: If system restarts, user likely wants fresh notifications, not old state
                # Only UUID mapping persists to maintain consistency across restarts
                
                # First restore UUID mapping from state (if available)
                if "track_id_to_uuid" in state:
                    with _uuid_mapping_lock:
                        for track_id_str, uuid_str in state["track_id_to_uuid"].items():
                            track_id = int(track_id_str)
                            _track_id_to_uuid[track_id] = uuid_str
                    self.log(f"[PERSIST] Restored UUID mapping for {len(state['track_id_to_uuid'])} vehicles (for consistency)")
                    self.log(f"[PERSIST] ℹ Notification status NOT restored - fresh restart, vehicles can send alerts")
                    
                    # Note: We explicitly do NOT restore sent_with_plate or sent_without_plate
                    # This allows new notifications on system restart (desired behavior)
                    # UUID mapping alone ensures consistency without blocking new detections
                else:
                    # BACKWARD COMPATIBILITY: Old state file without UUID mapping
                    # Generate UUID for each vehicle loaded from state
                    self.log(f"[PERSIST] ⚠ Old state file format (no UUID mapping). Generating UUIDs for {len(valid_track_ids)} vehicles...")
                    with _uuid_mapping_lock:
                        for track_id in valid_track_ids:
                            if track_id not in _track_id_to_uuid:
                                vehicle_uuid = str(uuid.uuid4())
                                _track_id_to_uuid[track_id] = vehicle_uuid
                                self.log(f"[PERSIST] Generated UUID for track_id={track_id}: {vehicle_uuid[:8]}")
                    
                    # IMPORTANT: Skip restoring old notification status (track_id based)
                    # Reason: track_ids can be reused after restart, so old notifications shouldn't block new detections
                    # Instead, use the freshly generated UUIDs which won't conflict
                    self.log(f"[PERSIST] ✓ Skipped old notification status (using new UUIDs instead)")

                self.log(
                    f"[PERSIST] ✓ Loaded state: {len(valid_track_ids)} vehicles from {state_file}"
                )
        except Exception as e:
            self.log(f"[PERSIST] Failed to load state: {e}")

    def check_and_reset_daily_tracking(self) -> None:
        """Check if it's a new day and reset tracking data if needed (thread-safe)."""
        today_str = datetime.now().strftime("%Y%m%d")
        
        # If last reset date is different from today, do daily reset
        if self._last_reset_date != today_str:
            with self._state_lock:
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

    def get_metrics(self) -> 'MetricsCollector':
        """
        Get metrics collector instance for accessing metrics.

        Returns:
            MetricsCollector: Current metrics collector instance
        """
        return self.metrics

    def _save_state(self) -> None:
        """Save vehicle state to JSON file for persistence (one file per day)."""
        try:
            # self.log(f"[PERSIST] _save_state: Attempting to acquire lock...")
            with self._state_lock:
                # self.log(f"[PERSIST] _save_state: Lock acquired")
                state_file = self._get_state_file_path()
                # self.log(f"[PERSIST] _save_state: state_file={state_file}")
                
                # Convert all data to JSON-serializable format
                # No need to filter - data already contains only today's vehicles (reset at day start)
                # self.log(f"[PERSIST] _save_state: Converting vehicle_plates...")
                today_plates = {
                    str(tid): plate
                    for tid, plate in self.vehicle_plates.items()
                }
                # self.log(f"[PERSIST] _save_state: Converted plates: {len(today_plates)}")
                
                # self.log(f"[PERSIST] _save_state: Converting vehicle_plate_counts...")
                today_plate_counts = {
                    str(tid): counts
                    for tid, counts in self.vehicle_plate_counts.items()
                }
                # self.log(f"[PERSIST] _save_state: Converted plate_counts: {len(today_plate_counts)}")
                
                # self.log(f"[PERSIST] _save_state: Converting vehicle_directions...")
                today_directions = {
                    str(tid): direction
                    for tid, direction in self.vehicle_directions.items()
                }
                # self.log(f"[PERSIST] _save_state: Converted directions: {len(today_directions)}")
                
                # Persist UUID mapping (for restart resilience)
                global _track_id_to_uuid, _uuid_mapping_lock
                uuid_mapping = {}
                with _uuid_mapping_lock:
                    uuid_mapping = {
                        str(tid): uuid_str
                        for tid, uuid_str in _track_id_to_uuid.items()
                    }
                
                # Persist notification sent status (now by UUID, not track_id)
                # self.log(f"[PERSIST] _save_state: Processing telegram sent status...")
                global _vehicle_telegram_sent_with_plate, _vehicle_telegram_sent_without_plate
                sent_with_plate_list = []
                sent_without_plate_list = []
                with _vehicle_telegram_sent_lock:
                    # self.log(f"[PERSIST] _save_state: Telegram lock acquired")
                    # Now storing UUIDs directly
                    for uuid_str in _vehicle_telegram_sent_with_plate:
                        sent_with_plate_list.append(uuid_str)
                    for uuid_str in _vehicle_telegram_sent_without_plate:
                        sent_without_plate_list.append(uuid_str)
                    # self.log(f"[PERSIST] _save_state: Telegram lock released")
                
                # self.log(f"[PERSIST] _save_state: Building state dict...")
                state = {
                    "vehicle_plates": today_plates,
                    "vehicle_plate_counts": today_plate_counts,
                    "vehicle_directions": today_directions,
                    "track_id_to_uuid": uuid_mapping,  # NEW: Persist UUID mapping
                    "sent_with_plate": sent_with_plate_list,
                    "sent_without_plate": sent_without_plate_list,
                    "timestamp": datetime.now().isoformat(),
                }
                # self.log(f"[PERSIST] _save_state: State dict built, size={len(str(state))} chars")

                # self.log(f"[PERSIST] _save_state: Writing to file {state_file}...")
                with open(state_file, "w", encoding="utf-8") as f:
                    json.dump(state, f, indent=2, ensure_ascii=False)
                # self.log(f"[PERSIST] _save_state: ✓ File written successfully")
            # self.log(f"[PERSIST] _save_state: Lock released")
        except Exception as e:
            self.log(f"[PERSIST] Failed to save state: {e}")
            import traceback
            self.log(f"[PERSIST] Traceback: {traceback.format_exc()}")

    def get_today_vehicles_summary(self) -> Dict[str, Any]:
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
                if plate_text == "?" or plate_text is None:  # ✅ Also skip None
                    # Only log once per vehicle to avoid spam in logs
                    if track_id not in self._vehicles_without_plate_logged:
                        missing_reason = "plate=None" if plate_text is None else "not in vehicle_plates"
                        self.log(f"[SUMMARY] vehicle_id={track_id} missing plate ({missing_reason})")
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


