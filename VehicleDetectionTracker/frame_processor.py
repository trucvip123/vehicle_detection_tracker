"""Frame detection and tracking processing."""

import math
import os
import glob
import time
import cv2
from datetime import datetime
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np
from VehicleDetectionTracker.config_loader import (
    get_detection_config,
    get_tracking_config,
)
from VehicleDetectionTracker.tracking_utils import map_direction_to_label
from VehicleDetectionTracker.frame_quality import (
    should_process_frame,
    get_frame_quality_summary,
)
from VehicleDetectionTracker.metrics import get_metrics_collector
from VehicleDetectionTracker.performance_timing import time_block


class FrameProcessor:
    """Handles vehicle detection and tracking in frames."""

    def __init__(self, model: Any, log_func: Callable[[str], None], gpu_optimizer=None) -> None:
        self.model = model
        self.log = log_func
        self.gpu_optimizer = gpu_optimizer  # GPU optimization module
        self.metrics = get_metrics_collector()  # Get metrics collector instance

        # Tracking data
        self.track_history = defaultdict(lambda: [])
        self.detected_vehicles = set()
        self.vehicle_timestamps = defaultdict(
            lambda: {"timestamps": [], "positions": []}
        )
        
        # Track previous state for debug logging (only log when changed)
        self._prev_active_vehicle_ids = set()  # Track previous active vehicles to only log when changed
        
        # Flag to skip persistence on first frame after reset
        self._force_fresh_tracker_next_frame = False
        
        # Persistent track ID mapping (NO timeout-based recycling during day!)
        # IDs increment continuously: 1, 2, 3, 4... until midnight daily reset
        self._next_track_id = 1  # Next ID to assign (increments monotonically during session)
        self._bytetrack_to_persistent_id = {}  # Map: ByteTrack ID → persistent ID
        self._retired_track_ids = set()  # Track IDs that have been used
        self._last_frame_bytetrack_ids = set()  # Track IDs from previous frame (detect reuse)
        
        # Load config for detection and tracking
        self.detection_config = get_detection_config()
        self.vehicle_classes = self.detection_config.get(
            "vehicle_classes", [2, 5, 6, 7, 8]
        )
        self.log(f"[DETECT] vehicle_classes sử dụng: {self.vehicle_classes}")
        
        # Memory cleanup settings
        self._last_cleanup = datetime.now()
        self._cleanup_interval = 60  # Clean up old vehicle data every 60 seconds
        self._vehicle_timeout = 300  # Remove vehicles not seen for 5 minutes (300s)
        self._last_cleanup_state = (0, 0)  # Track previous (total_vehicles, total_positions) to only log when changed

        # Frame quality validation settings
        self.quality_enabled = True
        self.quality_threshold = 50.0  # Minimum quality score (0-100)
        self.brightness_min = 40.0  # Minimum brightness (0-255)
        self.brightness_max = 210.0  # Maximum brightness (0-255)
        self.blur_variance_min = 100.0  # Minimum Laplacian variance for sharpness
        self.contrast_min = 15.0  # Minimum contrast (std dev)
        self.entropy_min = 3.0  # Minimum histogram entropy
        self._frames_rejected = 0  # Track rejected frames
        self._frames_processed = 0  # Track processed frames

    def enable_gpu_optimization(self, width: int, height: int):
        """
        Enable GPU optimization with resolution downscaling
        
        Args:
            width: Inference width (e.g., 1280, 960, 640)
            height: Inference height (e.g., 720, 540, 360)
            
        Performance impact (on GTX 1650 with RTSP 2880x1620):
          (1280, 720): ~18 FPS (balanced - default)
          (960, 540):  ~25 FPS (faster)
          (640, 360):  ~35 FPS (fastest)
        """
        if self.gpu_optimizer:
            self.gpu_optimizer.set_inference_resolution(width, height)
            self.log(f"[GPU] Inference resolution set to {width}x{height}")
            return True
        return False
    
    def get_gpu_optimization_status(self) -> Dict[str, Any]:
        """Get GPU optimization status"""
        if not self.gpu_optimizer:
            return {"enabled": False, "reason": "GPU optimizer not initialized"}
        
        return {
            "enabled": self.gpu_optimizer.use_gpu,
            "device": self.gpu_optimizer.device if self.gpu_optimizer.use_gpu else "CPU",
            "inference_resolution": self.gpu_optimizer.inference_resolution,
        }

    def reset_tracker(self) -> None:
        """
        Reset the ByteTrack tracker state to start fresh with vehicle IDs.
        Called at the start of each day to reset vehicle ID counter (ID=1, 2, 3, ...).
        
        This ensures:
        - Vehicle IDs restart from 1 each day
        - No ID carryover from previous day
        - Fresh tracking state for new day
        
        Strategy: Set a flag to skip persistence on the next frame,
        which forces the tracker to reinitialize cleanly.
        """
        try:
            # Set flag to force fresh tracker on next frame processing
            self._force_fresh_tracker_next_frame = True
            
            # Clear local track history (safe - doesn't affect model state)
            self.track_history.clear()
            self.detected_vehicles.clear()
            self.vehicle_timestamps.clear()
            self._prev_active_vehicle_ids.clear()
            
            # Reset persistent track ID mapping for new day
            self._next_track_id = 1
            self._bytetrack_to_persistent_id.clear()
            self._retired_track_ids.clear()
            self._last_frame_bytetrack_ids.clear()
            
            self.log("[TRACKER_RESET] 🔄 RESET CALLED - Vehicle ID will reset on next frame (persist=False)")
            self.log("[TRACKER_RESET] ✓ Vehicle ID reset prepared - next frame will reinitialize tracker")
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            self.log(f"[TRACKER_RESET] ⚠ Warning during tracker reset: {type(e).__name__}: {e}\n{error_trace}")
            # Non-critical error - system will continue with tracking

    def get_persistent_track_id(self, bytetrack_id: int, current_time: datetime = None) -> int:
        """
        Map ByteTrack ID to a persistent, non-reusable track ID.
        
        NO timeout-based recycling during the day!
        Persistent IDs increment continuously until midnight (daily reset only).
        
        Args:
            bytetrack_id: ID assigned by ByteTrack for current frame
            current_time: Current frame timestamp (unused - kept for compatibility)
            
        Returns:
            Persistent track ID (monotonically increasing, resets only at midnight)
        """
        # === ASSIGN NEW PERSISTENT ID OR RETURN EXISTING ===
        if bytetrack_id not in self._bytetrack_to_persistent_id:
            # New ByteTrack ID - assign next persistent ID (no timeout recycling!)
            persistent_id = self._next_track_id
            self._bytetrack_to_persistent_id[bytetrack_id] = persistent_id
            self._retired_track_ids.add(persistent_id)
            self._next_track_id += 1
            self.log(f"[PERSIST_ID] New vehicle: ByteTrack={bytetrack_id} → Persistent={persistent_id}")
        
        return self._bytetrack_to_persistent_id[bytetrack_id]

    def process_frame_streaming(self, frame: np.ndarray, frame_timestamp: datetime, plate_processor: Any) -> np.ndarray:
        """
        Optimized frame processing for streaming: Fast detection, background OCR.
        Only shows license plates in corner, no bounding boxes on vehicles.

        Args:
            frame (numpy.ndarray): Input frame for processing.
            frame_timestamp (datetime): Timestamp of the frame.
            plate_processor: PlateProcessor instance for handling plates.

        Returns:
            numpy.ndarray: Frame with license plates displayed in corner.
        """
        frame_start_time = datetime.now()  # Start timing frame processing
        frame_processing_time_start = time.time()  # For metrics
        
        # Check and reset daily tracking at start of each day
        plate_processor.check_and_reset_daily_tracking()
        
        # === VEHICLE DETECTION TIMING ===
        # with time_block("[VEHICLE_DETECT]", self.log):
        tracking_config = get_tracking_config()
        
        # Use fresh tracker on first frame after daily reset
        use_persist = True
        if self._force_fresh_tracker_next_frame:
            use_persist = False
            self._force_fresh_tracker_next_frame = False
            self.log("[TRACK] First frame after reset - using fresh tracker (persist=False)")
        
        # DEBUG: Log tracking state for each frame
        # self.log(f"[TRACK_DEBUG] Processing frame - use_persist={use_persist}, _force_fresh_tracker_next_frame={self._force_fresh_tracker_next_frame}")
        
        results = self.model.track(
            frame,
            persist=use_persist,
            tracker=tracking_config.get("tracker_type"),
            classes=self.vehicle_classes,
            verbose=False,
            conf=self.detection_config.get("confidence"),
            iou=self.detection_config.get("iou"),
            imgsz=self.detection_config.get("image_size"),
        )

        # Track currently detected vehicles
        current_track_ids = set()

        if (
            results is not None
            and results[0] is not None
            and results[0].boxes is not None
            and results[0].boxes.id is not None
        ):
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_id_list = (
                results[0].boxes.cls.int().cpu().tolist()
                if hasattr(results[0].boxes, "cls")
                else [None] * len(track_ids)
            )
            current_track_ids = set(track_ids)

            # === DETECT BYTETRACK REUSE AND CLEAR OLD MAPPINGS ===
            # ByteTrack reuses IDs when vehicles disappear for > 300 seconds.
            # When all ByteTrack IDs change AND we have old mappings:
            # 1. Clear ByteTrack→Persistent mappings (allow new assignments)
            # 2. Keep persistent ID counter ticking UP (don't reset to 1!)
            # 3. Clear old vehicle data from plate_processor (UUID, direction, plates)
            if (
                current_track_ids and 
                self._bytetrack_to_persistent_id and 
                not current_track_ids.intersection(self._last_frame_bytetrack_ids)
            ):
                # No overlap between current and previous frame ByteTrack IDs
                # AND we have old mappings: ByteTrack likely reset
                old_mappings = dict(self._bytetrack_to_persistent_id)
                old_persistent_ids = list(old_mappings.values())
                
                self.log(
                    f"[TRACK_REUSE] ⚠️ ByteTrack ID reuse detected! "
                    f"Current={sorted(current_track_ids)}, "
                    f"Previous={sorted(self._last_frame_bytetrack_ids)}, "
                    f"Old_Mappings={old_mappings}"
                )
                
                # Clear old ByteTrack→Persistent mappings only (keep counter ticking)
                self._bytetrack_to_persistent_id.clear()
                # ⚠️ IMPORTANT: DO NOT reset _next_track_id=1! Keep it incrementing
                # This ensures persistent IDs keep going: 1,2,3,4,5... (not 1,1,1,1)
                
                self.log(
                    f"[TRACK_REUSE] ✓ Cleared ByteTrack mappings. "
                    f"Persistent ID counter keeps incrementing (current={self._next_track_id}). "
                    f"Cleaning old vehicle data from plate_processor for old IDs: {old_persistent_ids}"
                )
                
                # Clear old vehicle data (UUID mapping, direction, plates) from plate_processor
                # so new vehicles with same persistent IDs won't inherit old data
                for old_persistent_id in old_persistent_ids:
                    plate_processor.clear_vehicle_data(old_persistent_id)
                    self.log(f"[TRACK_REUSE] ✓ Cleared plate_processor data for old persistent_id={old_persistent_id}")
            
            # Update last frame IDs for next iteration
            self._last_frame_bytetrack_ids = current_track_ids.copy()

            # Log all currently detected vehicle IDs
            self.log(
                f"[TRACK] Current detected vehicle IDs: {sorted(list(current_track_ids))}"
            )

            # COCO class mapping
            coco_class_map = {
                2: "car",
                3: "motorcycle",
                5: "bus",
                6: "train",
                7: "truck",
                8: "boat",
            }

            # Update tracking history and calculate directions
            timestamp_str = frame_timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
            # Add date-based subfolder under screenshots
            date_str = datetime.now().strftime("%Y%m%d")
            time_str = datetime.now().strftime("%H%M")
            
            # Collect vehicles for batch processing
            frame_vehicles_batch = {}

            for box, bytetrack_id, class_id in zip(boxes, track_ids, class_id_list):
                # Convert ByteTrack ID to persistent ID (no timeout recycling - only at midnight!)
                track_id = self.get_persistent_track_id(bytetrack_id)
                
                class_name = coco_class_map.get(class_id, str(class_id))
                x, y, w, h = box
                print(
                    f"[DEBUG] Processing track_id={track_id}, class_id={class_id}, box=({x},{y},{w},{h})"
                )
                if w < 230 or h < 90 or y - h / 2 < 10:
                    self.log(
                        f"[TRACK] vehicle_id={track_id} Skipping small/low vehicle: box=({x:.1f},{y:.1f},{w:.1f},{h:.1f})"
                    )
                    continue

                vehicle_dir = f"screenshots/{date_str}/{time_str}_{track_id}"
                os.makedirs(vehicle_dir, exist_ok=True)

                # Save vehicle frame
                try:
                    # Extract vehicle frame for OCR
                    vehicle_frame_save_img = frame[
                        int(y - h / 2 - 40) : int(y + h / 2 + 40),
                        int(x - w / 2) : int(x + w / 2),
                    ]
                    vehicle_frame = frame[
                        int(y - h / 2 + 200) : int(y + h / 2 + 40),
                        int(x - w / 2) : int(x + w / 2),
                    ]
                    filename = f"{vehicle_dir}/vehicle_frame_{timestamp_str}.png"
                    # filename_process = f"{vehicle_dir}/vehicle_frame_process_{timestamp_str}.png"
                    cv2.imwrite(filename, vehicle_frame_save_img)
                    self.log(f"[FRAME] Saved vehicle frame for track_id={track_id} at {filename}")
                except Exception as e:
                    self.log(f"Error saving frame: {e}")

                # Update last seen (thread-safe via plate_processor)
                plate_processor.update_vehicle_state(track_id, timestamp=frame_timestamp)
                self.log(
                    f"[DEBUG] Updated vehicle_last_seen[{track_id}] = {frame_timestamp}"
                )
                # Note: State will be saved only when notification is sent


                # Update tracking history
                if track_id not in self.track_history:
                    self.track_history[track_id] = []
                track = self.track_history[track_id]
                track.append((float(x), float(y)))
                tracking_config = get_tracking_config()
                max_history_length = tracking_config.get("max_history_length", 30)
                if len(track) > max_history_length:
                    track.pop(0)

                # Update timestamps and positions
                if track_id not in self.vehicle_timestamps:
                    self.vehicle_timestamps[track_id] = {
                        "timestamps": [],
                        "positions": [],
                    }
                self.vehicle_timestamps[track_id]["timestamps"].append(frame_timestamp)
                self.vehicle_timestamps[track_id]["positions"].append((x, y))
                
                # Log tracking history
                num_tracking_points = len(self.vehicle_timestamps[track_id]["positions"])
                self.log(
                    f"[TRACK] vehicle_id={track_id} tracking_points={num_tracking_points}, position=({x:.1f},{y:.1f})"
                )

                # Calculate direction_label
                positions = self.vehicle_timestamps[track_id]["positions"]
                direction_label = "Unknown"

                if len(positions) >= 2:
                    # Use only recent positions (not entire history) for accurate current direction
                    tracking_config = get_tracking_config()
                    max_history_length = tracking_config.get("max_history_length", 30)
                    recent_positions = positions[-max_history_length:]
                    
                    initial_x, initial_y = recent_positions[0]
                    final_x, final_y = recent_positions[-1]
                    direction = math.atan2(final_y - initial_y, final_x - initial_x)
                    direction_label = map_direction_to_label(direction)
                    
                    # Check if direction changed from previous update (thread-safe read)
                    prev_direction = plate_processor.get_vehicle_directions_copy().get(track_id, "Unknown")
                    self.log(
                        f"[TRACK] vehicle_id={track_id} direction_calculation: recent_points={len(recent_positions)}/{len(positions)}, from=({initial_x:.1f},{initial_y:.1f}) to=({final_x:.1f},{final_y:.1f}), angle={direction:.2f}, label={direction_label}, prev={prev_direction}"
                    )
                    
                    # Update direction (thread-safe via plate_processor)
                    plate_processor.update_vehicle_state(track_id, direction=direction_label)
                    # Note: State will be saved only when notification is sent
                else:
                    self.log(
                        f"[TRACK] vehicle_id={track_id} insufficient_tracking_points={len(positions)} (need >= 2)"
                    )

                # Log detected vehicle info
                self.log(
                    f"[TRACK] Detected vehicle: id={track_id}, class={class_name}, box=({x:.1f},{y:.1f},{w:.1f},{h:.1f}), direction={direction_label}"
                )

                if direction_label == "Unknown":
                    self.log(
                        f"[TRACK] vehicle_id={track_id} Skipping processing - direction is Unknown"
                    )
                    continue

                if "bottom" not in direction_label.lower():
                    self.log(
                        f"[TRACK] vehicle_id={track_id} Skipping processing - direction is {direction_label} (expected: bottom)"
                    )
                    continue

                # Add to batch for submission
                if vehicle_frame.size > 0:
                    frame_vehicles_batch[track_id] = {
                        'frame': vehicle_frame.copy(),
                        'direction': direction_label,
                        'timestamp': frame_timestamp,
                        'timestamp_str': timestamp_str,
                        'vehicle_dir': vehicle_dir
                    }
                    self.log(
                        f"[FRAME] Added vehicle_id={track_id} to batch for processing"
                    )
            
            # Submit entire batch for processing
            if frame_vehicles_batch:
                self.log(
                    f"[FRAME] Submitting batch processing for {len(frame_vehicles_batch)} vehicles"
                )
                plate_processor.submit_plate_processing_batch(frame_vehicles_batch)

        # Periodically clean up old vehicle data from memory
        self._cleanup_old_vehicle_data(frame_timestamp, plate_processor)
        
        # Only log when active vehicles change (avoid repeating same logs every frame)
        if current_track_ids != self._prev_active_vehicle_ids:
            if current_track_ids:
                self.log(
                    f"[FRAME] Frame processing completed. Active vehicles: {sorted(list(current_track_ids))}"
                )
            self._prev_active_vehicle_ids = current_track_ids.copy()
        
        # ===== RECORD METRICS =====
        frame_total_time = time.time() - frame_processing_time_start
        
        # Record frame processing metrics
        self.metrics.record_frame_processed(
            frame_total_time,
            is_quality_rejected=False,
            issues=[]
        )
        
        # Record detection metrics
        vehicles_count = len(current_track_ids)
        # Calculate average detection confidence if we have results
        avg_confidence = 0.0
        if (
            results is not None
            and results[0] is not None
            and hasattr(results[0].boxes, "conf")
            and results[0].boxes.conf is not None
        ):
            confidences = results[0].boxes.conf.cpu().numpy()
            if len(confidences) > 0:
                avg_confidence = float(confidences.mean())
        
        self.metrics.record_detection(
            confidence=avg_confidence,
            vehicles_detected_count=vehicles_count,
            processing_time=frame_total_time
        )
        
        # Record tracking metrics
        self.metrics.record_tracking_update(
            active_vehicles=len(current_track_ids),
            new_vehicle=False,  # Updated per-vehicle in loop above
            tracking_lost=False,
            reassigned=False
        )
        
        # return frame
    
    def _cleanup_old_vehicle_data(self, current_timestamp: datetime, plate_processor: Any) -> None:
        """
        Periodically clean up old vehicle data from memory to prevent memory leaks.
        Removes vehicles that haven't been seen for more than _vehicle_timeout seconds.
        
        Args:
            current_timestamp (datetime): Current frame timestamp
            plate_processor: PlateProcessor instance to check vehicle_last_seen
        """
        # === TIME BLOCK: CLEANUP CHECK ===
        # with time_block("[CLEANUP_CHECK]", self.log):
        # Only run cleanup every _cleanup_interval seconds
        time_since_cleanup = (current_timestamp - self._last_cleanup).total_seconds()
        if time_since_cleanup < self._cleanup_interval:
            return
        
        self._last_cleanup = current_timestamp
        
        # === TIME BLOCK: FIND EXPIRED VEHICLES ===
        # with time_block("[CLEANUP_FIND]", self.log):
        # Find vehicles to remove (not seen for >_vehicle_timeout seconds)
        vehicles_to_remove = []
        for track_id, ts_data in self.vehicle_timestamps.items():
            if plate_processor.vehicle_last_seen.get(track_id):
                last_seen = plate_processor.vehicle_last_seen[track_id]
                time_since_seen = (current_timestamp - last_seen).total_seconds()
                
                if time_since_seen > self._vehicle_timeout:
                    vehicles_to_remove.append((track_id, time_since_seen))
    
        # === TIME BLOCK: REMOVE OLD VEHICLES ===
        # with time_block("[CLEANUP_REMOVE]", self.log):
        # Remove old vehicles
        if vehicles_to_remove:
            # self.log(f"[MEMORY_CLEANUP] Cleaning up {len(vehicles_to_remove)} old vehicles")
            for track_id, time_since_seen in vehicles_to_remove:
                # Get memory stats before cleanup
                num_positions = len(self.vehicle_timestamps[track_id]["positions"])
                num_timestamps = len(self.vehicle_timestamps[track_id]["timestamps"])
                
                # Remove from memory
                del self.vehicle_timestamps[track_id]
                if track_id in self.track_history:
                    del self.track_history[track_id]
                
                # self.log(f"[MEMORY_CLEANUP] vehicle_id={track_id} removed (inactive for {time_since_seen:.0f}s, freed {num_positions} positions, {num_timestamps} timestamps)")
        
        # Log memory stats only when state changes (avoid spam logs every cleanup cycle)
        total_vehicles = len(self.vehicle_timestamps)
        total_positions = sum(len(v["positions"]) for v in self.vehicle_timestamps.values())
        current_state = (total_vehicles, total_positions)
        
        if current_state != self._last_cleanup_state:
            # self.log(f"[MEMORY_CLEANUP] Current state: {total_vehicles} vehicles, {total_positions} total positions in memory")
            self._last_cleanup_state = current_state

    def set_quality_validation(self, enabled: bool) -> None:
        """
        Enable or disable frame quality validation.

        Args:
            enabled (bool): True to enable quality validation, False to disable.
        """
        self.quality_enabled = enabled
        status = "ENABLED" if enabled else "DISABLED"
        self.log(f"[QUALITY] Frame quality validation {status}")

    def get_quality_stats(self) -> Dict[str, int]:
        """
        Get frame quality validation statistics.

        Returns:
            Dict with statistics:
            - frames_processed: Number of frames that passed quality check
            - frames_rejected: Number of frames that failed quality check
            - rejection_rate: Percentage of frames rejected (0-100)
        """
        total_frames = self._frames_processed + self._frames_rejected
        rejection_rate = (
            (self._frames_rejected / total_frames * 100) if total_frames > 0 else 0
        )
        return {
            "frames_processed": self._frames_processed,
            "frames_rejected": self._frames_rejected,
            "total_frames": total_frames,
            "rejection_rate": rejection_rate,
        }
