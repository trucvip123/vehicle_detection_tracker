"""Frame detection and tracking processing."""

import math
import os
import time
import cv2
from datetime import datetime
from collections import defaultdict
from typing import Any, Callable, Dict
import numpy as np
from VehicleDetectionTracker.config_loader import (
    get_detection_config,
    get_tracking_config,
)
from VehicleDetectionTracker.tracking_utils import map_direction_to_label


class FrameProcessor:
    """Handles vehicle detection and tracking in frames."""

    def __init__(self, model: Any, log_func: Callable[[str], None], gpu_optimizer=None) -> None:
        self.model = model
        self.log = log_func
        self.gpu_optimizer = gpu_optimizer  # GPU optimization module

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
        self._bytetrack_alias = {}  # Map: duplicate ByteTrack ID → canonical ByteTrack ID (same physical vehicle)
        self._bytetrack_last_seen = {}  # Map: ByteTrack ID -> last frame timestamp
        self._retired_track_ids = set()  # Track IDs that have been used
        self._last_frame_bytetrack_ids = set()  # Track IDs from previous frame (detect reuse)
        
        # Load config for detection and tracking
        self.detection_config = get_detection_config()
        self.vehicle_classes = self.detection_config.get(
            "vehicle_classes", [2, 5, 6, 7, 8]
        )
        self.tracking_config = get_tracking_config()
        self._bytetrack_reuse_timeout_seconds = self.tracking_config.get(
            "bytetrack_reuse_timeout_seconds", 2.0
        )
        self._duplicate_iou_threshold = self.tracking_config.get(
            "duplicate_iou_threshold", 0.65
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
            self._bytetrack_alias.clear()
            self._bytetrack_last_seen.clear()
            self._retired_track_ids.clear()
            self._last_frame_bytetrack_ids.clear()
            
            self.log("[TRACKER_RESET] 🔄 RESET CALLED - Vehicle ID will reset on next frame (persist=False)")
            self.log("[TRACKER_RESET] ✓ Vehicle ID reset prepared - next frame will reinitialize tracker")
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            self.log(f"[TRACKER_RESET] ⚠ Warning during tracker reset: {type(e).__name__}: {e}\n{error_trace}")
            # Non-critical error - system will continue with tracking

    def set_next_track_id(self, next_id: int) -> None:
        """
        Set the persistent track ID counter to continue from a given value.
        Called on system restart within the same day to avoid ID conflicts with
        vehicles already recorded in today's state file.

        Args:
            next_id: The value to start assigning from (should be max_existing_id + 1)
        """
        if next_id > self._next_track_id:
            self.log(f"[PERSIST_ID] Resuming track ID counter from {next_id} (was {self._next_track_id})")
            self._next_track_id = next_id

    def get_persistent_track_id(
        self,
        bytetrack_id: int,
        current_time: datetime = None,
        plate_processor: Any = None,
    ) -> int:
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
        last_seen = self._bytetrack_last_seen.get(bytetrack_id)

        # === RESOLVE DUPLICATE ALIAS ===
        # If this ByteTrack ID was previously suppressed as a duplicate of another
        # ByteTrack ID (same physical vehicle), reuse the canonical vehicle's
        # persistent ID instead of creating a phantom new vehicle. This handles the
        # case where ByteTrack alternates which of two overlapping IDs it emits.
        if (
            bytetrack_id not in self._bytetrack_to_persistent_id
            and bytetrack_id in self._bytetrack_alias
        ):
            # Follow the alias chain to the canonical ByteTrack ID.
            canonical_id = bytetrack_id
            seen_ids = {canonical_id}
            while (
                canonical_id in self._bytetrack_alias
                and self._bytetrack_alias[canonical_id] not in seen_ids
            ):
                canonical_id = self._bytetrack_alias[canonical_id]
                seen_ids.add(canonical_id)

            canonical_persistent_id = self._bytetrack_to_persistent_id.get(canonical_id)
            if canonical_persistent_id is not None:
                self._bytetrack_to_persistent_id[bytetrack_id] = canonical_persistent_id
                self.log(
                    f"[PERSIST_ID] Duplicate ByteTrack={bytetrack_id} resolved to "
                    f"canonical ByteTrack={canonical_id} → Persistent={canonical_persistent_id}"
                )

        # ByteTrack may recycle small IDs like 1,2 after a vehicle disappears.
        # If the same ByteTrack ID reappears after a gap, treat it as a new vehicle.
        if (
            bytetrack_id in self._bytetrack_to_persistent_id
            and current_time is not None
            and last_seen is not None
        ):
            gap_seconds = (current_time - last_seen).total_seconds()
            if gap_seconds > self._bytetrack_reuse_timeout_seconds:
                old_persistent_id = self._bytetrack_to_persistent_id.pop(bytetrack_id)
                self.log(
                    f"[TRACK_REUSE] ByteTrack={bytetrack_id} reappeared after {gap_seconds:.1f}s "
                    f"(threshold={self._bytetrack_reuse_timeout_seconds:.1f}s). "
                    f"Remapping old persistent_id={old_persistent_id} to a new vehicle ID."
                )
                if plate_processor is not None:
                    plate_processor.clear_vehicle_data(old_persistent_id)

        # === ASSIGN NEW PERSISTENT ID OR RETURN EXISTING ===
        if bytetrack_id not in self._bytetrack_to_persistent_id:
            # New ByteTrack ID - assign next persistent ID (no timeout recycling!)
            persistent_id = self._next_track_id
            self._bytetrack_to_persistent_id[bytetrack_id] = persistent_id
            self._retired_track_ids.add(persistent_id)
            self._next_track_id += 1
            self.log(f"[PERSIST_ID] New vehicle: ByteTrack={bytetrack_id} → Persistent={persistent_id}")
        
        if current_time is not None:
            self._bytetrack_last_seen[bytetrack_id] = current_time

        return self._bytetrack_to_persistent_id[bytetrack_id]

    @staticmethod
    def _xywh_to_xyxy(box) -> tuple:
        """Convert YOLO xywh box to xyxy tuple for overlap checks."""
        x, y, w, h = [float(v) for v in box]
        return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)

    @staticmethod
    def _bbox_iou(box_a: tuple, box_b: tuple) -> float:
        """Compute IoU for two xyxy boxes."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            return 0.0

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter_area
        return inter_area / union if union > 0 else 0.0

    def _filter_duplicate_detections(self, detections: list) -> list:
        """
        Suppress overlapping duplicate tracks for the same physical vehicle.

        Prefer detections whose ByteTrack IDs already have persistent history,
        then fall back to higher confidence and larger area.
        """
        def priority(det):
            existing_persistent_id = self._bytetrack_to_persistent_id.get(det["bytetrack_id"])
            history_len = len(self.track_history.get(existing_persistent_id, [])) if existing_persistent_id is not None else 0
            has_existing_mapping = 1 if existing_persistent_id is not None else 0
            area = float(det["box"][2]) * float(det["box"][3])
            return (has_existing_mapping, history_len, det["confidence"], area)

        kept = []
        for detection in sorted(detections, key=priority, reverse=True):
            candidate_xyxy = self._xywh_to_xyxy(detection["box"])
            duplicate_of = None
            for kept_detection in kept:
                kept_xyxy = self._xywh_to_xyxy(kept_detection["box"])
                if self._bbox_iou(candidate_xyxy, kept_xyxy) >= self._duplicate_iou_threshold:
                    duplicate_of = kept_detection
                    break

            if duplicate_of is not None:
                # Remember that this ByteTrack ID is the same physical vehicle as the
                # kept one. On future frames where only this duplicate ID appears
                # (the kept ID momentarily missing), we resolve it back to the same
                # persistent vehicle instead of spawning a phantom new vehicle.
                self._bytetrack_alias[detection["bytetrack_id"]] = duplicate_of["bytetrack_id"]
                self.log(
                    f"[TRACK_DEDUP] Suppressed duplicate ByteTrack={detection['bytetrack_id']} "
                    f"(class={detection['class_id']}, conf={detection['confidence']:.3f}) "
                    f"overlapping with ByteTrack={duplicate_of['bytetrack_id']} "
                    f"(class={duplicate_of['class_id']}, conf={duplicate_of['confidence']:.3f})"
                )
                continue

            kept.append(detection)

        return kept

    def process_frame_streaming(self, frame: np.ndarray, frame_timestamp: datetime, plate_processor: Any) -> np.ndarray:
        """
        Optimized frame processing for streaming: Fast detection, background OCR.

        Args:
            frame (numpy.ndarray): Input frame for processing.
            frame_timestamp (datetime): Timestamp of the frame.
            plate_processor: PlateProcessor instance for handling plates.

        Returns:
            numpy.ndarray: Frame with license plates displayed in corner.
        """
        # self.log(f"[FRAME] Received frame for processing at {frame_timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}, shape={frame.shape if frame is not None else 'None'}") 

        frame_processing_time_start = time.time()  # For metrics
        
        # Check and reset daily tracking at start of each day
        plate_processor.check_and_reset_daily_tracking()
        
        # === VEHICLE DETECTION TIMING ===
        # with time_block("[VEHICLE_DETECT]", self.log):
        tracking_config = self.tracking_config
        
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
            confidences = (
                results[0].boxes.conf.cpu().tolist()
                if hasattr(results[0].boxes, "conf") and results[0].boxes.conf is not None
                else [0.0] * len(track_ids)
            )
            class_id_list = (
                results[0].boxes.cls.int().cpu().tolist()
                if hasattr(results[0].boxes, "cls")
                else [None] * len(track_ids)
            )

            detections = [
                {
                    "box": box,
                    "bytetrack_id": bytetrack_id,
                    "class_id": class_id,
                    "confidence": float(confidence),
                }
                for box, bytetrack_id, class_id, confidence in zip(boxes, track_ids, class_id_list, confidences)
            ]
            detections = self._filter_duplicate_detections(detections)

            boxes = [detection["box"] for detection in detections]
            track_ids = [detection["bytetrack_id"] for detection in detections]
            class_id_list = [detection["class_id"] for detection in detections]

            current_track_ids = set(track_ids)

            self.log(f"[TRACK] Detected vehicle IDs: {sorted(list(current_track_ids))}")

            # Reuse detection is handled per ByteTrack ID in get_persistent_track_id()
            # using a time-gap threshold (bytetrack_reuse_timeout_seconds).
            # Avoid global no-overlap heuristics here because they can falsely clear
            # mappings when active IDs naturally change between consecutive frames.
            
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
                track_id = self.get_persistent_track_id(
                    bytetrack_id,
                    current_time=frame_timestamp,
                    plate_processor=plate_processor,
                )
                
                class_name = coco_class_map.get(class_id, str(class_id))
                x, y, w, h = box
                print(
                    f"[DEBUG] Processing track_id={track_id}, class_id={class_id}, box=({x},{y},{w},{h})"
                )
                if w < 230 or h < 90 or y - h / 2 < 8:
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
