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


class FrameProcessor:
    """Handles vehicle detection and tracking in frames."""

    def __init__(self, model: Any, log_func: Callable[[str], None]) -> None:
        self.model = model
        self.log = log_func
        self.metrics = get_metrics_collector()  # Get metrics collector instance

        # Tracking data
        self.track_history = defaultdict(lambda: [])
        self.detected_vehicles = set()
        self.vehicle_timestamps = defaultdict(
            lambda: {"timestamps": [], "positions": []}
        )
        
        # Track previous state for debug logging (only log when changed)
        self._prev_active_vehicle_ids = set()  # Track previous active vehicles to only log when changed
        
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
        
        # ===== FRAME QUALITY VALIDATION =====
        if self.quality_enabled:
            should_process, quality_metrics = should_process_frame(
                frame,
                quality_threshold=self.quality_threshold,
                brightness_min=self.brightness_min,
                brightness_max=self.brightness_max,
                blur_variance_min=self.blur_variance_min,
                contrast_min=self.contrast_min,
                entropy_min=self.entropy_min,
                log_func=self.log,
            )
            
            if not should_process:
                self._frames_rejected += 1
                quality_summary = get_frame_quality_summary(quality_metrics)
                self.log(f"{quality_summary} [REJECTED]")
                # Record frame quality rejection in metrics
                frame_processing_time = time.time() - frame_processing_time_start
                self.metrics.record_frame_processed(
                    frame_processing_time,
                    is_quality_rejected=True,
                    issues=quality_metrics.get("issues", [])
                )
                return frame
            
            self._frames_processed += 1
        
        # ===== END FRAME QUALITY VALIDATION =====
        
        tracking_config = get_tracking_config()
        results = self.model.track(
            frame,
            persist=True,
            tracker=tracking_config.get("tracker_type", "bytetrack.yaml"),
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

            for box, track_id, class_id in zip(boxes, track_ids, class_id_list):
                class_name = coco_class_map.get(class_id, str(class_id))
                x, y, w, h = box
                print(
                    f"[DEBUG] Processing track_id={track_id}, class_id={class_id}, box=({x},{y},{w},{h})"
                )
                if w < 230 or h < 90 or y - h / 2 < 10:
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
                    # cv2.imwrite(filename_process, vehicle_frame)
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
        
        return frame
    
    def _cleanup_old_vehicle_data(self, current_timestamp: datetime, plate_processor: Any) -> None:
        """
        Periodically clean up old vehicle data from memory to prevent memory leaks.
        Removes vehicles that haven't been seen for more than _vehicle_timeout seconds.
        
        Args:
            current_timestamp (datetime): Current frame timestamp
            plate_processor: PlateProcessor instance to check vehicle_last_seen
        """
        # Only run cleanup every _cleanup_interval seconds
        time_since_cleanup = (current_timestamp - self._last_cleanup).total_seconds()
        if time_since_cleanup < self._cleanup_interval:
            return
        
        self._last_cleanup = current_timestamp
        
        # Find vehicles to remove (not seen for >_vehicle_timeout seconds)
        vehicles_to_remove = []
        for track_id, ts_data in self.vehicle_timestamps.items():
            if plate_processor.vehicle_last_seen.get(track_id):
                last_seen = plate_processor.vehicle_last_seen[track_id]
                time_since_seen = (current_timestamp - last_seen).total_seconds()
                
                if time_since_seen > self._vehicle_timeout:
                    vehicles_to_remove.append((track_id, time_since_seen))
        
        # Remove old vehicles
        if vehicles_to_remove:
            self.log(f"[MEMORY_CLEANUP] Cleaning up {len(vehicles_to_remove)} old vehicles")
            for track_id, time_since_seen in vehicles_to_remove:
                # Get memory stats before cleanup
                num_positions = len(self.vehicle_timestamps[track_id]["positions"])
                num_timestamps = len(self.vehicle_timestamps[track_id]["timestamps"])
                
                # Remove from memory
                del self.vehicle_timestamps[track_id]
                if track_id in self.track_history:
                    del self.track_history[track_id]
                
                self.log(f"[MEMORY_CLEANUP] vehicle_id={track_id} removed (inactive for {time_since_seen:.0f}s, freed {num_positions} positions, {num_timestamps} timestamps)")
        
        # Log memory stats only when state changes (avoid spam logs every cleanup cycle)
        total_vehicles = len(self.vehicle_timestamps)
        total_positions = sum(len(v["positions"]) for v in self.vehicle_timestamps.values())
        current_state = (total_vehicles, total_positions)
        
        if current_state != self._last_cleanup_state:
            self.log(f"[MEMORY_CLEANUP] Current state: {total_vehicles} vehicles, {total_positions} total positions in memory")
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

    def set_quality_thresholds(
        self,
        quality_threshold: Optional[float] = None,
        brightness_min: Optional[float] = None,
        brightness_max: Optional[float] = None,
        blur_variance_min: Optional[float] = None,
        contrast_min: Optional[float] = None,
        entropy_min: Optional[float] = None,
    ) -> None:
        """
        Set custom frame quality validation thresholds.

        Args:
            quality_threshold (float): Minimum quality score (0-100)
            brightness_min (float): Minimum brightness (0-255)
            brightness_max (float): Maximum brightness (0-255)
            blur_variance_min (float): Minimum Laplacian variance for sharpness
            contrast_min (float): Minimum contrast (std dev)
            entropy_min (float): Minimum histogram entropy
        """
        if quality_threshold is not None:
            self.quality_threshold = quality_threshold
        if brightness_min is not None:
            self.brightness_min = brightness_min
        if brightness_max is not None:
            self.brightness_max = brightness_max
        if blur_variance_min is not None:
            self.blur_variance_min = blur_variance_min
        if contrast_min is not None:
            self.contrast_min = contrast_min
        if entropy_min is not None:
            self.entropy_min = entropy_min

        self.log(
            f"[QUALITY] Thresholds updated: quality={self.quality_threshold}, "
            f"brightness=[{self.brightness_min}, {self.brightness_max}], "
            f"blur_var={self.blur_variance_min}, contrast={self.contrast_min}, "
            f"entropy={self.entropy_min}"
        )

    def get_metrics(self) -> 'MetricsCollector':
        """
        Get metrics collector instance for accessing metrics.

        Returns:
            MetricsCollector: Current metrics collector instance
        """
        return self.metrics

