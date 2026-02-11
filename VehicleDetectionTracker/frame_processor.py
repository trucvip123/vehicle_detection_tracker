"""Frame detection and tracking processing."""

import math
import os
import cv2
from datetime import datetime
from collections import defaultdict
from VehicleDetectionTracker.config_loader import (
    get_detection_config,
    get_tracking_config,
)
from VehicleDetectionTracker.tracking_utils import map_direction_to_label


class FrameProcessor:
    """Handles vehicle detection and tracking in frames."""

    def __init__(self, model, log_func):
        self.model = model
        self.log = log_func

        # Tracking data
        self.track_history = defaultdict(lambda: [])
        self.detected_vehicles = set()
        self.vehicle_timestamps = defaultdict(
            lambda: {"timestamps": [], "positions": []}
        )
        # Temporary vehicle IDs for current session (detected but not yet persisted in JSON)
        self.temp_vehicle_ids = set()
        
        # Load config for detection and tracking
        self.detection_config = get_detection_config()
        self.vehicle_classes = self.detection_config.get(
            "vehicle_classes", [2, 5, 6, 7, 8]
        )
        self.log(f"[DETECT] vehicle_classes sử dụng: {self.vehicle_classes}")

    def process_frame_streaming(self, frame, frame_timestamp, plate_processor):
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
        # Check and reset daily tracking at start of each day
        plate_processor.check_and_reset_daily_tracking()
        
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
            time_str = datetime.now().strftime("%H%M%S")

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

                # Update last seen and reset missing frames
                plate_processor.vehicle_last_seen[track_id] = frame_timestamp
                self.log(
                    f"[DEBUG] Updated vehicle_last_seen[{track_id}] = {frame_timestamp}"
                )
                plate_processor._save_state()  # Persist state after updating

                # Reset missing frames counter when vehicle is detected
                plate_processor.vehicle_missing_frames[track_id] = 0

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
                timestamps = self.vehicle_timestamps[track_id]["timestamps"]
                positions = self.vehicle_timestamps[track_id]["positions"]
                direction_label = "Unknown"

                if len(positions) >= 2:
                    initial_x, initial_y = positions[0]
                    final_x, final_y = positions[-1]
                    direction = math.atan2(final_y - initial_y, final_x - initial_x)
                    direction_label = map_direction_to_label(direction)
                    
                    # Check if direction changed from previous update
                    prev_direction = plate_processor.vehicle_directions.get(track_id, "Unknown")
                    self.log(
                        f"[TRACK] vehicle_id={track_id} direction_calculation: points={len(positions)}, from=({initial_x:.1f},{initial_y:.1f}) to=({final_x:.1f},{final_y:.1f}), angle={direction:.2f}, label={direction_label}, prev={prev_direction}"
                    )
                    
                    plate_processor.vehicle_directions[track_id] = direction_label
                    plate_processor._save_state()  # Persist state after updating direction
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

                # Process plate in background
                if vehicle_frame.size > 0:
                    timestamp_str = frame_timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    self.log(
                        f"[FRAME] Submitting plate processing for vehicle_id={track_id}"
                    )
                    plate_processor.submit_plate_processing(
                        track_id,
                        vehicle_frame.copy(),
                        direction_label,
                        frame_timestamp,
                        timestamp_str,
                        vehicle_dir=vehicle_dir,
                    )

        # Update missing frame counts for vehicles not detected
        # Separate tracking: today_vehicle_ids (from JSON) vs temp_vehicle_ids (current session)
        today_str = datetime.now().strftime("%Y%m%d")
        today_vehicle_ids = set()

        for track_id, ts in plate_processor.vehicle_last_seen.items():
            if hasattr(ts, "strftime") and ts.strftime("%Y%m%d") == today_str:
                today_vehicle_ids.add(track_id)
        
        # print("[DEBUG] today_vehicle_ids (from JSON):", today_vehicle_ids)
        
        if self.temp_vehicle_ids:
            self.log(f"[DEBUG] temp_vehicle_ids (session): {self.temp_vehicle_ids}")
        # print("[DEBUG] current_track_ids (now):", current_track_ids)
        
        # Step 1: Remove vehicles from temp_vehicle_ids if already saved in JSON
        self.temp_vehicle_ids = self.temp_vehicle_ids - today_vehicle_ids
        
        # Step 2: Add newly detected vehicles to temp_vehicle_ids
        new_vehicles = current_track_ids - today_vehicle_ids
        self.temp_vehicle_ids.update(new_vehicles)
        if new_vehicles:
            self.log(
                f"[FRAME] Step 1-2: new_vehicles={new_vehicles}, updated temp_vehicle_ids={sorted(self.temp_vehicle_ids)}"
            )
        
        # Step 3: Calculate missing vehicles (in temp_vehicle_ids but not in current_track_ids)
        missing_ids = self.temp_vehicle_ids - current_track_ids
        if missing_ids:
            self.log(
                f"[FRAME] Step 3: missing_ids={missing_ids}"
            )

        # Step 4: Increment missing_frames only for vehicles in temp_vehicle_ids (and are entering, not exiting)
        for track_id in missing_ids:
            # Check if vehicle is entering (not exiting)
            direction = plate_processor.vehicle_directions.get(track_id, "Unknown")
            # Only count missing frames for entering vehicles (not for exiting/"top" vehicles)
            if direction and "top" not in direction.lower():
                if track_id not in plate_processor.vehicle_missing_frames:
                    plate_processor.vehicle_missing_frames[track_id] = 0
                plate_processor.vehicle_missing_frames[track_id] += 1
                self.log(
                    f"[TRACK] vehicle_id={track_id} missing_frame_count={plate_processor.vehicle_missing_frames[track_id]}"
                )
            else:
                # Skip counting for exiting vehicles
                self.log(
                    f"[TRACK] vehicle_id={track_id} Skipping missing_frame count (direction={direction}, vehicle is exiting)"
                )

        # Step 5: Remove vehicles with missing_frame_count > 1000
        vehicles_to_remove = [
            track_id
            for track_id, count in plate_processor.vehicle_missing_frames.items()
            if count > 100
        ]
        
        if vehicles_to_remove:
            self.log(
                f"[FRAME] Step 5: Removing {len(vehicles_to_remove)} old vehicles: {vehicles_to_remove}"
            )
        
        for track_id in vehicles_to_remove:
            self.log(
                f"[FRAME] Removing vehicle_id={track_id} (missing_frame_count={plate_processor.vehicle_missing_frames[track_id]})"
            )
            # Remove from all tracking dictionaries
            plate_processor.vehicle_missing_frames.pop(track_id, None)
            plate_processor.vehicle_last_seen.pop(track_id, None)
            plate_processor.vehicle_directions.pop(track_id, None)
            plate_processor.vehicle_plates.pop(track_id, None)
            plate_processor.vehicle_plate_counts.pop(track_id, None)
            self.track_history.pop(track_id, None)
            self.vehicle_timestamps.pop(track_id, None)
            self.temp_vehicle_ids.discard(track_id)

        if current_track_ids:
            self.log(
                f"[FRAME] Frame processing completed. Active vehicles: {sorted(list(current_track_ids))}"
            )
        return frame
