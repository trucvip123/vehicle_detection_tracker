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
from VehicleDetectionTracker.image_utils_helper import map_direction_to_label


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
        # Load config for detection and tracking
        self.detection_config = get_detection_config()
        self.vehicle_classes = self.detection_config.get("vehicle_classes", [2, 5, 6, 7, 8])
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
        tracking_config = get_tracking_config()
        results = self.model.track(
            frame,
            persist=True,
            tracker=tracking_config.get("tracker_type", "bytetrack.yaml"),
            classes=self.vehicle_classes,
            verbose=False,
            conf=self.detection_config.get("confidence", 0.3),
            iou=self.detection_config.get("iou", 0.45),
            imgsz=self.detection_config.get("image_size", 1280),
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
            class_id_list = results[0].boxes.cls.int().cpu().tolist() if hasattr(results[0].boxes, 'cls') else [None]*len(track_ids)
            current_track_ids = set(track_ids)

            # Log all currently detected vehicle IDs
            self.log(f"[TRACK] Current detected vehicle IDs: {sorted(list(current_track_ids))}")

            # COCO class mapping
            coco_class_map = {2: "car", 3: "motorcycle", 5: "bus", 6: "train", 7: "truck", 8: "boat"}

            # Update tracking history and calculate directions
            timestamp_str = frame_timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
            for box, track_id, class_id in zip(boxes, track_ids, class_id_list):
                class_name = coco_class_map.get(class_id, str(class_id))
                x, y, w, h = box
                if w < 230 or h < 90 or y - h / 2 < 10:
                    continue

                # Add date-based subfolder under screenshots
                date_str = datetime.now().strftime("%Y%m%d")
                time_str = datetime.now().strftime("%H%M%S")
                vehicle_dir = f"screenshots/{date_str}/{time_str}_{track_id}"
                os.makedirs(vehicle_dir, exist_ok=True)

                # Save vehicle frame
                try:
                    # Extract vehicle frame for OCR
                    vehicle_frame_save_img = frame[
                        int(y - h / 2) : int(y + h / 2),
                        int(x - w / 2) : int(x + w / 2),
                    ]
                    vehicle_frame = frame[
                        int(y - h / 2 + 160) : int(y + h / 2 + 30),
                        int(x - w / 2) : int(x + w / 2),
                    ]
                    filename = f"{vehicle_dir}/vehicle_frame_{timestamp_str}.png"
                    cv2.imwrite(filename, vehicle_frame_save_img)
                except Exception as e:
                    self.log(f"Error saving frame: {e}")

                # Update last seen and reset missing frames
                plate_processor.vehicle_last_seen[track_id] = frame_timestamp
                if track_id not in plate_processor.vehicle_missing_frames:
                    plate_processor.vehicle_missing_frames[track_id] = 0
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

                # Calculate direction_label
                timestamps = self.vehicle_timestamps[track_id]["timestamps"]
                positions = self.vehicle_timestamps[track_id]["positions"]
                direction_label = "Unknown"

                if len(positions) >= 2:
                    initial_x, initial_y = positions[0]
                    final_x, final_y = positions[-1]
                    direction = math.atan2(final_y - initial_y, final_x - initial_x)
                    direction_label = map_direction_to_label(direction)

                    plate_processor.vehicle_directions[track_id] = direction_label
                
                # Log detected vehicle info
                self.log(f"[TRACK] Detected vehicle: id={track_id}, class={class_name}, box=({x:.1f},{y:.1f},{w:.1f},{h:.1f}), direction={direction_label}")

                if direction_label == "Unknown":
                    continue

                # Process plate in background
                if vehicle_frame.size > 0:
                    timestamp_str = frame_timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    plate_processor.submit_plate_processing(
                        track_id,
                        vehicle_frame.copy(),
                        direction_label,
                        timestamp_str,
                        vehicle_dir=vehicle_dir,
                    )

        # Update missing frame counts for vehicles not detected
        all_tracked_ids = set(plate_processor.vehicle_last_seen.keys())
        missing_ids = all_tracked_ids - current_track_ids

        for track_id in missing_ids:
            if track_id not in plate_processor.vehicle_missing_frames:
                plate_processor.vehicle_missing_frames[track_id] = 0
            plate_processor.vehicle_missing_frames[track_id] += 1

        return frame
