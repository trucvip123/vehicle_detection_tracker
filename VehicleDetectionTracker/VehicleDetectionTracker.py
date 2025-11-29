import math
import cv2
import base64
from collections import defaultdict
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import time

from ultralytics import YOLO
import numpy as np
from ultralytics.utils.plotting import colors

from datetime import datetime
import os
import pandas as pd
from VehicleDetectionTracker.function.paddleocr_wrapper import create_paddleocr_reader
from VehicleDetectionTracker.plate_utils import (
    initialize_plate_detector,
    preprocess_plate_image,
    detect_license_plate_sync,
    detect_license_plate_async,
)
from VehicleDetectionTracker.utils.send_bot import send_notify_to_telegram
import logging

logging.getLogger("ultralytics").setLevel(logging.WARNING)  # Suppress ultralytics logging


class VehicleDetectionTracker:

    def __init__(self, model_path="yolov8n.pt", excel_output_path="vehicle_data.xlsx", initialize_all_models=True, stream_frame_size=None):
        """
        Initialize the VehicleDetection class.

        Args:
            model_path (str): Path to the YOLO model file.
            excel_output_path (str): Path to Excel file for saving vehicle data.
            initialize_all_models (bool): If True, initialize all models (YOLO, plate detector, OCR) 
                                         immediately. If False, use lazy loading for OCR.
        """
        print("Initializing Vehicle Detection Tracker...")
        
        # Load the YOLO model (always loaded first)
        print("Loading YOLO vehicle detection model...")
        self.model = YOLO(model_path)
        print("✓ YOLO model loaded")
        
        self.track_history = defaultdict(lambda: [])  # History of vehicle tracking
        self.detected_vehicles = set()  # Set of detected vehicles
        self.color_classifier = None
        self.model_classifier = None
        self.vehicle_timestamps = defaultdict(
            list
        )  # Keep track of timestamps for each tracked vehicle
        
        # Initialize license plate detector model (always loaded)
        print("Loading license plate detector model...")
        self.plate_model = None
        self._initialize_plate_detector()
        print("✓ License plate detector loaded")
        
        # Initialize OCR reader based on flag
        self.ocr_reader = None
        if initialize_all_models:
            print("Initializing OCR reader...")
            self._initialize_ocr_reader()
            print("✓ OCR reader initialized")
        else:
            print("⚠ OCR reader will be initialized on first use (lazy loading)")
        
        self.text_plate = None
        # Optional target frame size for resizing streaming/video frames.
        # Should be a tuple (width, height) or None to disable resizing.
        # Example: stream_frame_size=(640, 480)
        self.stream_frame_size = stream_frame_size
        
        # Thread pool for async operations
        self._executor = ThreadPoolExecutor(max_workers=4)
        # Lock for thread-safe model access
        self._model_lock = threading.Lock()
        # Store detected license plates per vehicle ID for streaming display
        self.vehicle_plates = {}  # {track_id: plate_text} - most recent plate
        # Store license plate detection counts for each vehicle
        self.vehicle_plate_counts = defaultdict(lambda: defaultdict(int))  # {track_id: {plate_text: count}}
        # Track which vehicles have been saved to Excel (to avoid duplicates)
        self.vehicle_saved_to_excel = set()  # {track_id}
        # Store direction_label for each vehicle
        self.vehicle_directions = {}  # {track_id: direction_label}
        # Track last seen frame for each vehicle
        self.vehicle_last_seen = {}  # {track_id: timestamp}
        # Track number of consecutive frames vehicle has been missing
        self.vehicle_missing_frames = {}  # {track_id: missing_frame_count}
        # Track which vehicles have already sent Telegram notifications (send only once per ID)
        self.vehicle_telegram_sent = set()  # {track_id}
        # Excel output file path
        self.excel_output_path = excel_output_path
        # Lock for Excel file operations
        self._excel_lock = threading.Lock()
        # Initialize Excel file if it doesn't exist
        self._initialize_excel_file()
        
        print("✓ All initialization complete!")

    def _initialize_ocr_reader(self):
        """
        Initialize OCR reader (PaddleOCR).
        """
        if self.ocr_reader is None:
            self.ocr_reader = create_paddleocr_reader(
                lang="en", use_angle_cls=True, show_log=False
            )

    def ensure_all_models_initialized(self):
        """
        Ensure all models are initialized. Useful if initialize_all_models=False was used.
        Call this before streaming to avoid delays during processing.
        """
        if self.model is None:
            raise RuntimeError("YOLO model not initialized!")
        
        if self.plate_model is None:
            print("Warning: Plate detector not initialized, reinitializing...")
            self._initialize_plate_detector()
        
        if self.ocr_reader is None:
            print("Warning: OCR reader not initialized, initializing now...")
            self._initialize_ocr_reader()
            print("✓ OCR reader initialized")
    
    def get_initialization_status(self):
        """
        Get status of model initialization.
        
        Returns:
            dict: Status of each model (initialized or not)
        """
        return {
            "yolo_model": self.model is not None,
            "plate_detector": self.plate_model is not None,
            "ocr_reader": self.ocr_reader is not None,
        }

    def _initialize_classifiers(self):
        """
        Initialize classifiers (legacy method, now mainly for OCR).
        """
        # if self.color_classifier is None:
        #     self.color_classifier = ColorClassifier()
        # if self.model_classifier is None:
        #     self.model_classifier = ModelClassifier()
        if self.ocr_reader is None:
            self._initialize_ocr_reader()

    def _initialize_excel_file(self):
        """
        Initialize Excel file with headers if it doesn't exist.
        """
        if not os.path.exists(self.excel_output_path):
            df = pd.DataFrame(columns=["Vehicle_ID", "License_Plate", "Direction_Label", "Timestamp"])
            df.to_excel(self.excel_output_path, index=False, engine='openpyxl')
            print(f"Created Excel file: {self.excel_output_path}")

    def _save_to_excel(self, vehicle_id, license_plate, direction_label, timestamp):
        """
        Save vehicle data to Excel file (thread-safe).
        Only saves once per vehicle_id.

        Args:
            vehicle_id: Vehicle track ID
            license_plate: Detected license plate text
            direction_label: Vehicle direction label
            timestamp: Detection timestamp
        """
        try:
            # Check if already saved
            if vehicle_id in self.vehicle_saved_to_excel:
                return
            
            with self._excel_lock:
                # Double-check to avoid race condition
                if vehicle_id in self.vehicle_saved_to_excel:
                    return
                
                # Read existing data
                if os.path.exists(self.excel_output_path):
                    df = pd.read_excel(self.excel_output_path, engine='openpyxl')
                else:
                    df = pd.DataFrame(columns=["Vehicle_ID", "License_Plate", "Direction_Label", "Timestamp"])

                # Append new row
                new_row = {
                    "Vehicle_ID": vehicle_id,
                    "License_Plate": license_plate,
                    "Direction_Label": direction_label,
                    "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S")
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

                # Save to Excel
                df.to_excel(self.excel_output_path, index=False, engine='openpyxl')
                
                # Mark as saved
                self.vehicle_saved_to_excel.add(vehicle_id)
        except Exception as e:
            print(f"Error saving to Excel: {e}")

    def _get_most_detected_plate(self, track_id):
        """
        Get the license plate with highest detection count for a vehicle.
        
        Args:
            track_id: Vehicle track ID
            
        Returns:
            tuple: (plate_text, count) or (None, 0) if no plates detected
        """
        if track_id not in self.vehicle_plate_counts or not self.vehicle_plate_counts[track_id]:
            return None, 0
        
        plate_counts = self.vehicle_plate_counts[track_id]
        # Find plate with maximum count
        most_detected_plate = max(plate_counts.items(), key=lambda x: x[1])
        return most_detected_plate

    def _save_vehicle_if_complete(self, track_id, current_timestamp):
        """
        Save vehicle to Excel if it's complete (no longer detected or after threshold).
        Saves the license plate with highest detection count.
        
        Args:
            track_id: Vehicle track ID
            current_timestamp: Current frame timestamp
        """
        # Skip if already saved
        if track_id in self.vehicle_saved_to_excel:
            return
        
        # Get most detected plate
        plate_text, count = self._get_most_detected_plate(track_id)
        
        if plate_text and count > 0:
            # Get direction and timestamp
            direction_label = self.vehicle_directions.get(track_id, "Unknown")
            timestamp = self.vehicle_last_seen.get(track_id, current_timestamp)
            
            # Save to Excel
            self._save_to_excel(track_id, plate_text, direction_label, timestamp)
            print(f"Vehicle {track_id} saved: {plate_text} (detected {count} times) - {direction_label}")

    def _initialize_plate_detector(self):
        """
        Initialize the license plate detector model.
        Downloads the model if not present.
        """
        # Use centralized initializer in plate_utils
        try:
            self.plate_model = initialize_plate_detector("model/LP_detector.pt")
        except Exception as e:
            print(f"Error loading license plate model: {e}")
            self.plate_model = None

    def _preprocess_plate_image(self, plate_image):
        """
        Preprocess the license plate image for better OCR results.

        Args:
            plate_image (numpy.ndarray): The cropped license plate image.

        Returns:
            numpy.ndarray: Preprocessed image
        """
        return preprocess_plate_image(plate_image)

    def _detect_license_plate(self, vehicle_frame):
        """
        Detect license plate and recognize its text from a vehicle frame.

        Args:
            vehicle_frame (numpy.ndarray): The cropped frame containing the vehicle.

        Returns:
            dict: Dictionary containing license plate text, confidence score, and coordinates.
        """
        # Delegate to plate_utils synchronous detector
        return detect_license_plate_sync(self.plate_model, vehicle_frame, self.ocr_reader, self._model_lock)

    # async def _ocr_attempt(self, ocr_reader, plate_image, cc, ct):
    #     """
    #     Single OCR attempt with rotation parameters.
    #     Wrapped in async to allow parallel execution.
    #     """
    #     # This method is now handled in plate_utils (used by async detector). Keep wrapper for compatibility.
    #     loop = asyncio.get_event_loop()
    #     rotated_image = utils_rotate.deskew(plate_image, cc, ct)
    #     with self._model_lock:
    #         # lp = await loop.run_in_executor(self._executor, lambda: helper.read_plate(ocr_reader, rotated_image))
    #         lp = await loop.run_in_executor(self._executor, lambda: ocr_reader.read_license_plate(rotated_image))
    #     return lp, cc, ct

    async def _detect_license_plate_async(self, vehicle_frame):
        """
        Async version: Detect license plate and recognize its text from a vehicle frame.
        Uses parallel OCR attempts for better performance.

        Args:
            vehicle_frame (numpy.ndarray): The cropped frame containing the vehicle.

        Returns:
            dict: Dictionary containing license plate text, confidence score, and coordinates.
        """
        # Delegate to plate_utils async detector
        return await detect_license_plate_async(self.plate_model, vehicle_frame, self.ocr_reader, self._executor, self._model_lock)

    def _map_direction_to_label(self, direction):
        # Define direction ranges in radians and their corresponding labels
        direction_ranges = {
            (-math.pi / 8, math.pi / 8): "Right",
            (math.pi / 8, 3 * math.pi / 8): "Bottom Right",
            (3 * math.pi / 8, 5 * math.pi / 8): "Bottom",
            (5 * math.pi / 8, 7 * math.pi / 8): "Bottom Left",
            (7 * math.pi / 8, -7 * math.pi / 8): "Left",
            (-7 * math.pi / 8, -5 * math.pi / 8): "Top Left",
            (-5 * math.pi / 8, -3 * math.pi / 8): "Top",
            (-3 * math.pi / 8, -math.pi / 8): "Top Right",
        }
        for angle_range, label in direction_ranges.items():
            if angle_range[0] <= direction <= angle_range[1]:
                return label
        return "Unknown"  # Return "Unknown" if the direction doesn't match any defined range

    def _encode_image_base64(self, image):
        """
        Encode an image as base64.

        Args:
            image (numpy.ndarray): The image to be encoded.

        Returns:
            str: Base64-encoded image.
        """
        _, buffer = cv2.imencode(".jpg", image)
        image_base64 = base64.b64encode(buffer).decode()
        return image_base64

    def _decode_image_base64(self, image_base64):
        """
        Decode a base64-encoded image.

        Args:
            image_base64 (str): Base64-encoded image data.

        Returns:
            numpy.ndarray or None: Decoded image as a numpy array or None if decoding fails.
        """
        try:
            image_data = base64.b64decode(image_base64)
            image_np = np.frombuffer(image_data, dtype=np.uint8)
            image = cv2.imdecode(image_np, flags=cv2.IMREAD_COLOR)
            return image
        except Exception as e:
            return None

    def _increase_brightness(self, image, factor=1.5):
        """
        Increases the brightness of an image by multiplying its pixels by a factor.

        :param image: The input image in numpy array format.
        :param factor: The brightness increase factor. A value greater than 1 will increase brightness.
        :return: The image with increased brightness.
        """
        brightened_image = cv2.convertScaleAbs(image, alpha=factor, beta=0)
        return brightened_image

    def _convert_meters_per_second_to_kmph(self, meters_per_second):
        # 1 m/s is approximately 3.6 km/h
        kmph = meters_per_second * 3.6
        return kmph

    def _draw_plate_text_corner(self, frame, plates_dict):
        """
        Draw detected license plates at the top-left corner of the frame.
        Each vehicle's plate is displayed on a separate line.
        
        Args:
            frame (numpy.ndarray): Frame to draw on
            plates_dict (dict): Dictionary of {track_id: plate_text}
        """
        if not plates_dict:
            return frame
        
        # Draw background rectangle for text
        h, w = frame.shape[:2]
        num_plates = len(plates_dict)
        text_height = 30
        padding = 10
        bg_height = num_plates * text_height + padding * 2
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (10, 10),
            (400, bg_height),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw text for each detected plate
        y_offset = padding + 25
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        
        for idx, (track_id, plate_text) in enumerate(plates_dict.items()):
            if plate_text and plate_text != "unknown":
                text = f"Vehicle {track_id}: {plate_text}"
                cv2.putText(
                    frame,
                    text,
                    (15, y_offset + idx * text_height),
                    font,
                    font_scale,
                    (0, 255, 0),  # Green color
                    font_thickness
                )
        
        return frame

    async def _process_plate_async_background(self, track_id, vehicle_frame, direction_label=None, timestamp=None):
        """
        Process license plate detection in background without blocking.
        Tracks license plate detection counts (doesn't save to Excel immediately).
        
        Args:
            track_id: Vehicle track ID
            vehicle_frame: Cropped vehicle frame
            direction_label: Vehicle direction label (if available)
            timestamp: Detection timestamp
        """
        try:
            # Skip if already saved to Excel
            if track_id in self.vehicle_saved_to_excel:
                return
            
            license_plate_info = await self._detect_license_plate_async(vehicle_frame)
            plate_text = license_plate_info.get("text") if license_plate_info else None
            
            if plate_text and plate_text != "unknown":
                # Update most recent plate for display
                self.vehicle_plates[track_id] = plate_text
                
                # Increment detection count for this plate
                self.vehicle_plate_counts[track_id][plate_text] += 1
                
                # Get direction if available
                if direction_label:
                    self.vehicle_directions[track_id] = direction_label
                elif track_id not in self.vehicle_directions:
                    self.vehicle_directions[track_id] = "Unknown"
                
                # Update last seen timestamp
                if timestamp:
                    self.vehicle_last_seen[track_id] = timestamp
                
                # Send Telegram notification only once per vehicle ID
                if track_id not in self.vehicle_telegram_sent:
                    print(f"Sending Telegram notification for vehicle {track_id}...")
                    # send_notify_to_telegram is synchronous; run it in a thread so
                    # awaiting here does not try to await a non-awaitable object.
                    await asyncio.to_thread(send_notify_to_telegram, plate_text, direction_label, timestamp)
                    self.vehicle_telegram_sent.add(track_id)
        except Exception as e:
            print(f"Background plate detection error for vehicle {track_id}: {e}")

    def _process_plate_background_sync(self, track_id, vehicle_frame, direction_label=None, timestamp=None):
        """
        Sync wrapper for background plate processing using ThreadPoolExecutor.
        Tracks license plate detection counts (doesn't save to Excel immediately).
        
        Args:
            track_id: Vehicle track ID
            vehicle_frame: Cropped vehicle frame
            direction_label: Vehicle direction label (if available)
            timestamp: Detection timestamp
        """
        try:
            # Skip if already saved to Excel
            if track_id in self.vehicle_saved_to_excel:
                return
            
            # Use sync version for simplicity in streaming mode
            license_plate_info = self._detect_license_plate(vehicle_frame)
            plate_text = license_plate_info.get("text") if license_plate_info else None
            
            if plate_text and plate_text != "unknown":
                # Update most recent plate for display
                self.vehicle_plates[track_id] = plate_text
                
                # Increment detection count for this plate
                self.vehicle_plate_counts[track_id][plate_text] += 1
                
                # Get direction if available, otherwise use cached or "Unknown"
                if direction_label:
                    self.vehicle_directions[track_id] = direction_label
                elif track_id not in self.vehicle_directions:
                    self.vehicle_directions[track_id] = "Unknown"
                
                # Update last seen timestamp
                if timestamp:
                    self.vehicle_last_seen[track_id] = timestamp

                # Send Telegram notification only once per vehicle ID
                if track_id not in self.vehicle_telegram_sent:
                    print(f"Sending Telegram notification for vehicle {track_id}...")
                    send_notify_to_telegram(plate_text, direction_label, timestamp)
                    self.vehicle_telegram_sent.add(track_id)
        except Exception as e:
            print(f"Background plate detection error for vehicle {track_id}: {e}")

    def process_frame_streaming(self, frame, frame_timestamp):
        """
        Optimized frame processing for streaming: Fast detection, background OCR.
        Only shows license plates in corner, no bounding boxes on vehicles.
        Perfect for real-time camera streaming.

        Args:
            frame (numpy.ndarray): Input frame for processing.
            frame_timestamp (datetime): Timestamp of the frame.

        Returns:
            numpy.ndarray: Frame with license plates displayed in corner (no vehicle boxes).
        """
        # Quick vehicle detection (no OCR blocking)
        # brightened_frame = self._increase_brightness(frame)

        results = self.model.track(
            frame, persist=True, tracker="bytetrack.yaml"
        )
        # print("Results: ", results)

        # display_frame = frame.copy()

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
            current_track_ids = set(track_ids)
            
            # Update tracking history and calculate directions
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                if w < 250 or w > 350:
                    continue
                 # Save brightened frame to PNG file
                try:
                    timestamp_str = frame_timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    # Extract vehicle frame for OCR
                    vehicle_frame = frame[
                        int(y - h / 2 - 10) : int(y + h / 2 + 10), int(x - w / 2) : int(x + w / 2)
                    ]
                    filename = f"screenshots/vehicle_frame_{timestamp_str}.png"
                    cv2.imwrite(filename, vehicle_frame)
                except Exception as e:
                    print(f"Error saving brightened frame: {e}")
        
                # Update last seen
                self.vehicle_last_seen[track_id] = frame_timestamp
                # Reset missing frame count when vehicle is detected
                self.vehicle_missing_frames[track_id] = 0
                
                # Update tracking history
                if track_id not in self.track_history:
                    self.track_history[track_id] = []
                track = self.track_history[track_id]
                track.append((float(x), float(y)))
                max_history_length = 30
                if len(track) > max_history_length:
                    track.pop(0)
                
                # Update timestamps
                if track_id not in self.vehicle_timestamps:
                    self.vehicle_timestamps[track_id] = {
                        "timestamps": [],
                        "positions": [],
                    }
                self.vehicle_timestamps[track_id]["timestamps"].append(frame_timestamp)
                self.vehicle_timestamps[track_id]["positions"].append((x, y))
                
                # Calculate direction_label if enough data
                timestamps = self.vehicle_timestamps[track_id]["timestamps"]
                positions = self.vehicle_timestamps[track_id]["positions"]
                direction_label = "Unknown"
                
                if len(positions) >= 2:
                    initial_x, initial_y = positions[0]
                    final_x, final_y = positions[-1]
                    direction = math.atan2(final_y - initial_y, final_x - initial_x)
                    direction_label = self._map_direction_to_label(direction)
                    self.vehicle_directions[track_id] = direction_label
                

                
                # Submit OCR to background thread (non-blocking)
                # Continue processing to get multiple detections for better accuracy
                if vehicle_frame.size > 0:
                    # Only process if not already saved to Excel
                    if track_id not in self.vehicle_saved_to_excel:
                        # Process every frame to accumulate detection counts
                        # This allows us to choose the most frequently detected plate
                        self._executor.submit(
                            self._process_plate_background_sync,
                            track_id,
                            vehicle_frame.copy(),  # Copy to avoid frame modification issues
                            direction_label,
                            frame_timestamp
                        )
        
        # Update missing frame counts for vehicles not detected in this frame
        all_tracked_ids = set(self.vehicle_last_seen.keys())
        missing_ids = all_tracked_ids - current_track_ids
        
        for track_id in missing_ids:
            # Increment missing frame count
            if track_id not in self.vehicle_missing_frames:
                self.vehicle_missing_frames[track_id] = 0
            self.vehicle_missing_frames[track_id] += 1
            
            # Save vehicle if missing for 10 consecutive frames
            if (track_id not in self.vehicle_saved_to_excel and 
                self.vehicle_missing_frames[track_id] >= 10):
                self._save_vehicle_if_complete(track_id, frame_timestamp)
            
        # Draw detected plates at corner (from previous detections)
        # display_frame = self._draw_plate_text_corner(display_frame, self.vehicle_plates)
        
        return frame

    async def process_frame_streaming_async(self, frame, frame_timestamp):
        """
        Async version of streaming processing.
        Faster than sync version for high-frequency streaming.

        Args:
            frame (numpy.ndarray): Input frame for processing.
            frame_timestamp (datetime): Timestamp of the frame.

        Returns:
            numpy.ndarray: Frame with license plates displayed in corner.
        """
        # Quick vehicle detection (async)
        loop = asyncio.get_event_loop()
        brightened_frame = self._increase_brightness(frame)
        results = await loop.run_in_executor(
            self._executor,
            lambda: self.model.track(brightened_frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        )

        display_frame = frame.copy()

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
            current_track_ids = set(track_ids)
            
            # Update tracking history and calculate directions
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                
                # Update last seen
                self.vehicle_last_seen[track_id] = frame_timestamp
                # Reset missing frame count when vehicle is detected
                self.vehicle_missing_frames[track_id] = 0
                
                # Update tracking history
                if track_id not in self.track_history:
                    self.track_history[track_id] = []
                track = self.track_history[track_id]
                track.append((float(x), float(y)))
                max_history_length = 30
                if len(track) > max_history_length:
                    track.pop(0)
                
                # Update timestamps
                if track_id not in self.vehicle_timestamps:
                    self.vehicle_timestamps[track_id] = {
                        "timestamps": [],
                        "positions": [],
                    }
                self.vehicle_timestamps[track_id]["timestamps"].append(frame_timestamp)
                self.vehicle_timestamps[track_id]["positions"].append((x, y))
                
                # Calculate direction_label if enough data
                timestamps = self.vehicle_timestamps[track_id]["timestamps"]
                positions = self.vehicle_timestamps[track_id]["positions"]
                direction_label = "Unknown"
                
                if len(positions) >= 2:
                    initial_x, initial_y = positions[0]
                    final_x, final_y = positions[-1]
                    direction = math.atan2(final_y - initial_y, final_x - initial_x)
                    direction_label = self._map_direction_to_label(direction)
                    self.vehicle_directions[track_id] = direction_label
                
                # Extract vehicle frame for OCR
                vehicle_frame = frame[
                    int(y - h / 2) : int(y + h / 2), int(x - w / 2) : int(x + w / 2)
                ]
                
                if vehicle_frame.size > 0:
                    # Only process if not already saved to Excel
                    if track_id not in self.vehicle_saved_to_excel:
                        # Process every frame to accumulate detection counts
                        # This allows us to choose the most frequently detected plate
                        # Fire async task, don't wait
                        asyncio.create_task(
                            self._process_plate_async_background(track_id, vehicle_frame, direction_label, frame_timestamp)
                        )
        
        # Update missing frame counts for vehicles not detected in this frame
        all_tracked_ids = set(self.vehicle_last_seen.keys())
        missing_ids = all_tracked_ids - current_track_ids
        
        for track_id in missing_ids:
            # Increment missing frame count
            if track_id not in self.vehicle_missing_frames:
                self.vehicle_missing_frames[track_id] = 0
            self.vehicle_missing_frames[track_id] += 1
            
            # Save vehicle if missing for 10 consecutive frames
            if (track_id not in self.vehicle_saved_to_excel and 
                self.vehicle_missing_frames[track_id] >= 10):
                self._save_vehicle_if_complete(track_id, frame_timestamp)
        
        # Draw plates from cache (previous detections)
        display_frame = self._draw_plate_text_corner(display_frame, self.vehicle_plates)
        
        return display_frame

    def process_video_streaming(self, video_path, display_window=True, max_reconnect_attempts=10, reconnect_delay=1):
        """
        Process video/camera stream with optimized performance and auto-reconnect.
        Only shows license plates in corner, no vehicle bounding boxes.
        For .mp4 files: plays once and exits (no replay).
        For streams (RTSP, camera): auto-reconnect enabled.
        
        Args:
            video_path (str or int): Path to video file or camera index (0 for webcam), or RTSP URL
            display_window (bool): Whether to display the video window
            max_reconnect_attempts (int): Maximum number of reconnect attempts (0 for infinite)
            reconnect_delay (int): Delay in seconds between reconnect attempts
        """
        # Check if this is an MP4 file (play once, no replay)
        is_mp4_file = isinstance(video_path, str) and video_path.lower().endswith('.mp4')
        
        def create_capture(video_path):
            """Create VideoCapture with optimized settings for RTSP streams."""
            cap = cv2.VideoCapture(video_path)
            
            # Optimize for RTSP streams (reduce buffering, improve latency)
            if isinstance(video_path, str) and video_path.startswith('rtsp://'):
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer to reduce latency
                cap.set(cv2.CAP_PROP_FPS, 30)  # Set expected FPS
                # Set timeout for RTSP (in milliseconds)
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
            
            return cap
        
        cap = None
        consecutive_failures = 0
        max_consecutive_failures = 30  # Allow some consecutive failures before reconnect
        
        while True:
            # Create or recreate capture
            if cap is None or not cap.isOpened():
                time.sleep(reconnect_delay)
                
                # Release old capture if exists
                if cap is not None:
                    cap.release()
                
                cap = create_capture(video_path)
                
                if not cap.isOpened():
                    print(f"Không thể mở camera/video stream: {video_path}")
                    consecutive_failures += 1
                    if consecutive_failures >= 5:
                        print("Không thể kết nối sau nhiều lần thử. Kiểm tra đường dẫn RTSP hoặc kết nối mạng.")
                        break
                    # For MP4 files, don't retry - just exit
                    if is_mp4_file:
                        break
                    continue
                
                print("Đã kết nối camera thành công!")
                consecutive_failures = 0
            
            try:
                success, frame = cap.read()
                
                if not success or frame is None:
                    consecutive_failures += 1
                    
                    # For MP4 files, exit immediately when video ends
                    if is_mp4_file:
                        print("Video playback completed.")
                        break
                    
                    if consecutive_failures >= max_consecutive_failures:
                        print(f"Mất kết nối sau {consecutive_failures} frame liên tiếp. Đang thử kết nối lại...")
                        cap.release()
                        cap = None
                        consecutive_failures = 0
                        continue
                    
                    # If just a few failures, try to continue reading
                    continue
                
                # Reset failure counter on success
                consecutive_failures = 0
                
                # Optionally resize the frame to reduce processing cost
                if self.stream_frame_size and frame is not None:
                    try:
                        frame = cv2.resize(frame, self.stream_frame_size, interpolation=cv2.INTER_AREA)
                    except Exception:
                        pass

                timestamp = datetime.now()

                # Fast processing (no blocking OCR)
                try:
                    display_frame = self.process_frame_streaming(frame, timestamp)
                except Exception as e:
                    print(f"Lỗi xử lý frame: {e}")
                    continue

                if display_window:
                    cv2.imshow("Vehicle Detection - Streaming Mode", display_frame)

                # Break on 'q' key
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                    
            except Exception as e:
                # Handle codec errors (like HEVC errors) gracefully
                error_msg = str(e).lower()
                if 'hevc' in error_msg or 'codec' in error_msg or 'ref with poc' in error_msg:
                    # Codec error - just skip this frame and continue
                    print(f"Lỗi codec (bỏ qua frame): {e}")
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        print("Quá nhiều lỗi codec liên tiếp. Đang thử kết nối lại...")
                        cap.release()
                        cap = None
                        consecutive_failures = 0
                    continue
                else:
                    # Other errors - might need reconnect
                    print(f"Lỗi không mong muốn: {e}")
                    consecutive_failures += 1
                    if consecutive_failures >= 10:
                        print("Đang thử kết nối lại sau lỗi...")
                        cap.release()
                        cap = None
                        consecutive_failures = 0
        
        # Cleanup
        try:
            # Save any remaining vehicles before closing
            final_timestamp = datetime.now()
            for track_id in self.vehicle_last_seen.keys():
                if track_id not in self.vehicle_saved_to_excel:
                    self._save_vehicle_if_complete(track_id, final_timestamp)
            
            if cap is not None:
                cap.release()
            if display_window:
                cv2.destroyAllWindows()
        except Exception as e:
            print(f"Lỗi khi cleanup: {e}")

    async def process_camera_stream_async(self, camera_index=0):
        """
        Async camera streaming with optimized performance.
        
        Args:
            camera_index (int): Camera index (default 0 for webcam)
            
        Yields:
            numpy.ndarray: Frames with license plates displayed in corner
        """
        cap = cv2.VideoCapture(camera_index)
        
        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                
                timestamp = datetime.now()
                display_frame = await self.process_frame_streaming_async(frame, timestamp)
                yield display_frame
        finally:
            # Save any remaining vehicles before closing
            final_timestamp = datetime.now()
            for track_id in self.vehicle_last_seen.keys():
                if track_id not in self.vehicle_saved_to_excel:
                    self._save_vehicle_if_complete(track_id, final_timestamp)
            
            cap.release()

    def cleanup(self):
        """
        Clean up resources (ThreadPoolExecutor).
        Save any remaining vehicles that haven't been saved yet.
        Call this when done using the tracker.
        """
        # Wait for executor to finish any pending tasks
        if hasattr(self, '_executor') and self._executor:
            self._executor.shutdown(wait=True)
        
        # Save any remaining vehicles that haven't been saved
        final_timestamp = datetime.now()
        for track_id in self.vehicle_last_seen.keys():
            if track_id not in self.vehicle_saved_to_excel:
                self._save_vehicle_if_complete(track_id, final_timestamp)
        
        # Clear plate cache
        self.vehicle_plates.clear()
