import asyncio
import cv2
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

from VehicleDetectionTracker.function import utils_rotate, helper


def _ensure_log_dir():
    """Ensure logs directory exists."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    return log_dir


def _log(message):
    """Print log message with datetime timestamp and save to file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"

    # Print to console
    print(log_message)

    # Write to file
    try:
        log_dir = _ensure_log_dir()
        log_file = log_dir / f"plate_{datetime.now().strftime('%Y-%m-%d')}.log"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_message + "\n")
    except Exception as e:
        # Don't fail if logging fails, just print error
        print(f"Error writing to log file: {e}")


def initialize_plate_detector(model_path="model/LP_detector.pt", device=None):
    """Load and return the license plate detector (YOLOv8/YOLOv5) or None on error.

    Args:
        model_path (str): Path to the model file (YOLOv8 or YOLOv5 format)
        device (str): Device to use ('cuda' or 'cpu'). If None, auto-detect.
    """
    try:
        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda:0":
            # Convert 'cuda:0' to 'cuda' for compatibility
            device = "cuda"
        
        # Verify model file exists
        model_file = Path(model_path)
        if not model_file.exists():
            _log(f"[PLATE_DETECTOR] ⚠ Model file not found: {model_path}, trying alternative path...")
            # Try alternative path with default name
            model_file = Path("model/license_plate_detector.pt")
            if model_file.exists():
                model_path = str(model_file)
            else:
                raise FileNotFoundError(f"Model file not found at {model_path} or model/license_plate_detector.pt")

        # Try loading as YOLOv8 model first
        _log(f"[PLATE_DETECTOR] Loading model from {model_path} on device={device}...")
        try:
            plate_model = YOLO(model_path)
            plate_model.to(device)
            
            if device == "cuda" and torch.cuda.is_available():
                _log(
                    f"[PLATE_DETECTOR] ✓ YOLOv8 model loaded and moved to GPU: {torch.cuda.get_device_name(0)}"
                )
            else:
                _log("[PLATE_DETECTOR] ✓ YOLOv8 model loaded (using CPU)")
            
            return plate_model
            
        except Exception as yolov8_error:
            # Fallback: Try loading as YOLOv5 model using torch.hub
            _log(f"[PLATE_DETECTOR] ⚠ YOLOv8 loading failed: {str(yolov8_error)[:100]}...")
            _log(f"[PLATE_DETECTOR] Attempting YOLOv5 fallback loading...")
            
            try:
                # Add yolov5 directory to sys.path for local loading
                import sys
                yolov5_path = Path(__file__).parent.parent / "yolov5"
                if str(yolov5_path) not in sys.path:
                    sys.path.insert(0, str(yolov5_path))
                
                # Try torch.hub.load for YOLOv5
                plate_model = torch.hub.load(
                    "ultralytics/yolov5",
                    "custom",
                    path=model_path,
                    force_reload=False,
                    device=device,
                )
                
                _log(
                    f"[PLATE_DETECTOR] ✓ YOLOv5 model loaded (fallback method)"
                )
                return plate_model
                
            except Exception as yolov5_error:
                _log(f"[PLATE_DETECTOR] ⚠ YOLOv5 fallback also failed: {yolov5_error}")
                raise Exception(f"Failed to load model with both YOLOv8 and YOLOv5 methods. YOLOv8 error: {yolov8_error}, YOLOv5 error: {yolov5_error}")
    
    except Exception as e:
        _log(f"Error loading license plate model: {e}")
        return None


def preprocess_plate_image(plate_image):
    try:
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        denoised = cv2.fastNlMeansDenoising(thresh)
        kernel = np.ones((1, 1), np.uint8)
        dilated = cv2.dilate(denoised, kernel, iterations=1)
        return dilated
    except Exception as e:
        _log(f"Error in plate image preprocessing: {e}")
        return plate_image


def _sync_plate_inference(plate_model, vehicle_frame, model_lock, size=None):
    """Run plate model synchronously in a thread-safe way and return results or None (YOLOv8)."""
    if plate_model is None:
        return None

    # Check if vehicle_frame is valid
    if vehicle_frame is None or vehicle_frame.size == 0:
        _log("[PLATE_INFERENCE] ❌ vehicle_frame is None or empty")
        return None

    # Load config if size not specified
    if size is None:
        try:
            from VehicleDetectionTracker.config_loader import get_plate_detection_config

            plate_config = get_plate_detection_config()
            size = plate_config.get(
                "image_size", 640
            )  # Default: 640 for YOLOv8
        except:
            size = 640

    try:
        with model_lock:
            # YOLOv8: inference with imgsz parameter
            results = plate_model.predict(vehicle_frame, imgsz=size, verbose=False)
            # Returns a list of Results objects, take the first one
            return results[0] if results else None
    except Exception as e:
        _log(f"[PLATE_INFERENCE] ❌ Error during inference with imgsz={size}: {e}")
        # Fallback: try without size parameter (use default)
        try:
            with model_lock:
                results = plate_model.predict(vehicle_frame, verbose=False)
                return results[0] if results else None
        except Exception as e2:
            _log(f"[PLATE_INFERENCE] ❌ Fallback inference also failed: {e2}")
            return None


def detect_license_plate_sync(
    plate_model,
    vehicle_frame,
    ocr_reader,
    model_lock,
    timestamp_str,
    vehicle_dir="screenshots",
):
    """Detect license plate synchronously with detailed logging for debugging (YOLOv8)."""
    try:
        _log("[PLATE_DETECT] Bắt đầu detect license plate")

        # Check plate model
        if plate_model is None:
            _log("[PLATE_DETECT] ❌ plate_model is None, return None")
            return {"text": None, "count": None}

        _log(
            f"[PLATE_DETECT] Vehicle frame shape: {vehicle_frame.shape if vehicle_frame is not None else 'None'}"
        )

        # Run plate detection inference
        _log("[PLATE_DETECT] Đang chạy plate model inference...")
        results = _sync_plate_inference(plate_model, vehicle_frame, model_lock)

        if results is None:
            _log("[PLATE_DETECT] ❌ Inference results is None")
            return {"text": None, "count": None}

        # YOLOv8: Use boxes attribute instead of pred
        boxes = results.boxes
        num_detections = len(boxes) if boxes is not None else 0
        
        if num_detections == 0:
            _log("[PLATE_DETECT] ❌ Không có detection nào")
            return {"text": None, "count": 0}

        _log(f"[PLATE_DETECT] Raw detections: {num_detections}")
        
        # Log all detections
        for i, box in enumerate(boxes):
            bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            conf = float(box.conf[0])
            _log(f"[PLATE_DETECT] Detection {i}: bbox={bbox}, confidence={conf:.3f}")

        # Get best detection (highest confidence)
        best_box = boxes[0]  # boxes are sorted by confidence by default
        confidence = float(best_box.conf[0])
        x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())

        _log(
            f"[PLATE_DETECT] Best detection: bbox=({x1},{y1},{x2},{y2}), confidence={confidence:.3f}"
        )

        # Check plate length
        length_plate = x2 - x1
        height_plate = y2 - y1
        _log(
            f"[PLATE_DETECT] Plate dimensions: width={length_plate}, height={height_plate}"
        )

        # Load config for plate detection
        try:
            from VehicleDetectionTracker.config_loader import (
                get_plate_detection_config,
            )

            plate_config = get_plate_detection_config()
            min_width = plate_config.get("min_width", 40)
            min_height = plate_config.get("min_height", 20)
            min_confidence = plate_config.get("min_confidence", 0.25)
        except:
            min_width = 40
            min_height = 20
            min_confidence = 0.25

        # Check confidence threshold
        if confidence < min_confidence:
            _log(
                f"[PLATE_DETECT] ❌ Confidence quá thấp ({confidence:.3f} < {min_confidence}), return None"
            )
            return {"text": None, "count": num_detections}

        if length_plate < min_width:
            _log(
                f"[PLATE_DETECT] ❌ Plate quá nhỏ (width={length_plate} < {min_width}), return None"
            )
            return {"text": None, "count": num_detections}

        if height_plate < min_height:
            _log(
                f"[PLATE_DETECT] ❌ Plate quá nhỏ (height={height_plate} < {min_height}), return None"
            )
            return {"text": None, "count": num_detections}

        # Validate bbox coordinates against frame dimensions to avoid index out of bounds
        frame_height, frame_width = vehicle_frame.shape[:2]
        x1 = max(0, min(x1, frame_width - 1))
        y1 = max(0, min(y1, frame_height - 1))
        x2 = max(x1 + 1, min(x2, frame_width))
        y2 = max(y1 + 1, min(y2, frame_height))

        # Extract plate image
        plate_image = vehicle_frame[y1:y2, x1:x2]
        if plate_image.size == 0:
            _log("[PLATE_DETECT] ❌ Plate image size = 0, return None")
            return {"text": None, "count": num_detections}

        _log(f"[PLATE_DETECT] ✓ Extracted plate image shape: {plate_image.shape}")

        # Save plate image
        filename = f"{vehicle_dir}/license_frame_{timestamp_str}.png"
        cv2.imwrite(filename, plate_image)

        # Check OCR reader
        if ocr_reader is None:
            _log("[PLATE_DETECT] ⚠ OCR reader is None, return bbox only")
            return {"text": None, "count": num_detections}

        # Try OCR with different deskew directions
        _log("[PLATE_DETECT] Bắt đầu OCR với các hướng xoay khác nhau...")
        lp = "unknown"
        for direction in [-1, 1, 0]:  # left, right, auto
            for center_thres in [0, 1]:
                try:
                    deskewed_image = utils_rotate.deskew(
                        plate_image, direction, center_thres
                    )
                    _log(
                        f"[PLATE_DETECT] OCR attempt: direction={direction}, center_thres={center_thres}, deskewed_shape={deskewed_image.shape if deskewed_image is not None else 'None'}"
                    )

                    with model_lock:
                        lp = ocr_reader.read_license_plate(deskewed_image)

                    _log(
                        f"[PLATE_DETECT] OCR result (direction={direction}, center_thres={center_thres}): '{lp}'"
                    )

                    if lp != "unknown" and lp is not None:
                        _log(
                            f"[PLATE_DETECT] ✓ Tìm thấy biển số: '{lp}' (direction={direction}, center_thres={center_thres})"
                        )
                        return {"text": lp, "count": num_detections}
                except Exception as ocr_error:
                    _log(
                        f"[PLATE_DETECT] ⚠ OCR error (direction={direction}, center_thres={center_thres}): {ocr_error}"
                    )

        _log(
            f"[PLATE_DETECT] ⚠ Không đọc được biển số sau tất cả các lần thử, return: '{lp}'"
        )
        return {"text": lp, "count": num_detections}
    except Exception as e:
        _log(f"[PLATE_DETECT] ❌ ERROR in license plate detection: {e}")
        import traceback

        _log(f"[PLATE_DETECT] Traceback: {traceback.format_exc()}")
        return {"text": None, "count": None}
