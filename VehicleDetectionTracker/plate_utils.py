import asyncio
import cv2
import numpy as np
import torch
from datetime import datetime
from pathlib import Path

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
    """Load and return the license plate detector (yolov5 custom) or None on error.

    Args:
        model_path (str): Path to the model file
        device (str): Device to use ('cuda' or 'cpu'). If None, auto-detect.
    """
    try:
        # Tự động detect device nếu không được chỉ định
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif device == "cuda":
            # Convert 'cuda' to 'cuda:0' for YOLOv5 compatibility
            device = "cuda:0"

        plate_model = torch.hub.load(
            "yolov5",
            "custom",
            path=model_path,
            force_reload=False,
            source="local",
            device=device,  # Chỉ định device
        )

        # Chuyển model sang GPU nếu cần
        if "cuda" in device and torch.cuda.is_available():
            plate_model.to(device)
            _log(
                f"[PLATE_DETECTOR] ✓ Model loaded và chuyển sang GPU: {torch.cuda.get_device_name(0)}"
            )
        else:
            _log("[PLATE_DETECTOR] ✓ Model loaded (sử dụng CPU)")

        return plate_model
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
    """Run plate model synchronously in a thread-safe way and return results or None."""
    if plate_model is None:
        return None

    # Check if vehicle_frame is valid
    if vehicle_frame is None or vehicle_frame.size == 0:
        _log("[PLATE_INFERENCE] ❌ vehicle_frame is None or empty")
        return None

    # Load config nếu size không được chỉ định
    if size is None:
        try:
            from VehicleDetectionTracker.config_loader import get_plate_detection_config

            plate_config = get_plate_detection_config()
            size = plate_config.get(
                "image_size", 1280
            )  # Default: 1280 cho license plate
        except:
            size = 640

    try:
        with model_lock:
            # YOLOv5 AutoShape model: set imgsz attribute before calling
            # Save original imgsz to restore later (if needed)
            original_imgsz = getattr(plate_model, "imgsz", None)
            plate_model.imgsz = size
            result = plate_model(vehicle_frame)
            # Restore original imgsz if it existed
            if original_imgsz is not None:
                plate_model.imgsz = original_imgsz
            return result
    except Exception as e:
        _log(f"[PLATE_INFERENCE] ❌ Error during inference with imgsz={size}: {e}")
        # Fallback: thử gọi không có size parameter (use default)
        try:
            with model_lock:
                return plate_model(vehicle_frame)
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
    """Detect license plate synchronously with detailed logging for debugging."""
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

        # Log all raw detections for debugging
        pred = results.pred[0]
        num_detections = pred.shape[0]
        if num_detections == 0:
            _log("[PLATE_DETECT] ❌ Không có detection nào (pred shape = 0)")
            return {"text": None, "count": None}

        _log(f"[PLATE_DETECT] Raw detections: {num_detections}")
        
        for i in range(num_detections):
            bbox = pred[i][:4].tolist()
            conf = float(pred[i][4])
            _log(f"[PLATE_DETECT] Detection {i}: bbox={bbox}, confidence={conf:.3f}")

        pred = results.pred[0]
        num_detections = pred.shape[0]
        _log(f"[PLATE_DETECT] ✓ Tìm thấy {num_detections} detection(s)")

        # Get best detection (highest confidence)
        best_det_idx = pred[:, 4].argmax()
        best_det = pred[best_det_idx]
        confidence = float(best_det[4])
        x1, y1, x2, y2 = map(int, best_det[:4].tolist())

        _log(
            f"[PLATE_DETECT] Best detection: bbox=({x1},{y1},{x2},{y2}), confidence={confidence:.3f}"
        )

        # Check plate length
        length_plate = x2 - x1
        height_plate = y2 - y1
        _log(
            f"[PLATE_DETECT] Plate dimensions: width={length_plate}, height={height_plate}"
        )

        # Load config cho plate detection
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

        # Validate bbox coordinates against frame dimensions để tránh index out of bounds
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
        # Save license frame in the same track_id folder
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
