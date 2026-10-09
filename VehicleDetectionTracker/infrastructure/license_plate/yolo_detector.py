"""Ultralytics YOLO adapter for license-plate detection."""

from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

import torch
from ultralytics import YOLO

from VehicleDetectionTracker.domain.license_plate.entities import (
    PlateBoundingBox,
    PlateDetection,
)
from VehicleDetectionTracker.logging_utils import log as shared_log


class YOLOPlateDetector:
    def __init__(
        self,
        model: Any,
        model_lock: Any = None,
        image_size: int = 640,
        log_func: Optional[Callable[[str], None]] = None,
    ) -> None:
        import threading

        self.model = model
        self.model_lock = model_lock or threading.Lock()
        self.image_size = image_size
        self._log = log_func or shared_log

    def predict_raw(self, image: Any, image_size: Optional[int] = None) -> Any:
        if self.model is None or image is None or getattr(image, "size", 0) == 0:
            return None

        size = image_size or self.image_size
        try:
            with self.model_lock:
                results = self.model.predict(image, imgsz=size, verbose=False)
            return results[0] if results else None
        except Exception as error:
            self._log(
                f"[PLATE_INFERENCE] ❌ Error during inference with imgsz={size}: {error}"
            )
            try:
                with self.model_lock:
                    results = self.model.predict(image, verbose=False)
                return results[0] if results else None
            except Exception as fallback_error:
                self._log(
                    f"[PLATE_INFERENCE] ❌ Fallback inference also failed: {fallback_error}"
                )
                return None

    def detect(self, image: Any, image_size: Optional[int] = None) -> List[PlateDetection]:
        result = self.predict_raw(image, image_size)
        return self._convert_result(result)

    def detect_batch(
        self,
        images: Sequence[Any],
        image_size: Optional[int] = None,
    ) -> List[List[PlateDetection]]:
        if not images:
            return []
        if self.model is None:
            return [[] for _ in images]
        if any(image is None or getattr(image, "size", 0) == 0 for image in images):
            raise ValueError("True-batch input contains an empty image")

        size = image_size or self.image_size
        try:
            with self.model_lock:
                results = self.model.predict(
                    source=list(images),
                    imgsz=size,
                    batch=len(images),
                    verbose=False,
                )
            if results is None or len(results) != len(images):
                raise RuntimeError(
                    f"YOLO returned {0 if results is None else len(results)} results "
                    f"for {len(images)} input images"
                )
            return [self._convert_result(result) for result in results]
        except Exception as error:
            self._log(f"[PLATE_INFERENCE] ❌ True-batch inference failed: {error}")
            raise

    @staticmethod
    def _convert_result(result: Any) -> List[PlateDetection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        detections = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append(
                PlateDetection(
                    PlateBoundingBox(x1, y1, x2, y2),
                    float(box.conf[0]),
                )
            )
        return detections


def load_plate_detector(model_path: str = "model/LP_detector.pt", device: Optional[str] = None) -> Any:
    """Load YOLOv8, preserving the existing YOLOv5 fallback behavior."""
    log = shared_log
    try:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda:0":
            device = "cuda"

        model_file = Path(model_path)
        if not model_file.exists():
            log(
                f"[PLATE_DETECTOR] ⚠ Model file not found: {model_path}, trying alternative path..."
            )
            model_file = Path("model/license_plate_detector.pt")
            if model_file.exists():
                model_path = str(model_file)
            else:
                raise FileNotFoundError(
                    f"Model file not found at {model_path} or model/license_plate_detector.pt"
                )

        log(f"[PLATE_DETECTOR] Loading model from {model_path} on device={device}...")
        try:
            plate_model = YOLO(model_path)
            plate_model.to(device)
            if device == "cuda" and torch.cuda.is_available():
                log(
                    f"[PLATE_DETECTOR] ✓ YOLOv8 model loaded and moved to GPU: {torch.cuda.get_device_name(0)}"
                )
            else:
                log("[PLATE_DETECTOR] ✓ YOLOv8 model loaded (using CPU)")
            return plate_model
        except Exception as yolov8_error:
            log(f"[PLATE_DETECTOR] ⚠ YOLOv8 loading failed: {str(yolov8_error)[:100]}...")
            log("[PLATE_DETECTOR] Attempting YOLOv5 fallback loading...")
            try:
                import sys

                yolov5_path = Path(__file__).parents[2].parent / "yolov5"
                if str(yolov5_path) not in sys.path:
                    sys.path.insert(0, str(yolov5_path))
                plate_model = torch.hub.load(
                    "ultralytics/yolov5",
                    "custom",
                    path=model_path,
                    force_reload=False,
                    device=device,
                )
                log("[PLATE_DETECTOR] ✓ YOLOv5 model loaded (fallback method)")
                return plate_model
            except Exception as yolov5_error:
                log(f"[PLATE_DETECTOR] ⚠ YOLOv5 fallback also failed: {yolov5_error}")
                raise Exception(
                    "Failed to load model with both YOLOv8 and YOLOv5 methods. "
                    f"YOLOv8 error: {yolov8_error}, YOLOv5 error: {yolov5_error}"
                )
    except Exception as error:
        log(f"Error loading license plate model: {error}")
        return None