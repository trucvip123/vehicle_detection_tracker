"""License-plate infrastructure adapters."""

from VehicleDetectionTracker.infrastructure.license_plate.batch_accumulator import (
    BatchAccumulator,
)
from VehicleDetectionTracker.infrastructure.license_plate.inference_queue import (
    InferenceQueue,
)
from VehicleDetectionTracker.infrastructure.license_plate.ocr_service import (
    LicensePlateOCRService,
)
from VehicleDetectionTracker.infrastructure.license_plate.plate_image_processor import (
    PlateImageProcessor,
)
from VehicleDetectionTracker.infrastructure.license_plate.yolo_detector import (
    YOLOPlateDetector,
    load_plate_detector,
)

__all__ = [
    "BatchAccumulator",
    "InferenceQueue",
    "LicensePlateOCRService",
    "PlateImageProcessor",
    "YOLOPlateDetector",
    "load_plate_detector",
]