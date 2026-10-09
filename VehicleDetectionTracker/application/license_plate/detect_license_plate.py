"""Synchronous license-plate detection use case."""

from typing import Any, Callable, Optional

from VehicleDetectionTracker.application.license_plate.dto import PlateDetectionConfig
from VehicleDetectionTracker.domain.license_plate.entities import PlateDetectionResult
from VehicleDetectionTracker.domain.license_plate.interfaces import (
    LicensePlateOCR,
    PlateDetector,
)


class DetectLicensePlateUseCase:
    def __init__(
        self,
        plate_detector: PlateDetector,
        ocr_service: Optional[LicensePlateOCR],
        image_processor: Any,
        config: PlateDetectionConfig,
        log_func: Callable[[str], None],
    ) -> None:
        self.plate_detector = plate_detector
        self.ocr_service = ocr_service
        self.image_processor = image_processor
        self.config = config
        self.log = log_func

    def execute(
        self,
        vehicle_frame: Any,
        timestamp_str: str,
        vehicle_dir: str,
        track_id: Any = None,
    ) -> PlateDetectionResult:
        try:
            self.log(f"[PLATE_DETECT] vehicle_id={track_id} Bắt đầu detect license plate")
            self.log(
                f"[PLATE_DETECT] vehicle_id={track_id} Vehicle frame shape: "
                f"{vehicle_frame.shape if vehicle_frame is not None else 'None'}"
            )
            self.log(f"[PLATE_DETECT] vehicle_id={track_id} Đang chạy plate model inference...")

            detection_region = self.image_processor.crop_detection_region(vehicle_frame)
            if detection_region is None or detection_region.size == 0:
                self.log(f"[PLATE_DETECT] vehicle_id={track_id} ❌ Inference results is None")
                return PlateDetectionResult(None, None)

            detections = self.plate_detector.detect(
                detection_region,
                image_size=self.config.image_size,
            )
            detection_count = len(detections)
            if not detection_count:
                self.log(f"[PLATE_DETECT] vehicle_id={track_id} ❌ Không có detection nào")
                return PlateDetectionResult(text=None, count=0)

            for index, detection in enumerate(detections):
                box = detection.bounding_box
                self.log(
                    f"[PLATE_DETECT] vehicle_id={track_id} Detection {index}: "
                    f"bbox={[box.x1, box.y1, box.x2, box.y2]}, "
                    f"confidence={detection.confidence:.3f}"
                )

            best_detection = detections[0]
            box = best_detection.bounding_box
            confidence = best_detection.confidence
            width = box.x2 - box.x1
            height = box.y2 - box.y1
            self.log(
                f"[PLATE_DETECT] vehicle_id={track_id} Best detection: "
                f"bbox=({box.x1},{box.y1},{box.x2},{box.y2}), confidence={confidence:.3f}"
            )
            self.log(
                f"[PLATE_DETECT] vehicle_id={track_id} Plate dimensions: "
                f"width={width}, height={height}"
            )

            if confidence < self.config.min_confidence:
                self.log(
                    f"[PLATE_DETECT] vehicle_id={track_id} ❌ Confidence quá thấp "
                    f"({confidence:.3f} < {self.config.min_confidence}), return None"
                )
                return PlateDetectionResult(None, detection_count, confidence)
            if width < self.config.min_width:
                self.log(
                    f"[PLATE_DETECT] vehicle_id={track_id} ❌ Plate quá nhỏ "
                    f"(width={width} < {self.config.min_width}), return None"
                )
                return PlateDetectionResult(None, detection_count, confidence)
            if height < self.config.min_height:
                self.log(
                    f"[PLATE_DETECT] vehicle_id={track_id} ❌ Plate quá nhỏ "
                    f"(height={height} < {self.config.min_height}), return None"
                )
                return PlateDetectionResult(None, detection_count, confidence)

            plate_image, _ = self.image_processor.crop_bounding_box(
                detection_region,
                box,
            )
            if plate_image.size == 0:
                self.log(f"[PLATE_DETECT] vehicle_id={track_id} ❌ Plate image size = 0, return None")
                return PlateDetectionResult(None, detection_count, confidence)

            self.log(
                f"[PLATE_DETECT] vehicle_id={track_id} ✓ Extracted plate image shape: "
                f"{plate_image.shape}"
            )
            self.image_processor.save_plate_image(vehicle_dir, timestamp_str, plate_image)

            if self.ocr_service is None:
                self.log(f"[PLATE_DETECT] vehicle_id={track_id} ⚠ OCR reader is None, return bbox only")
                return PlateDetectionResult(None, detection_count, confidence)

            text = self.ocr_service.read(plate_image)
            self.log(f"[PLATE_DETECT] vehicle_id={track_id} ✓ Tìm thấy biển số: '{text}'")
            return PlateDetectionResult(text, detection_count, confidence)
        except Exception as error:
            import traceback

            self.log(f"[PLATE_DETECT] vehicle_id={track_id} ❌ ERROR in license plate detection: {error}")
            self.log(f"[PLATE_DETECT] vehicle_id={track_id} Traceback: {traceback.format_exc()}")
            return PlateDetectionResult(None, None)