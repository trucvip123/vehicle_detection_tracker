"""Batch orchestration for per-vehicle plate detection and OCR."""

from datetime import datetime
from typing import Any, Callable, Dict, Mapping, Optional

from VehicleDetectionTracker.application.license_plate.dto import PlateDetectionConfig
from VehicleDetectionTracker.domain.license_plate.interfaces import (
    LicensePlateOCR,
    PlateDetector,
)


class BatchDetectLicensePlatesUseCase:
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

    def execute(self, vehicle_frames: Mapping[Any, Any]) -> Dict[Any, dict]:
        start_time = datetime.now()
        batch_results: Dict[Any, dict] = {}
        self.log(f"[BATCH_DETECT] Starting batch processing: {len(vehicle_frames)} vehicles")
        self.log(
            f"[BATCH_DETECT] Step 1: Running plate detection on "
            f"{len(vehicle_frames)} vehicles..."
        )

        detection_start = datetime.now()
        detections_by_vehicle = {}
        vehicle_items = list(vehicle_frames.items())
        batch_detector = getattr(self.plate_detector, "detect_batch", None)
        use_true_batch = (
            self.config.use_true_batch
            and len(vehicle_items) > 1
            and callable(batch_detector)
        )

        if use_true_batch:
            self.log(
                f"[BATCH_DETECT] Using true-batch YOLO inference for "
                f"{len(vehicle_items)} vehicles..."
            )
            for track_id, _frame in vehicle_items:
                self.log(f"[BATCH_DETECT]   track_id={track_id} Running detection...")
            try:
                batch_detections = batch_detector(
                    [frame for _track_id, frame in vehicle_items],
                    image_size=self.config.image_size,
                )
                if len(batch_detections) != len(vehicle_items):
                    raise RuntimeError(
                        f"Detector returned {len(batch_detections)} results "
                        f"for {len(vehicle_items)} vehicles"
                    )
                for (track_id, _frame), detections in zip(vehicle_items, batch_detections):
                    detections_by_vehicle[track_id] = detections
                    self.log(
                        f"[BATCH_DETECT]   track_id={track_id} ✓ "
                        f"Detected {len(detections)} plates"
                    )
            except Exception as error:
                self.log(
                    f"[BATCH_DETECT] True-batch failed, falling back to per-vehicle "
                    f"inference: {error}"
                )
                self._detect_individually(vehicle_items, detections_by_vehicle)
        else:
            self._detect_individually(vehicle_items, detections_by_vehicle)

        detection_ms = (datetime.now() - detection_start).total_seconds() * 1000
        self.log(f"[BATCH_DETECT] Step 1 complete: Detection took {detection_ms:.1f}ms")
        self.log("[BATCH_DETECT] Step 2: Extracting plate crops...")

        plate_crops = {}
        metadata = {}
        for track_id, detections in detections_by_vehicle.items():
            num_detections = len(detections)
            frame = vehicle_frames[track_id]
            if num_detections:
                best_detection = detections[0]
                box = best_detection.bounding_box
                confidence = best_detection.confidence
                try:
                    plate_image, clipped_box = self.image_processor.crop_bounding_box(frame, box)
                    if plate_image.size > 0:
                        plate_crops[track_id] = plate_image
                        metadata[track_id] = {
                            "num_detections": num_detections,
                            "confidence": confidence,
                            "bbox": (
                                clipped_box.x1,
                                clipped_box.y1,
                                clipped_box.x2,
                                clipped_box.y2,
                            ),
                        }
                        self.log(
                            f"[BATCH_DETECT]   track_id={track_id} ✓ Extracted plate crop "
                            f"(conf={confidence:.3f})"
                        )
                    else:
                        metadata[track_id] = {
                            "num_detections": num_detections,
                            "confidence": confidence,
                            "bbox": None,
                        }
                        self.log(f"[BATCH_DETECT]   track_id={track_id} ⚠ Plate crop empty")
                except Exception as error:
                    self.log(f"[BATCH_DETECT]   track_id={track_id} ❌ Crop extraction error: {error}")
                    metadata[track_id] = {
                        "num_detections": num_detections,
                        "confidence": 0,
                        "bbox": None,
                    }
            else:
                metadata[track_id] = {
                    "num_detections": 0,
                    "confidence": 0,
                    "bbox": None,
                }

        self.log(f"[BATCH_DETECT] Step 2 complete: Extracted {len(plate_crops)} plate crops")
        self.log(f"[BATCH_DETECT] Step 3: Running batch OCR on {len(plate_crops)} plates...")
        ocr_start = datetime.now()
        ocr_results = self._run_batch_ocr(plate_crops)
        ocr_ms = (datetime.now() - ocr_start).total_seconds() * 1000
        self.log(f"[BATCH_DETECT] Step 3 complete: OCR took {ocr_ms:.1f}ms")

        self.log("[BATCH_DETECT] Step 4: Combining results...")
        for track_id in vehicle_frames:
            if track_id in plate_crops:
                batch_results[track_id] = {
                    "text": ocr_results.get(track_id, "unknown"),
                    "count": metadata[track_id]["num_detections"],
                    "confidence": metadata[track_id]["confidence"],
                }
                self.log(
                    f"[BATCH_DETECT]   track_id={track_id} Result: "
                    f"text='{batch_results[track_id]['text']}', "
                    f"count={batch_results[track_id]['count']}"
                )
            else:
                batch_results[track_id] = {
                    "text": None,
                    "count": metadata[track_id]["num_detections"],
                    "confidence": 0,
                }

        total_ms = (datetime.now() - start_time).total_seconds() * 1000
        self.log(
            f"[BATCH_DETECT] ✓ Batch processing complete: {len(batch_results)} vehicles, "
            f"detection={detection_ms:.1f}ms, ocr={ocr_ms:.1f}ms, total={total_ms:.1f}ms"
        )
        return batch_results

    def _detect_individually(self, vehicle_items, detections_by_vehicle) -> None:
        for track_id, frame in vehicle_items:
            try:
                self.log(f"[BATCH_DETECT]   track_id={track_id} Running detection...")
                detections = self.plate_detector.detect(
                    frame,
                    image_size=self.config.image_size,
                )
                detections_by_vehicle[track_id] = detections
                self.log(
                    f"[BATCH_DETECT]   track_id={track_id} ✓ Detected {len(detections)} plates"
                )
            except Exception as error:
                self.log(f"[BATCH_DETECT]   track_id={track_id} ❌ Detection error: {error}")
                detections_by_vehicle[track_id] = []

    def _run_batch_ocr(self, plate_crops: Mapping[Any, Any]) -> Dict[Any, str]:
        if self.ocr_service is None:
            self.log("[BATCH_OCR] ⚠ OCR reader is None, skipping batch OCR")
            return {track_id: "unknown" for track_id in plate_crops}

        try:
            track_ids = list(plate_crops)
            if getattr(self.ocr_service, "supports_batch", False):
                self.log("[BATCH_OCR] Using native batch OCR processing...")
            else:
                self.log(
                    "[BATCH_OCR] OCR reader doesn't support batch, running sequential OCR "
                    "under single lock..."
                )
            values = self.ocr_service.read_batch([plate_crops[track_id] for track_id in track_ids])
            results = {
                track_id: value
                for track_id, value in zip(track_ids, values)
            }
            for track_id, value in results.items():
                self.log(f"[BATCH_OCR]   track_id={track_id} Result: '{value}'")
            self.log(f"[BATCH_OCR] ✓ Batch OCR complete: {len(results)} results")
            return results
        except Exception as error:
            import traceback

            self.log(f"[BATCH_OCR] ❌ Error in batch OCR: {error}")
            self.log(f"[BATCH_OCR] Traceback: {traceback.format_exc()}")
            return {track_id: "unknown" for track_id in plate_crops}