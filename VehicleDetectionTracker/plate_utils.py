"""Backward-compatible facade for license-plate services.

Existing callers should continue importing this module. New implementation
details live in the domain, application, and infrastructure packages.
"""

import threading
from pathlib import Path
from typing import Any, Callable, Optional

from VehicleDetectionTracker.application.license_plate.batch_detect_license_plates import (
    BatchDetectLicensePlatesUseCase,
)
from VehicleDetectionTracker.application.license_plate.detect_license_plate import (
    DetectLicensePlateUseCase,
)
from VehicleDetectionTracker.application.license_plate.dto import PlateDetectionConfig
from VehicleDetectionTracker.application.license_plate.submit_detection_async import (
    SubmitLicensePlateDetectionUseCase,
)
from VehicleDetectionTracker.config_loader import (
    get_batch_inference_config,
    get_plate_detection_config,
)
from VehicleDetectionTracker.domain.license_plate.entities import PlateDetectionResult
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
from VehicleDetectionTracker.logging_utils import log as shared_log


_inference_queue = None
_batch_accumulator = None
_inference_queue_lock = threading.Lock()
_batch_accumulator_lock = threading.Lock()


def _ensure_log_dir() -> Path:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    return log_dir


def _log(message: str) -> None:
    shared_log(message)


def initialize_inference_queue(num_workers: int = 6) -> InferenceQueue:
    global _inference_queue
    with _inference_queue_lock:
        if _inference_queue is None:
            _inference_queue = InferenceQueue(num_workers=num_workers, log_func=_log)
        return _inference_queue


def get_inference_queue() -> InferenceQueue:
    global _inference_queue
    with _inference_queue_lock:
        if _inference_queue is None:
            _inference_queue = InferenceQueue(num_workers=6, log_func=_log)
        return _inference_queue


def initialize_batch_accumulator(
    batch_size: int = 8,
    time_threshold_ms: int = 500,
) -> BatchAccumulator:
    global _batch_accumulator
    with _batch_accumulator_lock:
        if _batch_accumulator is None:
            config = PlateDetectionConfig.from_mapping(
                get_plate_detection_config(),
                get_batch_inference_config(),
            )
            _batch_accumulator = BatchAccumulator(
                batch_size=batch_size,
                time_threshold_ms=time_threshold_ms,
                max_batch_wait_ms=config.max_batch_wait_ms,
                log_func=_log,
            )
        return _batch_accumulator


def get_batch_accumulator() -> BatchAccumulator:
    global _batch_accumulator
    with _batch_accumulator_lock:
        if _batch_accumulator is None:
            config = PlateDetectionConfig.from_mapping(
                get_plate_detection_config(),
                get_batch_inference_config(),
            )
            _batch_accumulator = BatchAccumulator(
                max_batch_wait_ms=config.max_batch_wait_ms,
                log_func=_log,
            )
        return _batch_accumulator


def initialize_plate_detector(
    model_path: str = "model/LP_detector.pt",
    device: Optional[str] = None,
) -> Any:
    return load_plate_detector(model_path=model_path, device=device)


def preprocess_plate_image(plate_image: Any) -> Any:
    return PlateImageProcessor(log_func=_log).preprocess_plate_image(plate_image)


def _get_plate_config() -> PlateDetectionConfig:
    return PlateDetectionConfig.from_mapping(
        get_plate_detection_config(),
        get_batch_inference_config(),
    )


def _sync_plate_inference(
    plate_model: Any,
    square_frame: Any,
    model_lock: Any,
    size: Optional[int] = None,
) -> Any:
    """Compatibility wrapper returning the original Ultralytics result object."""
    config = _get_plate_config()
    detector = YOLOPlateDetector(
        plate_model,
        model_lock=model_lock,
        image_size=size or config.image_size,
        log_func=_log,
    )
    return detector.predict_raw(square_frame, image_size=size)


def _serialize_result(result: PlateDetectionResult) -> dict:
    return {"text": result.text, "count": result.count}


def detect_license_plate_sync(
    plate_model: Any,
    vehicle_frame: Any,
    ocr_reader: Any,
    model_lock: Any,
    timestamp_str: str,
    vehicle_dir: str = "screenshots",
    track_id: Any = None,
) -> dict:
    if plate_model is None:
        _log(f"[PLATE_DETECT] vehicle_id={track_id} ❌ plate_model is None, return None")
        return {"text": None, "count": None}

    config = _get_plate_config()
    detector = YOLOPlateDetector(
        plate_model,
        model_lock=model_lock,
        image_size=config.image_size,
        log_func=_log,
    )
    ocr_service = (
        LicensePlateOCRService(ocr_reader, model_lock=model_lock, log_func=_log)
        if ocr_reader is not None
        else None
    )
    use_case = DetectLicensePlateUseCase(
        detector,
        ocr_service,
        PlateImageProcessor(log_func=_log),
        config,
        _log,
    )
    return _serialize_result(
        use_case.execute(vehicle_frame, timestamp_str, vehicle_dir, track_id)
    )


def submit_plate_detection_async(
    plate_model: Any,
    vehicle_frame: Any,
    ocr_reader: Any,
    model_lock: Any,
    timestamp_str: str,
    callback: Optional[Callable[[Any], None]],
    vehicle_dir: str = "screenshots",
    track_id: Any = None,
) -> None:
    try:
        def detection_task() -> dict:
            return detect_license_plate_sync(
                plate_model,
                vehicle_frame,
                ocr_reader,
                model_lock,
                timestamp_str,
                vehicle_dir=vehicle_dir,
                track_id=track_id,
            )

        queue_instance = get_inference_queue()
        SubmitLicensePlateDetectionUseCase().execute(
            queue_instance,
            detection_task,
            (),
            callback,
        )
        _log(
            f"[PLATE_DETECT] vehicle_id={track_id} ✓ Submitted to inference queue "
            f"(queue_size={queue_instance.task_queue.qsize()})"
        )
    except Exception as error:
        _log(f"[PLATE_DETECT] vehicle_id={track_id} ❌ Error submitting to inference queue: {error}")
        if callback:
            callback({"text": None, "count": None})


def batch_detect_license_plates(
    plate_model: Any,
    vehicle_frames_dict: dict,
    ocr_reader: Any,
    model_lock: Any,
    detection_config: dict,
) -> dict:
    """Run plate detection and OCR per vehicle while preserving the legacy result shape."""
    config = _get_plate_config()
    detector = YOLOPlateDetector(
        plate_model,
        model_lock=model_lock,
        image_size=config.image_size,
        log_func=_log,
    )
    ocr_service = (
        LicensePlateOCRService(ocr_reader, model_lock=model_lock, log_func=_log)
        if ocr_reader is not None
        else None
    )
    use_case = BatchDetectLicensePlatesUseCase(
        detector,
        ocr_service,
        PlateImageProcessor(log_func=_log),
        config,
        _log,
    )
    return use_case.execute(vehicle_frames_dict)


def _batch_ocr_plates(
    ocr_reader: Any,
    plate_crops_dict: dict,
    model_lock: Any,
) -> dict:
    if not ocr_reader:
        _log("[BATCH_OCR] ⚠ OCR reader is None, skipping batch OCR")
        return {track_id: "unknown" for track_id in plate_crops_dict}

    try:
        track_ids = list(plate_crops_dict)
        service = LicensePlateOCRService(
            ocr_reader,
            model_lock=model_lock,
            log_func=_log,
        )
        if service.supports_batch:
            _log("[BATCH_OCR] Using native batch OCR processing...")
        else:
            _log(
                "[BATCH_OCR] OCR reader doesn't support batch, running sequential OCR "
                "under single lock..."
            )
        values = service.read_batch([plate_crops_dict[track_id] for track_id in track_ids])
        results = {}
        for track_id, value in zip(track_ids, values):
            results[track_id] = value
            _log(f"[BATCH_OCR]   track_id={track_id} Result: '{value}'")
        _log(f"[BATCH_OCR] ✓ Batch OCR complete: {len(results)} results")
        return results
    except Exception as error:
        import traceback

        _log(f"[BATCH_OCR] ❌ Error in batch OCR: {error}")
        _log(f"[BATCH_OCR] Traceback: {traceback.format_exc()}")
        return {track_id: "unknown" for track_id in plate_crops_dict}


__all__ = [
    "BatchAccumulator",
    "InferenceQueue",
    "_batch_ocr_plates",
    "_ensure_log_dir",
    "_log",
    "_sync_plate_inference",
    "batch_detect_license_plates",
    "detect_license_plate_sync",
    "get_batch_accumulator",
    "get_inference_queue",
    "initialize_batch_accumulator",
    "initialize_inference_queue",
    "initialize_plate_detector",
    "preprocess_plate_image",
    "submit_plate_detection_async",
]