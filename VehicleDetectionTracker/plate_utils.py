import asyncio
import cv2
import numpy as np
import torch

from VehicleDetectionTracker.function import utils_rotate, helper


def initialize_plate_detector(model_path="model/LP_detector.pt"):
    """Load and return the license plate detector (yolov5 custom) or None on error."""
    try:
        plate_model = torch.hub.load(
            "yolov5",
            "custom",
            path=model_path,
            force_reload=False,
            source="local",
        )
        return plate_model
    except Exception as e:
        print(f"Error loading license plate model: {e}")
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
        print(f"Error in plate image preprocessing: {e}")
        return plate_image


def _sync_plate_inference(plate_model, vehicle_frame, model_lock, size=640):
    """Run plate model synchronously in a thread-safe way and return results or None."""
    if plate_model is None:
        return None
    with model_lock:
        return plate_model(vehicle_frame, size=size)


def _async_plate_inference(plate_model, vehicle_frame, executor, model_lock, size=640):
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(executor, lambda: _sync_plate_inference(plate_model, vehicle_frame, model_lock, size))


def _run_ocr_attempt(ocr_reader, plate_image):
    # wrapper used by executor - synchronous
    # return helper.read_plate(ocr_reader, plate_image)
    return ocr_reader.read_license_plate(plate_image)


async def ocr_attempt_async(ocr_reader, plate_image, cc, ct, executor, model_lock):
    loop = asyncio.get_event_loop()
    rotated_image = utils_rotate.deskew(plate_image, cc, ct)
    # OCR reading uses helper.read_plate which is CPU-bound; run in executor
    with model_lock:
        lp = await loop.run_in_executor(executor, lambda: _run_ocr_attempt(ocr_reader, rotated_image))
    return lp, cc, ct


def detect_license_plate_sync(plate_model, vehicle_frame, ocr_reader, model_lock):
    try:
        if plate_model is None:
            return {"text": None, "bbox": None}

        results = _sync_plate_inference(plate_model, vehicle_frame, model_lock)

        if results is None or not results.pred[0].shape[0]:
            return {"text": None, "bbox": None}

        pred = results.pred[0]
        best_det_idx = pred[:, 4].argmax()
        best_det = pred[best_det_idx]
        x1, y1, x2, y2 = map(int, best_det[:4].tolist())

        length_plate = x2 - x1
        if length_plate < 40:
            return {"text": None, "bbox": None}

        plate_image = vehicle_frame[y1:y2, x1:x2]
        if plate_image.size == 0:
            return {"text": None, "bbox": None}

        if ocr_reader is None:
            # caller should initialize OCR; return unknown
            return {"text": None, "bbox": (x1, y1, x2, y2)}

        lp = "unknown"
        for cc in range(0, 2):
            for ct in range(0, 2):
                with model_lock:
                    # lp = helper.read_plate(ocr_reader, utils_rotate.deskew(plate_image, cc, ct))
                    lp = ocr_reader.read_license_plate(utils_rotate.deskew(plate_image, cc, ct))
                if lp != "unknown" and lp is not None:
                    return {"text": lp, "bbox": (x1, y1, x2, y2)}

        return {"text": lp, "bbox": (x1, y1, x2, y2)}
    except Exception as e:
        print(f"Error in license plate detection (sync): {e}")
        return {"text": None, "bbox": None}


async def detect_license_plate_async(plate_model, vehicle_frame, ocr_reader, executor, model_lock):
    try:
        if plate_model is None:
            return {"text": None, "bbox": None}

        # run detection in executor
        results = await _async_plate_inference(plate_model, vehicle_frame, executor, model_lock)

        if results is None or not results.pred[0].shape[0]:
            return {"text": None, "bbox": None}

        pred = results.pred[0]
        best_det_idx = pred[:, 4].argmax()
        best_det = pred[best_det_idx]
        x1, y1, x2, y2 = map(int, best_det[:4].tolist())

        length_plate = x2 - x1
        if length_plate < 40:
            return {"text": None, "bbox": None}

        plate_image = vehicle_frame[y1:y2, x1:x2]
        if plate_image.size == 0:
            return {"text": None, "bbox": None}

        if ocr_reader is None:
            return {"text": None, "bbox": (x1, y1, x2, y2)}

        # run OCR attempts in parallel
        tasks = []
        for cc in range(0, 2):
            for ct in range(0, 2):
                tasks.append(ocr_attempt_async(ocr_reader, plate_image, cc, ct, executor, model_lock))

        lp = "unknown"
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                continue
            result_lp, cc, ct = result
            if result_lp != "unknown" and result_lp is not None:
                lp = result_lp
                break

        return {"text": lp, "bbox": (x1, y1, x2, y2)}
    except Exception as e:
        print(f"Error in async license plate detection: {e}")
        return {"text": None, "bbox": None}
