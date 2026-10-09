"""OpenCV-backed license-plate image operations."""

from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import cv2
import numpy as np

from VehicleDetectionTracker.domain.license_plate.entities import PlateBoundingBox


class PlateImageProcessor:
    def __init__(self, log_func: Optional[Callable[[str], None]] = None) -> None:
        self._log = log_func or (lambda message: None)

    def crop_detection_region(self, vehicle_frame: Any) -> Any:
        """Return the centered square from the bottom half of a vehicle crop."""
        height, _ = vehicle_frame.shape[:2]
        bottom_half = vehicle_frame[height // 2 :, :, :]
        bottom_height, bottom_width = bottom_half.shape[:2]
        square_size = min(bottom_height, bottom_width)
        left = (bottom_width - square_size) // 2
        right = left + square_size
        return bottom_half[:, left:right, :]

    @staticmethod
    def crop_bounding_box(frame: Any, box: PlateBoundingBox) -> Tuple[Any, PlateBoundingBox]:
        """Clip a box to image bounds and return the cropped image and clipped box."""
        frame_height, frame_width = frame.shape[:2]
        x1 = max(0, min(box.x1, frame_width - 1))
        y1 = max(0, min(box.y1, frame_height - 1))
        x2 = max(x1 + 1, min(box.x2, frame_width))
        y2 = max(y1 + 1, min(box.y2, frame_height))
        clipped_box = PlateBoundingBox(x1, y1, x2, y2)
        return frame[y1:y2, x1:x2], clipped_box

    def save_plate_image(self, vehicle_dir: str, timestamp_str: str, image: Any) -> str:
        path = str(Path(vehicle_dir) / f"license_frame_{timestamp_str}.png")
        cv2.imwrite(path, image)
        return path

    def preprocess_plate_image(self, plate_image: Any) -> Any:
        try:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
            thresholded = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11,
                2,
            )
            denoised = cv2.fastNlMeansDenoising(thresholded)
            kernel = np.ones((1, 1), np.uint8)
            return cv2.dilate(denoised, kernel, iterations=1)
        except Exception as error:
            self._log(f"Error in plate image preprocessing: {error}")
            return plate_image