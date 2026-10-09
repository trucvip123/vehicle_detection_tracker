"""License-plate domain model and ports."""

from VehicleDetectionTracker.domain.license_plate.entities import (
    PlateBoundingBox,
    PlateDetection,
    PlateDetectionResult,
)
from VehicleDetectionTracker.domain.license_plate.interfaces import (
    LicensePlateOCR,
    PlateDetector,
)

__all__ = [
    "LicensePlateOCR",
    "PlateBoundingBox",
    "PlateDetection",
    "PlateDetectionResult",
    "PlateDetector",
]