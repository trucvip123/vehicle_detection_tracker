"""Ports used by license-plate application use cases."""

from typing import Any, List, Optional, Protocol, Sequence

from VehicleDetectionTracker.domain.license_plate.entities import PlateDetection


class PlateDetector(Protocol):
    def detect(
        self,
        image: Any,
        image_size: Optional[int] = None,
    ) -> List[PlateDetection]:
        """Return plate detections in detector confidence order."""

    def detect_batch(
        self,
        images: Sequence[Any],
        image_size: Optional[int] = None,
    ) -> List[List[PlateDetection]]:
        """Return per-image detections in input order."""


class LicensePlateOCR(Protocol):
    def read(self, image: Any) -> str:
        """Read one cropped plate image."""

    def read_batch(self, images: Sequence[Any]) -> List[str]:
        """Read multiple cropped plate images, sequentially if needed."""