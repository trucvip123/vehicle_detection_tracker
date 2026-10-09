"""Business values produced while detecting a license plate."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PlateBoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True)
class PlateDetection:
    bounding_box: PlateBoundingBox
    confidence: float


@dataclass(frozen=True)
class PlateDetectionResult:
    text: Optional[str]
    count: Optional[int]
    confidence: Optional[float] = None