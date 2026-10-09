"""Configuration and transport values for license-plate use cases."""

from dataclasses import dataclass
from typing import Mapping, Optional


@dataclass(frozen=True)
class PlateDetectionConfig:
    image_size: int = 640
    min_width: int = 40
    min_height: int = 20
    min_confidence: float = 0.25
    batch_size: int = 8
    time_threshold_ms: int = 500
    max_batch_wait_ms: int = 2000
    use_true_batch: bool = False

    @classmethod
    def from_mapping(
        cls,
        values: Optional[Mapping[str, object]] = None,
        batch_values: Optional[Mapping[str, object]] = None,
    ) -> "PlateDetectionConfig":
        values = values or {}
        batch_values = batch_values or {}
        return cls(
            image_size=int(values.get("image_size", cls.image_size)),
            min_width=int(values.get("min_width", cls.min_width)),
            min_height=int(values.get("min_height", cls.min_height)),
            min_confidence=float(values.get("min_confidence", cls.min_confidence)),
            batch_size=int(batch_values.get("batch_size", cls.batch_size)),
            time_threshold_ms=int(
                batch_values.get("time_threshold_ms", cls.time_threshold_ms)
            ),
            max_batch_wait_ms=int(
                batch_values.get("max_wait_time_ms", cls.max_batch_wait_ms)
            ),
            use_true_batch=bool(batch_values.get("use_true_batch", False)),
        )