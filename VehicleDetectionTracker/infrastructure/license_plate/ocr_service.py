"""Adapter around the project's existing OCR reader."""

from typing import Any, Callable, List, Optional, Sequence

from VehicleDetectionTracker.logging_utils import log as shared_log


class LicensePlateOCRService:
    def __init__(
        self,
        ocr_reader: Any,
        model_lock: Any = None,
        log_func: Optional[Callable[[str], None]] = None,
    ) -> None:
        import threading

        self.ocr_reader = ocr_reader
        self.model_lock = model_lock or threading.Lock()
        self._log = log_func or shared_log

    @property
    def supports_batch(self) -> bool:
        return hasattr(self.ocr_reader, "read_license_plate_batch")

    def read(self, image: Any) -> str:
        with self.model_lock:
            return self.ocr_reader.read_license_plate(image)

    def read_batch(self, images: Sequence[Any]) -> List[str]:
        if not self.ocr_reader:
            return ["unknown" for _ in images]

        if self.supports_batch:
            with self.model_lock:
                values = self.ocr_reader.read_license_plate_batch(list(images))
            return [value if value else "unknown" for value in values]

        results = []
        with self.model_lock:
            for image in images:
                try:
                    value = self.ocr_reader.read_license_plate(image)
                    results.append(value if value else "unknown")
                except Exception as error:
                    self._log(f"[BATCH_OCR] ❌ OCR error: {error}")
                    results.append("unknown")
        return results