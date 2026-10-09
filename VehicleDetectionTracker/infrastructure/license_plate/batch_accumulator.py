"""Thread-safe accumulator for partial or size-triggered vehicle batches."""

import threading
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from VehicleDetectionTracker.logging_utils import log as shared_log


class BatchAccumulator:
    def __init__(
        self,
        batch_size: int = 8,
        time_threshold_ms: int = 500,
        max_batch_wait_ms: int = 2000,
        log_func: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.batch_size = batch_size
        self.time_threshold_ms = time_threshold_ms
        self.max_batch_wait_ms = max_batch_wait_ms
        self.log_func = log_func or shared_log
        # Keep one vehicle entry with every sampled crop collected for that ID.
        self.pending_batch: Dict[Any, dict] = {}
        self.batch_lock = threading.Lock()
        self.batch_event = threading.Event()
        self.batch_start_time = None
        self.last_flush_time = datetime.now()
        self.is_processing = False
        self.total_processed = 0
        self.total_batches = 0
        self.log_func(
            f"[BATCH_ACCUM] Initialized: batch_size={batch_size}, "
            f"time_threshold={time_threshold_ms}ms, max_wait={max_batch_wait_ms}ms"
        )

    def add_detection(
        self,
        track_id: int,
        vehicle_frame: Any,
        timestamp_str: str,
        vehicle_dir: str,
        direction: str,
        timestamp: Any,
    ) -> bool:
        with self.batch_lock:
            if not self.pending_batch and self.batch_start_time is None:
                self.batch_start_time = datetime.now()

            frame_item = {
                "frame": vehicle_frame.copy(),
                "timestamp_str": timestamp_str,
                "vehicle_dir": vehicle_dir,
                "direction": direction,
                "timestamp": timestamp,
            }
            vehicle_item = self.pending_batch.get(track_id)
            if vehicle_item is None:
                vehicle_item = {**frame_item, "frames": []}
                self.pending_batch[track_id] = vehicle_item
            vehicle_item["frames"].append(frame_item)
            # Preserve the legacy fields as the latest frame for older consumers.
            vehicle_item.update(frame_item)

            batch_size_now = sum(len(item["frames"]) for item in self.pending_batch.values())
            elapsed_ms = (datetime.now() - self.batch_start_time).total_seconds() * 1000
            self.log_func(
                f"[BATCH_ACCUM] Added track_id={track_id}, "
                f"batch_size={batch_size_now}/{self.batch_size}, "
                f"vehicles={len(self.pending_batch)}, "
                f"elapsed={elapsed_ms:.0f}ms/{self.time_threshold_ms}ms"
            )
            should_process = (
                batch_size_now >= self.batch_size
                or elapsed_ms >= self.time_threshold_ms
            )
            if should_process:
                self.batch_event.set()
                self.log_func(
                    f"[BATCH_ACCUM] ✓ Batch trigger: size={batch_size_now} "
                    f"or time={elapsed_ms:.0f}ms"
                )
                return True
            return False

    def _take_pending_batch(self) -> dict:
        result = {
            track_id: {**vehicle_item, "frames": list(vehicle_item["frames"])}
            for track_id, vehicle_item in self.pending_batch.items()
        }
        self.pending_batch.clear()
        self.batch_start_time = None
        self.batch_event.clear()
        if result:
            frame_count = sum(len(item["frames"]) for item in result.values())
            self.log_func(
                f"[BATCH_ACCUM] Batch retrieved: frames={frame_count}, "
                f"vehicles={len(result)}"
            )
        return result

    def get_batch(self, wait_timeout_ms: Optional[int] = None) -> dict:
        """Wait up to the timeout for a trigger, then return pending items."""
        with self.batch_lock:
            if self.pending_batch:
                return self._take_pending_batch()
            self.batch_event.clear()

        if wait_timeout_ms:
            self.batch_event.wait(wait_timeout_ms / 1000.0)

        with self.batch_lock:
            return self._take_pending_batch()

    def flush(self) -> dict:
        with self.batch_lock:
            result = {
                track_id: {**vehicle_item, "frames": list(vehicle_item["frames"])}
                for track_id, vehicle_item in self.pending_batch.items()
            }
            if result:
                elapsed = (
                    (datetime.now() - self.batch_start_time).total_seconds() * 1000
                    if self.batch_start_time
                    else 0
                )
                frame_count = sum(len(item["frames"]) for item in result.values())
                self.log_func(
                    f"[BATCH_ACCUM] ✓✓ Batch flushed: frames={frame_count}, "
                    f"vehicles={len(result)}, "
                    f"elapsed={elapsed:.0f}ms"
                )
            self.pending_batch.clear()
            self.batch_start_time = None
            self.batch_event.clear()
            return result

    def get_batch_stats(self) -> dict:
        with self.batch_lock:
            elapsed_ms = (
                (datetime.now() - self.batch_start_time).total_seconds() * 1000
                if self.batch_start_time
                else 0
            )
            pending_frames = sum(
                len(item["frames"]) for item in self.pending_batch.values()
            )
            return {
                "pending_items": pending_frames,
                "pending_vehicles": len(self.pending_batch),
                "pending_track_ids": set(self.pending_batch),
                "elapsed_ms": elapsed_ms,
                "total_processed": self.total_processed,
                "total_batches": self.total_batches,
                "is_processing": self.is_processing,
            }