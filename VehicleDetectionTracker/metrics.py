"""
Comprehensive metrics and monitoring module for vehicle detection tracker.

Tracks:
- Frame processing statistics (frames processed, rejected, FPS)
- Detection performance (vehicles detected, confidence scores, processing time)
- Tracking metrics (active vehicles, total tracked, tracking continuity)
- Plate processing (detections, OCR success rate, confidence)
- Notification metrics (sent, success/failure rate, API calls)
- System health (processing time per frame, memory usage, thread pool utilization)
"""

import time
import threading
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, List, Optional, Tuple, Any
import psutil
import os


class MetricsCollector:
    """Thread-safe metrics collection and reporting system."""

    def __init__(self, history_size: int = 1000):
        """
        Initialize metrics collector.

        Args:
            history_size: Maximum number of historical data points to keep per metric
        """
        self._lock = threading.RLock()
        self.history_size = history_size
        self.start_time = datetime.now()
        self.process = psutil.Process(os.getpid())

        # Frame processing metrics
        self._frames_processed = 0
        self._frames_rejected = 0
        self._frame_processing_times = deque(maxlen=history_size)
        self._frame_quality_issues = {}  # Count of each issue type

        # Detection metrics
        self._detections_total = 0
        self._detection_confidence_scores = deque(maxlen=history_size)
        self._vehicles_detected_per_frame = deque(maxlen=history_size)
        self._detection_processing_times = deque(maxlen=history_size)

        # Tracking metrics
        self._tracking_total_vehicles = 0
        self._tracking_active_vehicles = 0
        self._tracking_id_losses = 0
        self._tracking_reassignments = 0
        self._tracking_continuity_history = deque(maxlen=history_size)

        # Plate processing metrics
        self._plate_detections = 0
        self._plate_ocr_attempts = 0
        self._plate_ocr_successes = 0
        self._plate_confidence_scores = deque(maxlen=history_size)
        self._plate_processing_times = deque(maxlen=history_size)

        # Notification metrics
        self._notifications_sent = 0
        self._notifications_successful = 0
        self._notifications_failed = 0
        self._notification_api_calls = 0  # Telegram API calls

        # Queue and threading metrics
        self._thread_pool_queue_size = 0
        self._thread_pool_max_workers = 0
        self._memory_usage_history = deque(maxlen=history_size)

    # ===== Frame Processing Metrics =====

    def record_frame_processed(
        self, processing_time: float, is_quality_rejected: bool = False, issues: Optional[List[str]] = None
    ) -> None:
        """
        Record a frame processing event.

        Args:
            processing_time: Time taken to process frame (seconds)
            is_quality_rejected: Whether frame was rejected due to quality
            issues: List of quality issues found (if rejected)
        """
        with self._lock:
            if is_quality_rejected:
                self._frames_rejected += 1
                if issues:
                    for issue in issues:
                        self._frame_quality_issues[issue] = self._frame_quality_issues.get(issue, 0) + 1
            else:
                self._frames_processed += 1
            
            self._frame_processing_times.append(processing_time)

    def record_frame_quality_check(self, brightness: float, contrast: float, blur_var: float, entropy: float) -> None:
        """Record detailed frame quality metrics for trending."""
        with self._lock:
            # Store as tuple for later analysis
            self._frame_quality_issues[f"brightness_{int(brightness)}"] = (
                self._frame_quality_issues.get(f"brightness_{int(brightness)}", 0) + 1
            )

    # ===== Detection Metrics =====

    def record_detection(
        self,
        confidence: float,
        vehicles_detected_count: int,
        processing_time: float,
    ) -> None:
        """
        Record detection results from a frame.

        Args:
            confidence: Average detection confidence score (0-1)
            vehicles_detected_count: Number of vehicles detected in frame
            processing_time: Time taken to perform detection (seconds)
        """
        with self._lock:
            self._detections_total += vehicles_detected_count
            self._detection_confidence_scores.append(confidence)
            self._vehicles_detected_per_frame.append(vehicles_detected_count)
            self._detection_processing_times.append(processing_time)

    def record_tracking_update(
        self,
        active_vehicles: int,
        new_vehicle: bool = False,
        tracking_lost: bool = False,
        reassigned: bool = False,
    ) -> None:
        """
        Record tracking state update.

        Args:
            active_vehicles: Current number of actively tracked vehicles
            new_vehicle: Whether a new vehicle was detected
            tracking_lost: Whether tracking was lost for any vehicle
            reassigned: Whether tracking ID was reassigned
        """
        with self._lock:
            self._tracking_active_vehicles = active_vehicles
            if new_vehicle:
                self._tracking_total_vehicles += 1
            if tracking_lost:
                self._tracking_id_losses += 1
            if reassigned:
                self._tracking_reassignments += 1

            # Record continuity ratio
            if self._tracking_total_vehicles > 0:
                loss_ratio = self._tracking_id_losses / max(1, self._tracking_total_vehicles)
                continuity = 1.0 - loss_ratio
                self._tracking_continuity_history.append(continuity)

    # ===== Plate Processing Metrics =====

    def record_plate_detection(
        self,
        confidence: float,
        processing_time: float,
        ocr_attempted: bool = False,
        ocr_success: bool = False,
    ) -> None:
        """
        Record plate detection and OCR processing.

        Args:
            confidence: Plate detection confidence (0-1)
            processing_time: Time to detect plate (seconds)
            ocr_attempted: Whether OCR was attempted on this detection
            ocr_success: Whether OCR successfully extracted text
        """
        with self._lock:
            self._plate_detections += 1
            self._plate_confidence_scores.append(confidence)
            self._plate_processing_times.append(processing_time)

            if ocr_attempted:
                self._plate_ocr_attempts += 1
                if ocr_success:
                    self._plate_ocr_successes += 1

    # ===== Notification Metrics =====

    def record_notification_sent(self, success: bool, api_call: bool = False) -> None:
        """
        Record notification sending event.

        Args:
            success: Whether notification was sent successfully
            api_call: Whether this involved an API call (e.g., Telegram)
        """
        with self._lock:
            self._notifications_sent += 1
            if success:
                self._notifications_successful += 1
            else:
                self._notifications_failed += 1
            if api_call:
                self._notification_api_calls += 1

    # ===== System Health Metrics =====

    def record_system_metrics(self, queue_size: int, max_workers: int) -> None:
        """
        Record system health metrics.

        Args:
            queue_size: Current thread pool queue size
            max_workers: Maximum number of worker threads
        """
        with self._lock:
            self._thread_pool_queue_size = queue_size
            self._thread_pool_max_workers = max_workers

            # Record memory usage
            try:
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                self._memory_usage_history.append(memory_mb)
            except Exception:
                pass

    # ===== Snapshot Methods =====

    def get_frame_metrics(self) -> Dict[str, Any]:
        """Get current frame processing metrics."""
        with self._lock:
            processed = self._frames_processed
            rejected = self._frames_rejected
            total = processed + rejected
            total_time = sum(self._frame_processing_times)
            avg_time = total_time / total if total > 0 else 0.0
            fps = 1.0 / avg_time if avg_time > 0 else 0.0

            return {
                "frames_processed": processed,
                "frames_rejected": rejected,
                "rejection_rate": (rejected / total * 100) if total > 0 else 0.0,
                "avg_processing_time_ms": avg_time * 1000,
                "estimated_fps": fps,
                "quality_issues": dict(self._frame_quality_issues),
            }

    def get_detection_metrics(self) -> Dict[str, Any]:
        """Get current detection performance metrics."""
        with self._lock:
            detections = self._detections_total
            confidence_scores = list(self._detection_confidence_scores)
            vehicles_per_frame = list(self._vehicles_detected_per_frame)
            processing_times = list(self._detection_processing_times)

            return {
                "total_detections": detections,
                "avg_detection_confidence": (
                    sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
                ),
                "avg_detections_per_frame": (
                    sum(vehicles_per_frame) / len(vehicles_per_frame) if vehicles_per_frame else 0.0
                ),
                "avg_detection_time_ms": (
                    (sum(processing_times) / len(processing_times) * 1000) if processing_times else 0.0
                ),
                "max_detection_time_ms": (
                    max(processing_times) * 1000 if processing_times else 0.0
                ),
            }

    def get_tracking_metrics(self) -> Dict[str, Any]:
        """Get current tracking metrics."""
        with self._lock:
            total = self._tracking_total_vehicles
            active = self._tracking_active_vehicles
            losses = self._tracking_id_losses
            reassignments = self._tracking_reassignments
            continuity_scores = list(self._tracking_continuity_history)

            return {
                "total_vehicles_tracked": total,
                "active_vehicles": active,
                "tracking_id_losses": losses,
                "tracking_reassignments": reassignments,
                "loss_rate": (losses / total * 100) if total > 0 else 0.0,
                "avg_tracking_continuity": (
                    sum(continuity_scores) / len(continuity_scores) if continuity_scores else 0.0
                ),
            }

    def get_plate_metrics(self) -> Dict[str, Any]:
        """Get current plate processing metrics."""
        with self._lock:
            detections = self._plate_detections
            ocr_attempts = self._plate_ocr_attempts
            ocr_successes = self._plate_ocr_successes
            confidence_scores = list(self._plate_confidence_scores)
            processing_times = list(self._plate_processing_times)

            return {
                "total_plate_detections": detections,
                "plate_ocr_attempts": ocr_attempts,
                "plate_ocr_successes": ocr_successes,
                "ocr_success_rate": (
                    (ocr_successes / ocr_attempts * 100) if ocr_attempts > 0 else 0.0
                ),
                "avg_plate_confidence": (
                    sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
                ),
                "avg_plate_processing_time_ms": (
                    (sum(processing_times) / len(processing_times) * 1000)
                    if processing_times
                    else 0.0
                ),
            }

    def get_notification_metrics(self) -> Dict[str, Any]:
        """Get current notification metrics."""
        with self._lock:
            sent = self._notifications_sent
            successful = self._notifications_successful
            failed = self._notifications_failed

            return {
                "notifications_sent": sent,
                "notifications_successful": successful,
                "notifications_failed": failed,
                "success_rate": (successful / sent * 100) if sent > 0 else 0.0,
                "api_calls_made": self._notification_api_calls,
            }

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system health metrics."""
        with self._lock:
            memory_data = list(self._memory_usage_history)
            uptime_seconds = (datetime.now() - self.start_time).total_seconds()

            return {
                "uptime_seconds": uptime_seconds,
                "thread_pool_queue_size": self._thread_pool_queue_size,
                "thread_pool_max_workers": self._thread_pool_max_workers,
                "current_memory_mb": memory_data[-1] if memory_data else 0.0,
                "avg_memory_mb": sum(memory_data) / len(memory_data) if memory_data else 0.0,
                "peak_memory_mb": max(memory_data) if memory_data else 0.0,
            }

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get all metrics at once."""
        return {
            "frame": self.get_frame_metrics(),
            "detection": self.get_detection_metrics(),
            "tracking": self.get_tracking_metrics(),
            "plate": self.get_plate_metrics(),
            "notification": self.get_notification_metrics(),
            "system": self.get_system_metrics(),
        }

    # ===== Reporting Methods =====

    def generate_summary_report(self) -> str:
        """Generate human-readable summary report of all metrics."""
        all_metrics = self.get_all_metrics()
        report = []
        report.append("\n" + "=" * 80)
        report.append("VEHICLE DETECTION TRACKER - METRICS SUMMARY")
        report.append("=" * 80)
        report.append(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Frame Metrics
        fm = all_metrics["frame"]
        report.append("[FRAME PROCESSING]")
        report.append(f"  Frames processed: {fm['frames_processed']:,}")
        report.append(f"  Frames rejected: {fm['frames_rejected']:,} ({fm['rejection_rate']:.1f}%)")
        report.append(f"  Avg processing time: {fm['avg_processing_time_ms']:.1f} ms")
        report.append(f"  Estimated FPS: {fm['estimated_fps']:.1f}")
        if fm["quality_issues"]:
            report.append(f"  Quality issues: {fm['quality_issues']}")
        report.append("")

        # Detection Metrics
        dm = all_metrics["detection"]
        report.append("[DETECTION PERFORMANCE]")
        report.append(f"  Total detections: {dm['total_detections']:,}")
        report.append(f"  Avg confidence: {dm['avg_detection_confidence']:.3f}")
        report.append(f"  Avg detections/frame: {dm['avg_detections_per_frame']:.2f}")
        report.append(f"  Avg detection time: {dm['avg_detection_time_ms']:.1f} ms")
        report.append(f"  Max detection time: {dm['max_detection_time_ms']:.1f} ms")
        report.append("")

        # Tracking Metrics
        tm = all_metrics["tracking"]
        report.append("[TRACKING METRICS]")
        report.append(f"  Total vehicles tracked: {tm['total_vehicles_tracked']:,}")
        report.append(f"  Currently active: {tm['active_vehicles']:,}")
        report.append(f"  Tracking ID losses: {tm['tracking_id_losses']:,}")
        report.append(f"  Tracking reassignments: {tm['tracking_reassignments']:,}")
        report.append(f"  Loss rate: {tm['loss_rate']:.2f}%")
        report.append(f"  Avg tracking continuity: {tm['avg_tracking_continuity']:.3f}")
        report.append("")

        # Plate Metrics
        pm = all_metrics["plate"]
        report.append("[PLATE PROCESSING]")
        report.append(f"  Total detections: {pm['total_plate_detections']:,}")
        report.append(f"  OCR attempts: {pm['plate_ocr_attempts']:,}")
        report.append(f"  OCR successes: {pm['plate_ocr_successes']:,}")
        report.append(f"  OCR success rate: {pm['ocr_success_rate']:.1f}%")
        report.append(f"  Avg confidence: {pm['avg_plate_confidence']:.3f}")
        report.append(f"  Avg processing time: {pm['avg_plate_processing_time_ms']:.1f} ms")
        report.append("")

        # Notification Metrics
        nm = all_metrics["notification"]
        report.append("[NOTIFICATIONS]")
        report.append(f"  Sent: {nm['notifications_sent']:,}")
        report.append(f"  Successful: {nm['notifications_successful']:,}")
        report.append(f"  Failed: {nm['notifications_failed']:,}")
        report.append(f"  Success rate: {nm['success_rate']:.1f}%")
        report.append(f"  API calls: {nm['api_calls_made']:,}")
        report.append("")

        # System Metrics
        sm = all_metrics["system"]
        uptime_hours = sm["uptime_seconds"] / 3600
        report.append("[SYSTEM HEALTH]")
        report.append(f"  Uptime: {uptime_hours:.1f} hours")
        report.append(f"  Current memory: {sm['current_memory_mb']:.1f} MB")
        report.append(f"  Avg memory: {sm['avg_memory_mb']:.1f} MB")
        report.append(f"  Peak memory: {sm['peak_memory_mb']:.1f} MB")
        report.append(f"  Thread pool queue: {sm['thread_pool_queue_size']}/{sm['thread_pool_max_workers']}")
        report.append("")

        report.append("=" * 80)
        return "\n".join(report)

    def reset_metrics(self) -> None:
        """Reset all metrics to initial state."""
        with self._lock:
            self._frames_processed = 0
            self._frames_rejected = 0
            self._frame_processing_times.clear()
            self._frame_quality_issues.clear()

            self._detections_total = 0
            self._detection_confidence_scores.clear()
            self._vehicles_detected_per_frame.clear()
            self._detection_processing_times.clear()

            self._tracking_total_vehicles = 0
            self._tracking_active_vehicles = 0
            self._tracking_id_losses = 0
            self._tracking_reassignments = 0
            self._tracking_continuity_history.clear()

            self._plate_detections = 0
            self._plate_ocr_attempts = 0
            self._plate_ocr_successes = 0
            self._plate_confidence_scores.clear()
            self._plate_processing_times.clear()

            self._notifications_sent = 0
            self._notifications_successful = 0
            self._notifications_failed = 0
            self._notification_api_calls = 0

            self._memory_usage_history.clear()
            self.start_time = datetime.now()

    def export_metrics_json(self) -> Dict[str, Any]:
        """Export all metrics as JSON-serializable dictionary."""
        all_metrics = self.get_all_metrics()
        all_metrics["timestamp"] = datetime.now().isoformat()
        all_metrics["uptime_seconds"] = (datetime.now() - self.start_time).total_seconds()
        return all_metrics

    def get_trending_data(self) -> Dict[str, Any]:
        """Get historical trending data for charting/visualization."""
        with self._lock:
            return {
                "frame_processing_times": list(self._frame_processing_times),
                "detection_confidence_scores": list(self._detection_confidence_scores),
                "vehicles_detected_per_frame": list(self._vehicles_detected_per_frame),
                "detection_processing_times": list(self._detection_processing_times),
                "plate_confidence_scores": list(self._plate_confidence_scores),
                "plate_processing_times": list(self._plate_processing_times),
                "tracking_continuity_history": list(self._tracking_continuity_history),
                "memory_usage_history": list(self._memory_usage_history),
            }


# Global metrics instance
_global_metrics = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create global metrics collector instance."""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = MetricsCollector()
    return _global_metrics


def reset_global_metrics() -> None:
    """Reset global metrics collector."""
    global _global_metrics
    _global_metrics = MetricsCollector()
