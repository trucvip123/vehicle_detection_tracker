"""Comprehensive tests for metrics and monitoring module."""

import pytest
import time
import threading
from datetime import datetime, timedelta
from VehicleDetectionTracker.metrics import MetricsCollector, get_metrics_collector, reset_global_metrics


class TestMetricsCollectorFrameMetrics:
    """Test frame processing metrics collection."""

    def test_record_frame_processed_basic(self):
        """Test basic frame processing recording."""
        metrics = MetricsCollector()
        metrics.record_frame_processed(0.016, is_quality_rejected=False)
        
        frame_metrics = metrics.get_frame_metrics()
        assert frame_metrics["frames_processed"] == 1
        assert frame_metrics["frames_rejected"] == 0
        assert frame_metrics["rejection_rate"] == 0.0

    def test_record_multiple_frames(self):
        """Test recording multiple frames."""
        metrics = MetricsCollector()
        
        # Record 10 good frames
        for i in range(10):
            metrics.record_frame_processed(0.010, is_quality_rejected=False)
        
        frame_metrics = metrics.get_frame_metrics()
        assert frame_metrics["frames_processed"] == 10
        assert frame_metrics["frames_rejected"] == 0
        assert 0 < frame_metrics["avg_processing_time_ms"] < 20

    def test_record_frame_rejected(self):
        """Test recording rejected frames."""
        metrics = MetricsCollector()
        
        # Record 5 good frames
        for _ in range(5):
            metrics.record_frame_processed(0.010, is_quality_rejected=False)
        
        # Record 5 rejected frames
        for _ in range(5):
            metrics.record_frame_processed(0.005, is_quality_rejected=True, issues=["Too dark"])
        
        frame_metrics = metrics.get_frame_metrics()
        assert frame_metrics["frames_processed"] == 5
        assert frame_metrics["frames_rejected"] == 5
        assert frame_metrics["rejection_rate"] == 50.0

    def test_quality_issues_tracking(self):
        """Test tracking of quality issues."""
        metrics = MetricsCollector()
        
        metrics.record_frame_processed(0.010, is_quality_rejected=True, issues=["Too dark", "Low contrast"])
        metrics.record_frame_processed(0.010, is_quality_rejected=True, issues=["Too dark"])
        metrics.record_frame_processed(0.010, is_quality_rejected=True, issues=["Too blurry"])
        
        frame_metrics = metrics.get_frame_metrics()
        issues = frame_metrics["quality_issues"]
        assert issues.get("Too dark", 0) == 2
        assert issues.get("Low contrast", 0) == 1
        assert issues.get("Too blurry", 0) == 1

    def test_fps_calculation(self):
        """Test FPS calculation from processing times."""
        metrics = MetricsCollector()
        
        # Record frames with ~30ms processing time (should be ~33 FPS)
        for _ in range(10):
            metrics.record_frame_processed(0.033)
        
        frame_metrics = metrics.get_frame_metrics()
        assert 20 < frame_metrics["estimated_fps"] < 40  # Approximately 30 FPS


class TestMetricsCollectorDetectionMetrics:
    """Test detection performance metrics."""

    def test_record_detection_basic(self):
        """Test basic detection recording."""
        metrics = MetricsCollector()
        metrics.record_detection(confidence=0.8, vehicles_detected_count=3, processing_time=0.050)
        
        detection_metrics = metrics.get_detection_metrics()
        assert detection_metrics["total_detections"] == 3
        assert detection_metrics["avg_detection_confidence"] == 0.8
        assert detection_metrics["avg_detections_per_frame"] == 3.0

    def test_multiple_detections(self):
        """Test multiple detection recordings."""
        metrics = MetricsCollector()
        
        metrics.record_detection(confidence=0.85, vehicles_detected_count=5, processing_time=0.045)
        metrics.record_detection(confidence=0.75, vehicles_detected_count=3, processing_time=0.040)
        metrics.record_detection(confidence=0.80, vehicles_detected_count=4, processing_time=0.042)
        
        detection_metrics = metrics.get_detection_metrics()
        assert detection_metrics["total_detections"] == 12  # 5 + 3 + 4
        assert 0.75 < detection_metrics["avg_detection_confidence"] < 0.85
        assert 3.5 < detection_metrics["avg_detections_per_frame"] < 4.5

    def test_detection_processing_time_tracking(self):
        """Test tracking of detection processing times."""
        metrics = MetricsCollector()
        
        times = [0.040, 0.045, 0.050, 0.048]
        for processing_time in times:
            metrics.record_detection(confidence=0.8, vehicles_detected_count=1, processing_time=processing_time)
        
        detection_metrics = metrics.get_detection_metrics()
        assert 44 < detection_metrics["avg_detection_time_ms"] < 48
        assert 48 < detection_metrics["max_detection_time_ms"] < 52


class TestMetricsCollectorTrackingMetrics:
    """Test tracking metrics."""

    def test_record_tracking_update_basic(self):
        """Test basic tracking update."""
        metrics = MetricsCollector()
        metrics.record_tracking_update(active_vehicles=5, new_vehicle=True)
        
        tracking_metrics = metrics.get_tracking_metrics()
        assert tracking_metrics["total_vehicles_tracked"] == 1
        assert tracking_metrics["active_vehicles"] == 5

    def test_tracking_losses(self):
        """Test tracking loss recording."""
        metrics = MetricsCollector()
        
        # Track 10 vehicles
        for i in range(10):
            metrics.record_tracking_update(active_vehicles=10-i, new_vehicle=True)
        
        # Lose tracking on 2 vehicles
        metrics.record_tracking_update(active_vehicles=8, tracking_lost=True)
        metrics.record_tracking_update(active_vehicles=8, tracking_lost=True)
        
        tracking_metrics = metrics.get_tracking_metrics()
        assert tracking_metrics["total_vehicles_tracked"] == 10
        assert tracking_metrics["tracking_id_losses"] == 2
        assert tracking_metrics["loss_rate"] == 20.0

    def test_tracking_reassignments(self):
        """Test tracking reassignment tracking."""
        metrics = MetricsCollector()
        
        for i in range(5):
            metrics.record_tracking_update(active_vehicles=5, new_vehicle=True, reassigned=(i % 2 == 0))
        
        tracking_metrics = metrics.get_tracking_metrics()
        assert tracking_metrics["total_vehicles_tracked"] == 5
        assert tracking_metrics["tracking_reassignments"] == 3  # 0, 2, 4


class TestMetricsCollectorPlateMetrics:
    """Test plate processing metrics."""

    def test_record_plate_detection_basic(self):
        """Test basic plate detection."""
        metrics = MetricsCollector()
        metrics.record_plate_detection(confidence=0.9, processing_time=0.030)
        
        plate_metrics = metrics.get_plate_metrics()
        assert plate_metrics["total_plate_detections"] == 1

    def test_ocr_success_rate(self):
        """Test OCR success rate tracking."""
        metrics = MetricsCollector()
        
        # Record 10 plate detections with OCR
        for i in range(10):
            success = i < 8  # First 8 succeed
            metrics.record_plate_detection(
                confidence=0.85,
                processing_time=0.035,
                ocr_attempted=True,
                ocr_success=success
            )
        
        plate_metrics = metrics.get_plate_metrics()
        assert plate_metrics["plate_ocr_attempts"] == 10
        assert plate_metrics["plate_ocr_successes"] == 8
        assert plate_metrics["ocr_success_rate"] == 80.0

    def test_plate_confidence_tracking(self):
        """Test plate detection confidence tracking."""
        metrics = MetricsCollector()
        
        confidences = [0.85, 0.90, 0.88, 0.92, 0.86]
        for conf in confidences:
            metrics.record_plate_detection(confidence=conf, processing_time=0.035)
        
        plate_metrics = metrics.get_plate_metrics()
        assert 0.88 < plate_metrics["avg_plate_confidence"] < 0.90


class TestMetricsCollectorNotificationMetrics:
    """Test notification metrics."""

    def test_record_notification_sent(self):
        """Test recording sent notifications."""
        metrics = MetricsCollector()
        
        metrics.record_notification_sent(success=True, api_call=True)
        metrics.record_notification_sent(success=True, api_call=True)
        metrics.record_notification_sent(success=False, api_call=True)
        
        notification_metrics = metrics.get_notification_metrics()
        assert notification_metrics["notifications_sent"] == 3
        assert notification_metrics["notifications_successful"] == 2
        assert notification_metrics["notifications_failed"] == 1
        assert notification_metrics["success_rate"] == pytest.approx(66.67, rel=1)
        assert notification_metrics["api_calls_made"] == 3


class TestMetricsCollectorSystemMetrics:
    """Test system health metrics."""

    def test_record_system_metrics(self):
        """Test recording system metrics."""
        metrics = MetricsCollector()
        
        metrics.record_system_metrics(queue_size=5, max_workers=20)
        
        system_metrics = metrics.get_system_metrics()
        assert system_metrics["thread_pool_queue_size"] == 5
        assert system_metrics["thread_pool_max_workers"] == 20
        assert system_metrics["current_memory_mb"] > 0

    def test_memory_tracking(self):
        """Test memory usage tracking."""
        metrics = MetricsCollector()
        
        for _ in range(5):
            metrics.record_system_metrics(queue_size=0, max_workers=4)
            time.sleep(0.01)
        
        system_metrics = metrics.get_system_metrics()
        assert system_metrics["current_memory_mb"] > 0
        assert system_metrics["avg_memory_mb"] > 0
        assert system_metrics["peak_memory_mb"] > 0


class TestMetricsReporting:
    """Test metrics reporting and export."""

    def test_summary_report_generation(self):
        """Test generation of summary report."""
        metrics = MetricsCollector()
        
        # Record various metrics
        metrics.record_frame_processed(0.016)
        metrics.record_detection(confidence=0.8, vehicles_detected_count=3, processing_time=0.050)
        metrics.record_tracking_update(active_vehicles=1, new_vehicle=True)
        metrics.record_plate_detection(confidence=0.9, processing_time=0.030, ocr_attempted=True, ocr_success=True)
        metrics.record_notification_sent(success=True, api_call=True)
        metrics.record_system_metrics(queue_size=2, max_workers=4)
        
        report = metrics.generate_summary_report()
        
        assert "VEHICLE DETECTION TRACKER - METRICS SUMMARY" in report
        assert "[FRAME PROCESSING]" in report
        assert "[DETECTION PERFORMANCE]" in report
        assert "[TRACKING METRICS]" in report
        assert "[PLATE PROCESSING]" in report
        assert "[NOTIFICATIONS]" in report
        assert "[SYSTEM HEALTH]" in report

    def test_export_metrics_json(self):
        """Test JSON export of metrics."""
        metrics = MetricsCollector()
        
        metrics.record_frame_processed(0.016)
        metrics.record_detection(confidence=0.8, vehicles_detected_count=3, processing_time=0.050)
        
        json_export = metrics.export_metrics_json()
        
        assert "timestamp" in json_export
        assert "uptime_seconds" in json_export
        assert "frame" in json_export
        assert "detection" in json_export

    def test_get_trending_data(self):
        """Test getting historical trending data."""
        metrics = MetricsCollector()
        
        # Record multiple samples
        for i in range(10):
            metrics.record_frame_processed(0.010 + i * 0.001)
            metrics.record_detection(confidence=0.8 - i * 0.01, vehicles_detected_count=5, processing_time=0.050)
        
        trending = metrics.get_trending_data()
        
        assert len(trending["frame_processing_times"]) == 10
        assert len(trending["detection_confidence_scores"]) == 10
        assert len(trending["vehicles_detected_per_frame"]) == 10


class TestMetricsReset:
    """Test metrics reset functionality."""

    def test_reset_metrics(self):
        """Test resetting all metrics."""
        metrics = MetricsCollector()
        
        # Record data
        for _ in range(10):
            metrics.record_frame_processed(0.010)
        metrics.record_detection(confidence=0.8, vehicles_detected_count=5, processing_time=0.050)
        
        # Verify data is recorded
        assert metrics.get_frame_metrics()["frames_processed"] == 10
        assert metrics.get_detection_metrics()["total_detections"] == 5
        
        # Reset
        metrics.reset_metrics()
        
        # Verify reset
        assert metrics.get_frame_metrics()["frames_processed"] == 0
        assert metrics.get_detection_metrics()["total_detections"] == 0


class TestThreadSafety:
    """Test thread safety of metrics collection."""

    def test_concurrent_frame_recording(self):
        """Test concurrent frame recording from multiple threads."""
        metrics = MetricsCollector()
        num_threads = 5
        frames_per_thread = 100
        
        def record_frames():
            for _ in range(frames_per_thread):
                metrics.record_frame_processed(0.010)
        
        threads = [threading.Thread(target=record_frames) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should have total of num_threads * frames_per_thread frames
        frame_metrics = metrics.get_frame_metrics()
        assert frame_metrics["frames_processed"] == num_threads * frames_per_thread

    def test_concurrent_metrics_access(self):
        """Test concurrent access to metrics from multiple threads."""
        metrics = MetricsCollector()
        errors = []
        
        def record_and_read():
            try:
                for _ in range(50):
                    metrics.record_frame_processed(0.010)
                    metrics.record_detection(confidence=0.8, vehicles_detected_count=1, processing_time=0.050)
                    _ = metrics.get_frame_metrics()
                    _ = metrics.get_detection_metrics()
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=record_and_read) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Thread errors: {errors}"


class TestGlobalMetricsCollector:
    """Test global metrics collector singleton."""

    def test_get_global_metrics(self):
        """Test getting global metrics collector."""
        reset_global_metrics()
        
        metrics1 = get_metrics_collector()
        metrics2 = get_metrics_collector()
        
        assert metrics1 is metrics2

    def test_global_metrics_persist(self):
        """Test that global metrics persist across calls."""
        reset_global_metrics()
        
        metrics = get_metrics_collector()
        metrics.record_frame_processed(0.010)
        
        # Get global metrics again
        metrics2 = get_metrics_collector()
        frame_metrics = metrics2.get_frame_metrics()
        
        assert frame_metrics["frames_processed"] == 1


class TestMetricsConsistency:
    """Test consistency between different metrics access methods."""

    def test_all_metrics_consistency(self):
        """Test that get_all_metrics returns consistent data."""
        metrics = MetricsCollector()
        
        # Record various metrics
        metrics.record_frame_processed(0.016)
        metrics.record_detection(confidence=0.8, vehicles_detected_count=3, processing_time=0.050)
        metrics.record_tracking_update(active_vehicles=1, new_vehicle=True)
        metrics.record_plate_detection(confidence=0.9, processing_time=0.030, ocr_attempted=True, ocr_success=True)
        metrics.record_notification_sent(success=True, api_call=True)
        
        # Get all metrics
        all_metrics = metrics.get_all_metrics()
        
        # Verify structure
        assert "frame" in all_metrics
        assert "detection" in all_metrics
        assert "tracking" in all_metrics
        assert "plate" in all_metrics
        assert "notification" in all_metrics
        assert "system" in all_metrics
        
        # Verify individual access matches all_metrics
        frame_metrics = metrics.get_frame_metrics()
        assert all_metrics["frame"] == frame_metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
