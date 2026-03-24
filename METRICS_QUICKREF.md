## Metrics Quick Reference

### 30-Second Overview

```python
from VehicleDetectionTracker import VehicleDetectionTracker

tracker = VehicleDetectionTracker()

# Metrics automatically collected during:
tracker.process_video_streaming("video.mp4")

# View metrics at ANY time:
print(tracker.get_metrics_summary())  # 👈 Print everything

# Or access specific metrics:
m = tracker.get_metrics()
print(f"FPS: {m['frame']['estimated_fps']:.1f}")
print(f"OCR success: {m['plate']['ocr_success_rate']:.1f}%")
print(f"Memory: {m['system']['current_memory_mb']:.1f} MB")

tracker.cleanup()
```

### Metric Categories

| Category | Key Metrics | Use Case |
|----------|------------|----------|
| **frame** | `estimated_fps`, `rejection_rate` | Real-time performance |
| **detection** | `avg_detection_confidence`, `avg_detections_per_frame` | Detection quality |
| **tracking** | `avg_tracking_continuity`, `loss_rate` | Tracking stability |
| **plate** | `ocr_success_rate`, `avg_plate_confidence` | OCR performance |
| **notification** | `success_rate`, `api_calls_made` | Notification delivery |
| **system** | `current_memory_mb`, `thread_pool_queue_size` | System health |

### Quick Commands

```python
# Get everything
all_metrics = tracker.get_metrics()

# Get one category
detection_metrics = tracker.get_metrics()["detection"]

# Print readable report
print(tracker.get_metrics_summary())

# Export as JSON
import json
metrics_dict = tracker.export_metrics_json()
json.dump(metrics_dict, open("metrics.json", "w"))

# Get historical data for plot
trending = tracker.get_trending_data()
# Contains: frame_processing_times, detection_confidence_scores, etc.

# Reset metrics for new session
tracker.reset_metrics()
```

### Common Patterns

#### Monitor FPS
```python
frame_m = tracker.get_metrics()["frame"]
print(f"FPS: {frame_m['estimated_fps']:.1f}")
print(f"Avg frame time: {frame_m['avg_processing_time_ms']:.1f}ms")
```

#### Check Quality
```python
frame_m = tracker.get_metrics()["frame"]
print(f"Frame rejection rate: {frame_m['rejection_rate']:.1f}%")
print(f"Quality issues: {frame_m['quality_issues']}")
```

#### Monitor Detection Accuracy
```python
det_m = tracker.get_metrics()["detection"]
print(f"Confidence: {det_m['avg_detection_confidence']:.3f}")
print(f"Vehicles/frame: {det_m['avg_detections_per_frame']:.1f}")
```

#### Check OCR Performance
```python
plate_m = tracker.get_metrics()["plate"]
print(f"OCR success rate: {plate_m['ocr_success_rate']:.1f}%")
print(f"Total plates read: {plate_m['plate_ocr_successes']}")
```

#### Monitor Notifications
```python
notif_m = tracker.get_metrics()["notification"]
print(f"Success rate: {notif_m['success_rate']:.1f}%")
print(f"API calls: {notif_m['api_calls_made']}")
```

#### Check System Health
```python
sys_m = tracker.get_metrics()["system"]
print(f"Memory: {sys_m['current_memory_mb']:.1f} MB (peak: {sys_m['peak_memory_mb']:.1f})")
print(f"Uptime: {sys_m['uptime_seconds']/3600:.1f} hours")
print(f"Thread queue: {sys_m['thread_pool_queue_size']}/{sys_m['thread_pool_max_workers']}")
```

### Monitoring Template

```python
import time

tracker = VehicleDetectionTracker()
tracker.process_video_streaming("video.mp4", display_window=True)

# Print metrics every 60 seconds (in separate thread)
import threading

def monitor():
    while True:
        time.sleep(60)
        print("\n" + "="*80)
        print(tracker.get_metrics_summary())
        print("="*80)

thread = threading.Thread(target=monitor, daemon=True)
thread.start()

# Main processing continues...
# Tracker will run until user closes window
tracker.cleanup()
```

### Full Report Structure

```python
metrics = {
    "frame": {
        "frames_processed": 1234,
        "frames_rejected": 56,
        "rejection_rate": 4.3,
        "avg_processing_time_ms": 12.5,
        "estimated_fps": 80.0,
        "quality_issues": {"Too dark": 45, "Too blurry": 11}
    },
    "detection": {
        "total_detections": 5678,
        "avg_detection_confidence": 0.856,
        "avg_detections_per_frame": 4.6,
        "avg_detection_time_ms": 45.3,
        "max_detection_time_ms": 123.4
    },
    "tracking": {
        "total_vehicles_tracked": 89,
        "active_vehicles": 12,
        "tracking_id_losses": 2,
        "tracking_reassignments": 0,
        "loss_rate": 2.2,
        "avg_tracking_continuity": 0.978
    },
    "plate": {
        "total_plate_detections": 342,
        "plate_ocr_attempts": 325,
        "plate_ocr_successes": 298,
        "ocr_success_rate": 91.7,
        "avg_plate_confidence": 0.923,
        "avg_plate_processing_time_ms": 32.1
    },
    "notification": {
        "notifications_sent": 87,
        "notifications_successful": 85,
        "notifications_failed": 2,
        "success_rate": 97.7,
        "api_calls_made": 87
    },
    "system": {
        "uptime_seconds": 3600.5,
        "thread_pool_queue_size": 2,
        "thread_pool_max_workers": 20,
        "current_memory_mb": 256.3,
        "avg_memory_mb": 245.1,
        "peak_memory_mb": 287.6
    }
}
```

### Troubleshooting

**Q: Why is FPS low?**
- Check: `avg_detection_time_ms` + `avg_plate_processing_time_ms`
- Check: `frame_rejection_rate` (too high = wasted cycles)
- Check: `thread_pool_queue_size` (backlog = bottleneck)

**Q: Why is OCR not working?**
- Check: `ocr_success_rate` (see if it's actually failing)
- Check: `total_plate_detections` (no detections = bigger problem)
- Check: `avg_plate_confidence` (low confidence = quality issue)

**Q: Is memory leaking?**
- Check: `current_memory_mb` trend (should be stable)
- Check: `peak_memory_mb` (should not keep increasing)
- Check: `tracking_reassignments` (high = many versioned IDs stored)

**Q: Are notifications working?**
- Check: `notifications_sent` (none = vehicles not disappearing)
- Check: `success_rate` (should be >95%)
- Check: `api_calls_made` (should match notifications_sent)

### Integration with External Tools

```python
# Prometheus metrics
from prometheus_client import Counter, Gauge

detected_vehicles = Counter('detected_vehicles', 'Total vehicles')
fps_gauge = Gauge('tracker_fps', 'FPS')

metrics = tracker.get_metrics()
detected_vehicles.inc(metrics['detection']['total_detections'])
fps_gauge.set(metrics['frame']['estimated_fps'])

# InfluxDB logging
from influxdb_client import InfluxDBClient
client = InfluxDBClient(url="http://localhost:8086", token="token")
write_api = client.write_api(write_options=SYNCHRONOUS)

metrics_json = tracker.export_metrics_json()
write_api.write(bucket="tracker", record=metrics_json)

# Grafana dashboard
# Create dashboard pointing to Prometheus/InfluxDB data
# Query: tracker_fps, detected_vehicles, etc.
```

### Key Thresholds to Monitor

| Metric | Good | Acceptable | Needs Attention |
|--------|------|-----------|-----------------|
| FPS | > 25 | 15-25 | < 15 |
| Frame rejection | < 5% | 5-10% | > 10% |
| Detection confidence | > 0.85 | 0.75-0.85 | < 0.75 |
| OCR success rate | > 90% | 70-90% | < 70% |
| Notification success | > 95% | 85-95% | < 85% |
| Memory growth | Stable | Slow growth | Rapid growth |
| Tracking loss rate | < 5% | 5-10% | > 10% |

### Files

- Full Guide: [METRICS_GUIDE.md](METRICS_GUIDE.md)
- Technical Details: [METRICS_IMPLEMENTATION.md](METRICS_IMPLEMENTATION.md)
- Test Suite: [tests/test_metrics.py](tests/test_metrics.py)
- Source Code: [VehicleDetectionTracker/metrics.py](VehicleDetectionTracker/metrics.py)

---

**Last Updated**: 2026-03-24  
**Version**: 1.0  
**Status**: Production Ready ✅
