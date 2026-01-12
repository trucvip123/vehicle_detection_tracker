"""
Example: Streaming camera with license plate display in corner
Optimized for real-time performance - no bounding boxes, only plate text
"""

from VehicleDetectionTracker.VehicleDetectionTracker import VehicleDetectionTracker
import threading
import time
from datetime import datetime, timedelta


def start_daily_summary_scheduler(tracker, hour=18, minute=0):
    """Start a background thread that runs daily at given hour:minute.

    The thread is a daemon so it doesn't block process exit.
    It calls `tracker.plate_processor.save_daily_vehicle_summary()` each day.
    """

    def _scheduler_loop():
        while True:
            now = datetime.now()
            target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if target <= now:
                target = target + timedelta(days=1)
            wait_seconds = (target - now).total_seconds()
            print(
                f"[Scheduler] Waiting {int(wait_seconds)}s until next daily summary at {hour:02d}:{minute:02d}"
            )
            time.sleep(wait_seconds)
            try:
                print(f"[Scheduler] Running daily summary at {datetime.now()}")
                # Call the plate processor summary (uses current date by default)
                tracker.plate_processor.save_daily_vehicle_summary()
            except Exception as e:
                print(f"[Scheduler] Error sending daily summary: {e}")
            # small sleep to avoid immediate re-run in case of clock adjustments
            time.sleep(1)

    t = threading.Thread(
        target=_scheduler_loop, daemon=True, name="DailySummaryScheduler"
    )
    t.start()


def main():
    # Initialize tracker
    # Để tránh mờ hình, không resize (giữ nguyên kích thước gốc từ camera/RTSP).
    tracker = VehicleDetectionTracker()

    # Start daily summary scheduler (runs in background at 18:00)
    start_daily_summary_scheduler(tracker, hour=18, minute=0)

    # Option 1: Process camera/webcam stream (press 'q' to quit)
    print("Starting camera stream... (Press 'q' to quit)")
    RTSP_SOURCE = "rtsp://admin:MOVYKV@aicamera.gotdns.ch:554/Streaming/Channels/101"
    VIDEO_SOURCE = r"video\e.mp4"

    try:
        tracker.process_video_streaming(
            RTSP_SOURCE
        )  # 0 for webcam, or use video file path
    finally:
        # Cleanup
        tracker.cleanup()


if __name__ == "__main__":
    main()
