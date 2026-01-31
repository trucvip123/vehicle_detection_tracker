"""
Example: Streaming camera with license plate display in corner
Optimized for real-time performance - no bounding boxes, only plate text
"""

from VehicleDetectionTracker.VehicleDetectionTracker import VehicleDetectionTracker


def main():
    # Initialize tracker
    # Để tránh mờ hình, không resize (giữ nguyên kích thước gốc từ camera/RTSP).
    tracker = VehicleDetectionTracker()

    # Option 1: Process camera/webcam stream (press 'q' to quit)
    print("Starting camera stream... (Press 'q' to quit)")
    RTSP_SOURCE = "rtsp://admin:MOVYKV@aicamera.ddns.net:554/Streaming/Channels/101"
    VIDEO_SOURCE = r"video\k.mp4"

    try:
        tracker.process_video_streaming(
            RTSP_SOURCE
        )  # 0 for webcam, or use video file path
    finally:
        # Cleanup
        tracker.cleanup()


if __name__ == "__main__":
    main()
