"""
Example: Streaming camera with license plate display in corner
Optimized for real-time performance - no bounding boxes, only plate text
"""

import os
import warnings

# Set environment variables BEFORE importing any libraries
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["PADDLE_EXTENSION_COMPILE_FLAG"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from VehicleDetectionTracker.VehicleDetectionTracker import VehicleDetectionTracker


def main():
    # Initialize tracker
    # Để tránh mờ hình, không resize (giữ nguyên kích thước gốc từ camera/RTSP).
    tracker = VehicleDetectionTracker()

    # Option 1: Process camera/webcam stream (press 'q' to quit)
    print("Starting camera stream... (Press 'q' to quit)")
    RTSP_SOURCE = "rtsp://admin:MOVYKV@aicamera.ddns.net:554/Streaming/Channels/101"
    VIDEO_SOURCE = r"video\n.mp4"

    tracker.process_video_streaming(
        VIDEO_SOURCE
    )  # 0 for webcam, or use RTSP_SOURCE for live stream
    
    # **CRITICAL:** Wait for all pending tasks and send notifications before exit
    print("\n" + "="*80)
    tracker.cleanup()


if __name__ == "__main__":
    main()
