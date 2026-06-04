"""
Example: Streaming camera with license plate display in corner
Optimized for real-time performance - no bounding boxes, only plate text
Includes frame quality validation to filter low-quality frames
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

    # To disable frame quality validation, uncomment:
    # tracker.frame_processor.set_quality_validation(False)
    
    print("\n[SETUP] Starting camera stream... (Press 'q' to quit)")
    
    RTSP_SOURCE = "rtsp://admin:MOVYKV@aicamera.serveminecraft.net:554/Streaming/Channels/101"
    VIDEO_SOURCE = r"video\0604.mp4"

    tracker.process_video_streaming(VIDEO_SOURCE)
    
    # **CRITICAL:** Wait for all pending tasks and send notifications before exit
    print("\n" + "="*80)
    print("[CLEANUP] Waiting for all pending tasks...")
    tracker.cleanup()
    
    # Print frame quality statistics
    quality_stats = tracker.frame_processor.get_quality_stats()
    print("\n" + "="*80)
    print("[QUALITY STATS] Frame processing summary:")
    print(f"  - Frames processed: {quality_stats['frames_processed']}")
    print(f"  - Frames rejected: {quality_stats['frames_rejected']}")
    print(f"  - Total frames: {quality_stats['total_frames']}")
    print(f"  - Rejection rate: {quality_stats['rejection_rate']:.1f}%")
    print(
        "\n[INFO] Higher rejection rate = filtering low-quality frames"
        "\n       Adjust thresholds if rejection rate > 30% or < 5%"
    )


if __name__ == "__main__":
    main()

