"""
Example: Streaming camera with license plate display in corner
Optimized for real-time performance - no bounding boxes, only plate text
"""

from VehicleDetectionTracker.VehicleDetectionTracker import VehicleDetectionTracker

def main():
    # Initialize tracker
    tracker = VehicleDetectionTracker(stream_frame_size=(1280, 720))  # Resize frames to 1280x720 for performance
    
    # Option 1: Process camera/webcam stream (press 'q' to quit)
    print("Starting camera stream... (Press 'q' to quit)")
    RTSP_SOURCE = "rtsp://admin:MOVYKV@aicamera.ddns.net:554/Streaming/Channels/101" 
    VIDEO_SOURCE = r"F:\OCR\License-Plate-Recognition\video\5.mp4"
    tracker.process_video_streaming(VIDEO_SOURCE)  # 0 for webcam, or use video file path
    
    # Cleanup
    tracker.cleanup()

if __name__ == "__main__":
    main()

