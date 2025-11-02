"""
Example: Streaming camera with license plate display in corner
Optimized for real-time performance - no bounding boxes, only plate text
"""

from VehicleDetectionTracker.VehicleDetectionTracker import VehicleDetectionTracker

def main():
    # Initialize tracker
    tracker = VehicleDetectionTracker()
    
    # Option 1: Process camera/webcam stream (press 'q' to quit)
    print("Starting camera stream... (Press 'q' to quit)")
    tracker.process_video_streaming(r"F:\OCR\License-Plate-Recognition\video\3.mp4")  # 0 for webcam, or use video file path
    
    # Option 2: Process video file
    # video_path = "path/to/your/video.mp4"
    # tracker.process_video_streaming(video_path)
    
    # Cleanup
    tracker.cleanup()

if __name__ == "__main__":
    main()

