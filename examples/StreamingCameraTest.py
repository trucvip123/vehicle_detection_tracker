"""
Example: Streaming camera with license plate display in corner
Optimized for real-time performance - no bounding boxes, only plate text

All models are initialized ONCE before streaming starts to avoid delays.
"""

from VehicleDetectionTracker.VehicleDetectionTracker import VehicleDetectionTracker

def main():
    # Initialize tracker with all models loaded immediately (default)
    # This ensures no model loading delays during streaming
    print("Initializing tracker with all models...")
    tracker = VehicleDetectionTracker(initialize_all_models=True)
    
    # Check initialization status
    status = tracker.get_initialization_status()
    print(f"Model status: {status}")
    
    # Option 1: Process camera/webcam stream (press 'q' to quit)
    print("\nStarting camera stream... (Press 'q' to quit)")
    tracker.process_video_streaming(0)  # 0 for webcam, or use video file path
    
    # Option 2: Process video file
    # video_path = "path/to/your/video.mp4"
    # tracker.process_video_streaming(video_path)
    
    # Cleanup
    tracker.cleanup()
    
    # Alternative: Lazy loading (not recommended for streaming)
    # tracker = VehicleDetectionTracker(initialize_all_models=False)
    # tracker.ensure_all_models_initialized()  # Call before streaming
    # tracker.process_video_streaming(0)

if __name__ == "__main__":
    main()

