"""Test script to evaluate tracking ID stability and analyze parameter effectiveness."""

import cv2
import sys
from datetime import datetime
from collections import defaultdict
from VehicleDetectionTracker.frame_processor import FrameProcessor
from VehicleDetectionTracker.device_utils import get_device
from VehicleDetectionTracker.config_loader import get_detection_config  # Import config loader
from ultralytics import YOLO

def analyze_id_switches(id_history):
    """Analyze ID switching patterns."""
    print("\n" + "="*60)
    print("📊 TRACKING ID ANALYSIS REPORT")
    print("="*60)
    
    # Count ID switches
    switches = 0
    switch_frames = []
    prev_id = None
    
    for frame_idx, track_ids in id_history.items():
        if len(track_ids) > 0:
            current_id = track_ids[0]  # First detected ID
            if prev_id is not None and current_id != prev_id:
                switches += 1
                switch_frames.append(frame_idx)
            prev_id = current_id
    
    # Statistics
    total_frames = len(id_history)
    frames_with_detection = sum(1 for ids in id_history.values() if len(ids) > 0)
    unique_ids = set()
    for ids in id_history.values():
        unique_ids.update(ids)
    
    print(f"\n📈 Statistics:")
    print(f"   Total frames: {total_frames}")
    print(f"   Frames with vehicle detection: {frames_with_detection}")
    print(f"   Unique vehicle IDs detected: {len(unique_ids)}")
    print(f"   Total ID switches: {switches}")
    
    if frames_with_detection > 0:
        switch_rate = (switches / frames_with_detection) * 100
        print(f"   ID switch rate: {switch_rate:.2f}% (lower is better)")
    
    if switch_frames:
        print(f"\n🔴 ID switches occurred at frames: {switch_frames}")
    else:
        print(f"\n✅ No ID switches detected - EXCELLENT!")
    
    print(f"\n🎯 Unique IDs found: {sorted(list(unique_ids))}")
    
    # Detailed tracking per ID
    print(f"\n📍 Tracking duration per ID:")
    id_durations = defaultdict(list)
    current_id = None
    start_frame = 0
    
    for frame_idx, track_ids in sorted(id_history.items()):
        if len(track_ids) > 0:
            id_in_frame = track_ids[0]
            if current_id != id_in_frame:
                if current_id is not None:
                    duration = frame_idx - start_frame
                    id_durations[current_id].append(duration)
                current_id = id_in_frame
                start_frame = frame_idx
        elif current_id is not None:
            duration = frame_idx - start_frame
            id_durations[current_id].append(duration)
            current_id = None
    
    for vehicle_id in sorted(id_durations.keys()):
        durations = id_durations[vehicle_id]
        total_duration = sum(durations)
        occurrences = len(durations)
        print(f"   ID {vehicle_id}: {occurrences} segment(s), total {total_duration} frames")
    
    print("\n" + "="*60)
    return switches, len(unique_ids), switch_rate if frames_with_detection > 0 else 0

def test_tracking_from_video(video_path):
    """Test tracking from video file and analyze ID stability."""
    print(f"\n🎬 Testing tracking from: {video_path}")
    print(f"⏰ Started at: {datetime.now()}")
    
    # Initialize
    def log_func(msg):
        print(f"[TEST] {msg}")
    
    device = get_device(log_func)
    print(f"📱 Using device: {device}")
    
    model = YOLO("yolov8n.pt")
    # YOLO handles device automatically, no need for .to(device)
    
    processor = FrameProcessor(model, log_func)
    
    # Load vehicle classes from config
    detection_config = get_detection_config()
    vehicle_classes = detection_config.get("vehicle_classes", [2, 3, 5, 7, 8])
    print(f"🔬 Testing with vehicle classes: {vehicle_classes}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"📹 Video: {fps} FPS, {total_frames} total frames")
    
    # Track IDs across frames
    id_history = defaultdict(list)
    frame_count = 0
    
    print(f"\n🔍 Processing frames...")
    
    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Show progress every 30 frames
        if frame_count % 30 == 0:
            print(f"   Frame {frame_count}/{total_frames} ({(frame_count/total_frames)*100:.1f}%)")
        
        # Run detection
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
            conf=0.4,
            iou=0.4,
            imgsz=1280,
            classes=vehicle_classes,  # Use classes from config
        )
        
        # Extract track IDs
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            id_history[frame_count] = track_ids
    
    cap.release()
    
    print(f"\n✅ Video processing completed")
    print(f"⏰ Finished at: {datetime.now()}")
    
    # Analyze results
    switches, unique_ids, switch_rate = analyze_id_switches(id_history)
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if switch_rate < 5:
        print(f"   ✅ Current parameters are GOOD! Only {switch_rate:.2f}% ID switches")
    elif switch_rate < 15:
        print(f"   ⚠️  Parameters are acceptable, but can be improved")
    else:
        print(f"   ❌ Too many ID switches ({switch_rate:.2f}%). Need to adjust:")
        print(f"      - Decrease track_high_thresh (currently 0.15)")
        print(f"      - Decrease track_low_thresh (currently 0.001)")
        print(f"      - Increase track_buffer (currently 300)")
        print(f"      - Increase match_thresh (currently 0.8)")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_file = sys.argv[1]
        test_tracking_from_video(video_file)
    else:
        print(f"Usage: python test_tracking_id.py <video_file>")
        print(f"Example: python test_tracking_id.py video.mp4")
        print(f"\nThis script will:")
        print(f"  1. Process the video with current tracker settings")
        print(f"  2. Count ID switches and analyze tracking stability")
        print(f"  3. Provide recommendations for parameter tuning")
