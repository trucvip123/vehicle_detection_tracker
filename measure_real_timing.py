#!/usr/bin/env python3
"""
Direct Timing Measurement - Patches existing code with timing instrumentation
Real-world timing measurement without code modification.

Usage:
    python measure_real_timing.py [video] [frames]
    
Example:
    python measure_real_timing.py video/h.mp4 30
"""

import time
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import cv2

# Timing collector
class TimingCollector:
    def __init__(self):
        self.timings = defaultdict(list)
        self.frame_timings = []
        self.current_frame = 0
    
    def record(self, module_name, duration_ms):
        """Record timing for a module."""
        self.timings[module_name].append(duration_ms)
    
    def start_frame(self):
        """Mark frame start."""
        self.frame_start = time.perf_counter()
        self.current_frame += 1
    
    def end_frame(self):
        """Mark frame end."""
        duration = (time.perf_counter() - self.frame_start) * 1000
        self.frame_timings.append(duration)
    
    def print_summary(self):
        """Print timing summary."""
        print("\n" + "=" * 80)
        print("TIMING MEASUREMENT RESULTS")
        print("=" * 80)
        
        if not self.frame_timings:
            print("❌ No data collected")
            return
        
        print(f"\nFrames analyzed: {len(self.frame_timings)}")
        print(f"Average frame time: {sum(self.frame_timings) / len(self.frame_timings):.2f}ms")
        
        print(f"\n--- Module Timing Breakdown ---\n")
        
        total_time = 0
        modules_data = []
        
        for module, times in sorted(self.timings.items()):
            if times:
                avg = sum(times) / len(times)
                total_t = sum(times)
                total_time += total_t
                modules_data.append((module, avg, min(times), max(times), total_t, len(times)))
        
        # Sort by total time
        modules_data.sort(key=lambda x: x[4], reverse=True)
        
        for idx, (module, avg, mn, mx, total, count) in enumerate(modules_data, 1):
            pct = (total / (sum(self.frame_timings) * 0.9) * 100) if self.frame_timings else 0
            print(f"{idx}. {module}")
            print(f"   Average: {avg:.2f}ms (count: {count})")
            print(f"   Min/Max: {mn:.2f}ms / {mx:.2f}ms")
            print(f"   Total: {total:.2f}ms")
            if pct > 0:
                print(f"   Percentage: {pct:.1f}%")
            print()
        
        print("=" * 80)

# Global timing collector
collector = TimingCollector()


def measure_module(module_name):
    """Decorator to measure module execution time."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            finally:
                duration = (time.perf_counter() - start) * 1000
                collector.record(module_name, duration)
            return result
        return wrapper
    return decorator


# Patch functions
def setup_timing_patches():
    """Apply timing patches to key functions."""
    import VehicleDetectionTracker.frame_processor as fp
    import VehicleDetectionTracker.plate_processor as pp
    import VehicleDetectionTracker.plate_utils as pu
    
    # Save original functions
    original_process_frame = fp.FrameProcessor.process_frame_streaming
    original_submit_plate = pp.PlateProcessor.submit_plate_processing
    
    # Patch process_frame_streaming
    @measure_module("Vehicle Detection")
    def patched_process_frame(self, frame, frame_timestamp):
        return original_process_frame(self, frame, frame_timestamp)
    
    # Patch submit_plate_processing
    @measure_module("Plate Processing")
    def patched_submit_plate(self, track_id, vehicle_frame, direction_label, 
                            frame_timestamp, timestamp_str, vehicle_dir):
        return original_submit_plate(self, track_id, vehicle_frame, direction_label,
                                    frame_timestamp, timestamp_str, vehicle_dir)
    
    # Apply patches
    fp.FrameProcessor.process_frame_streaming = patched_process_frame
    pp.PlateProcessor.submit_plate_processing = patched_submit_plate
    
    print("✓ Timing patches applied")


def run_measurement(video_path, num_frames):
    """Run real-time measurement."""
    
    print("=" * 80)
    print("REAL-TIME TIMING MEASUREMENT")
    print("=" * 80)
    print(f"Video: {video_path}")
    print(f"Frames: {num_frames}")
    print()
    
    # Setup patches
    setup_timing_patches()
    
    # Import after patching
    from VehicleDetectionTracker.VehicleDetectionTracker import VehicleDetectionTracker
    from VehicleDetectionTracker.device_utils import get_device
    from VehicleDetectionTracker.config_loader import get_config
    
    # Initialize
    device, _ = get_device(print)
    config = get_config()
    
    print("Initializing detection system...")
    detector = VehicleDetectionTracker(
        model_path="yolov8n.pt"
    )
    print(f"✓ System initialized\n")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps:.1f}")
    print(f"Processing frames...")
    print("-" * 80)
    
    frame_count = 0
    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        collector.start_frame()
        
        # Process frame
        try:
            detector.process_frame_streaming(frame, datetime.now())
            print(f"Frame {frame_count}: Processed")
        except Exception as e:
            print(f"Frame {frame_count}: Error - {e}")
        
        collector.end_frame()
    
    cap.release()
    
    # Print results
    collector.print_summary()


if __name__ == "__main__":
    from datetime import datetime
    
    video = sys.argv[1] if len(sys.argv) > 1 else "video/h.mp4"
    frames = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    
    if not Path(video).exists():
        print(f"❌ Video not found: {video}")
        print("\nAvailable videos:")
        for f in Path("video").glob("*.mp4"):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name} ({size_mb:.1f}MB)")
        sys.exit(1)
    
    run_measurement(video, frames)
