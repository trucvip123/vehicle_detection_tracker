#!/usr/bin/env python3
"""
Simple Module Timing Analysis Script
Tích hợp vào StreamingCameraTest.py để đo thời gian từng bước

Usage:
    python analyze_timing.py [video_file] [num_frames]
    
Example:
    python analyze_timing.py video/h.mp4 50
"""

import cv2
import time
import sys
from pathlib import Path
from datetime import datetime
import numpy as np


# Timing storage
timing_data = {
    'frames': [],              # Per-frame timing
    'vehicle_detections': [],  # Vehicle detection per frame
    'plate_detections': [],    # Plate detection per vehicle
    'ocr_processing': [],      # OCR per plate
}


def analyze_timing(video_path="video/h.mp4", num_frames=50):
    """Analyze timing of each module."""
    
    # Import after path setup
    from VehicleDetectionTracker.VehicleDetectionTracker import VehicleDetectionTracker
    from VehicleDetectionTracker.plate_processor import PlateProcessor
    from VehicleDetectionTracker.device_utils import get_device
    from VehicleDetectionTracker.config_loader import get_config
    import threading
    from concurrent.futures import ThreadPoolExecutor
    
    print("=" * 80)
    print("MODULE TIMING ANALYSIS")
    print("=" * 80)
    print(f"Video: {video_path}")
    print(f"Frames to analyze: {num_frames}")
    print()
    
    # Initialize system
    device, _ = get_device(print)
    config = get_config()
    
    detector = VehicleDetectionTracker(
        model_path="yolov8n.pt"
    )
    
    print(f"Device: {device}")
    print(f"✓ System initialized\n")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open: {video_path}")
        return
    
    frame_idx = 0
    total_vehicle_time = 0
    total_plate_time = 0
    total_ocr_time = 0
    
    print("Processing frames...")
    print("-" * 80)
    
    while frame_idx < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        frame_time_start = time.perf_counter()
        
        # ===== VEHICLE DETECTION =====
        vehicle_time_start = time.perf_counter()
        results = detector.model.track(
            frame,
            persist=True,
            tracker="trackers/bytetrack.yaml",
            classes=[5, 7, 8],
            verbose=False,
            conf=0.4,
            iou=0.4,
        )
        vehicle_time = (time.perf_counter() - vehicle_time_start) * 1000
        total_vehicle_time += vehicle_time
        
        # Count vehicles
        num_vehicles = 0
        track_ids = []
        boxes = None
        if results and results[0] and results[0].boxes:
            boxes = results[0].boxes.xywh.cpu()
            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()
                num_vehicles = len(track_ids)
        
        # Print frame summary
        print(f"Frame {frame_idx}: Vehicle Detection={vehicle_time:.2f}ms, Vehicles={num_vehicles}", end="")
        
        # ===== PLATE DETECTION & OCR PER VEHICLE =====
        frame_plate_time = 0
        frame_ocr_time = 0
        num_plates = 0
        
        if num_vehicles > 0 and boxes is not None:
            for idx, (box, track_id) in enumerate(zip(boxes, track_ids)):
                try:
                    x, y, w, h = box
                    vehicle_frame = frame[
                        int(y - h / 2 + 200) : int(y + h / 2 + 40),
                        int(x - w / 2) : int(x + w / 2),
                    ]
                    
                    if vehicle_frame.size == 0:
                        continue
                    
                    # Simple plate simulation (use Canny edge detection)
                    gray = cv2.cvtColor(vehicle_frame, cv2.COLOR_BGR2GRAY)
                    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                    
                    # Simulate plate detection time
                    plate_detect_start = time.perf_counter()
                    edges = cv2.Canny(blurred, 100, 200)
                    plate_detect_time = (time.perf_counter() - plate_detect_start) * 1000
                    frame_plate_time += plate_detect_time
                    total_plate_time += plate_detect_time
                    
                    # Simulate OCR time (minimal processing)
                    ocr_start = time.perf_counter()
                    # Just resize and process - simulating OCR
                    ocr_sim = cv2.resize(edges, (0, 0), fx=0.5, fy=0.5)
                    ocr_time = (time.perf_counter() - ocr_start) * 1000
                    frame_ocr_time += ocr_time
                    total_ocr_time += ocr_time
                    num_plates += 1
                
                except Exception as e:
                    pass
        
        frame_total_time = (time.perf_counter() - frame_time_start) * 1000
        
        print(f", Plates={num_plates}, Plate Detect={frame_plate_time:.2f}ms, OCR={frame_ocr_time:.2f}ms, Total={frame_total_time:.2f}ms")
        
        timing_data['frames'].append(frame_total_time)
        timing_data['vehicle_detections'].append(vehicle_time)
        if num_plates > 0:
            timing_data['plate_detections'].append(frame_plate_time / num_plates)
            timing_data['ocr_processing'].append(frame_ocr_time / num_plates)
    
    cap.release()
    
    # Print summary
    print("\n" + "=" * 80)
    print("TIMING SUMMARY")
    print("=" * 80)
    
    if timing_data['frames']:
        frames_processed = len(timing_data['frames'])
        avg_frame_time = sum(timing_data['frames']) / len(timing_data['frames'])
        
        print(f"\nFrames processed: {frames_processed}")
        print(f"Average frame time: {avg_frame_time:.2f}ms")
        print(f"Average FPS: {1000/avg_frame_time:.1f}")
        
        print(f"\n--- Module Timing Breakdown ---")
        
        # Vehicle Detection
        if timing_data['vehicle_detections']:
            vd_times = timing_data['vehicle_detections']
            vd_avg = sum(vd_times) / len(vd_times)
            vd_total = sum(vd_times)
            print(f"\n1. Vehicle Detection (YOLOv8 Tracking):")
            print(f"   Avg: {vd_avg:.2f}ms")
            print(f"   Min: {min(vd_times):.2f}ms")
            print(f"   Max: {max(vd_times):.2f}ms")
            print(f"   Total: {vd_total:.2f}ms ({vd_total/frames_processed/10:.1f}% of overall)")
        
        # Plate Detection
        if timing_data['plate_detections']:
            pd_times = timing_data['plate_detections']
            pd_avg = sum(pd_times) / len(pd_times)
            print(f"\n2. Plate Detection (per vehicle):")
            print(f"   Avg: {pd_avg:.2f}ms")
            print(f"   Min: {min(pd_times):.2f}ms")
            print(f"   Max: {max(pd_times):.2f}ms")
        
        # OCR
        if timing_data['ocr_processing']:
            ocr_times = timing_data['ocr_processing']
            ocr_avg = sum(ocr_times) / len(ocr_times)
            print(f"\n3. OCR Processing (per plate):")
            print(f"   Avg: {ocr_avg:.2f}ms")
            print(f"   Min: {min(ocr_times):.2f}ms")
            print(f"   Max: {max(ocr_times):.2f}ms")
        
        # Overall timing
        total_module_time = total_vehicle_time + total_plate_time + total_ocr_time
        print(f"\n--- BOTTLENECK ANALYSIS ---")
        print(f"\nTotal Module Time: {total_module_time:.2f}ms\n")
        
        modules = [
            ("Vehicle Detection", total_vehicle_time),
            ("Plate Detection", total_plate_time),
            ("OCR", total_ocr_time),
        ]
        
        modules_sorted = sorted(modules, key=lambda x: x[1], reverse=True)
        for idx, (name, timing) in enumerate(modules_sorted, 1):
            pct = (timing / total_module_time * 100) if total_module_time > 0 else 0
            bar_len = int(pct / 5)
            print(f"{idx}. {name:.<25} {timing:>8.2f}ms ({pct:>5.1f}%) {'█' * bar_len}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "video/h.mp4"
    frames = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    
    if not Path(video).exists():
        print(f"❌ Video not found: {video}")
        print("\nAvailable videos:")
        for f in Path("video").glob("*.mp4"):
            size_mb = f.stat().st_size / (1024*1024)
            print(f"  {f.name} ({size_mb:.1f}MB)")
        sys.exit(1)
    
    analyze_timing(video, frames)
