#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PUBLIC PROFILING - Công cụ đo lường nhanh chóng

Chạy: python quick_profile.py [video] [frames]
Ví dụ: python quick_profile.py video/h.mp4 50
"""

import cv2
import time
import sys
from pathlib import Path
from datetime import datetime


def quick_profile(video_path="video/h.mp4", num_frames=50):
    """Profiling nhanh kiểu đơn giản."""
    
    print("=" * 80)
    print("🔍 MODULE PROFILING - QUICK START")
    print("=" * 80)
    print(f"📹 Video: {video_path}")
    print(f"📊 Frames: {num_frames}")
    print()
    
    try:
        # Import components
        print("⏳ Loading components...")
        from VehicleDetectionTracker.VehicleDetectionTracker import VehicleDetectionTracker
        from VehicleDetectionTracker.device_utils import get_device
        from VehicleDetectionTracker.config_loader import get_config
        
        device, _ = get_device(print)
        config = get_config()
        
        print(f"✓ Device: {device}")
        print("✓ Loading models...")
        
        detector = VehicleDetectionTracker(
            model_path="yolov8n.pt"
        )
        print("✓ Models loaded\n")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return
        
        # Timing data
        vehicle_times = []
        frame_times = []
        frame_count = 0
        
        print("⏱️ Processing frames...")
        print("-" * 80)
        
        while frame_count < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Measure frame processing
            frame_start = time.perf_counter()
            
            # Run detection
            try:
                detector.process_frame_streaming(frame, datetime.now())
            except:
                pass  # Ignore errors
            
            frame_time = (time.perf_counter() - frame_start) * 1000
            frame_times.append(frame_time)
            
            # Progress
            bar_length = 40
            progress = int((frame_count / num_frames) * bar_length)
            bar = "█" * progress + "░" * (bar_length - progress)
            fps = frame_count / sum(frame_times[:frame_count]) * 10  # Rough FPS
            print(f"\r[{bar}] {frame_count}/{num_frames} | {frame_time:.0f}ms/frame | ~{fps:.1f}fps", end="")
        
        cap.release()
        print("\n" + "-" * 80)
        
        if not frame_times:
            print("❌ No frames processed")
            return
        
        # Analysis
        print("\n📊 PROFILING RESULTS")
        print("=" * 80)
        
        avg_time = sum(frame_times) / len(frame_times)
        min_time = min(frame_times)
        max_time = max(frame_times)
        total_time = sum(frame_times)
        avg_fps = 1000 / avg_time if avg_time > 0 else 0
        
        print(f"\n✓ Processed {frame_count} frames")
        print(f"\nFrame Processing Time:")
        print(f"  🔹 Minimum:  {min_time:>7.2f}ms")
        print(f"  🔹 Average:  {avg_time:>7.2f}ms")
        print(f"  🔹 Maximum:  {max_time:>7.2f}ms")
        print(f"  🔹 Total:    {total_time:>7.2f}ms")
        print(f"  🔹 FPS:      {avg_fps:>7.1f}")
        
        # Estimation breakdown (based on typical ratios)
        # These are estimates - actual may vary
        vehicle_detection_pct = 40  # Typical: 30-50%
        plate_detection_pct = 20    # Typical: 15-25%
        ocr_pct = 40                # Typical: 30-50%
        
        vehicle_est = total_time * vehicle_detection_pct / 100
        plate_est = total_time * plate_detection_pct / 100
        ocr_est = total_time * ocr_pct / 100
        
        print(f"\n🎯 Estimated Module Breakdown (typical ratios):")
        print(f"  Assuming: {vehicle_detection_pct}% Vehicle + {plate_detection_pct}% Plate + {ocr_pct}% OCR")
        print()
        
        # Create bars
        modules = [
            ("Vehicle Detection", vehicle_est, vehicle_detection_pct),
            ("Plate Detection", plate_est, plate_detection_pct),
            ("OCR Processing", ocr_est, ocr_pct),
        ]
        
        for name, est_time, pct in sorted(modules, key=lambda x: x[2], reverse=True):
            bar_len = int(pct / 5)
            bar = "█" * bar_len
            print(f"  {name:.<25} {est_time:>8.0f}ms ({pct:>3}%) {bar}")
        
        print("\n" + "=" * 80)
        print("💡 NOTES:")
        print("  • This is ESTIMATED breakdown based on typical ratios")
        print("  • For ACTUAL measurements, run: python analyze_timing.py")
        print("  • OCR is typically the biggest bottleneck (30-60%)")
        print("  • Solution: Enable batch_inference (Approach 2)")
        print()
        print("📖 Documentation: see MODULE_PROFILING_GUIDE.md")
        print("=" * 80)
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("   Make sure all dependencies are installed")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "video/h.mp4"
    frames = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    
    # Check video exists
    if not Path(video).exists():
        print(f"❌ Video not found: {video}\n")
        print("📹 Available videos:")
        video_dir = Path("video")
        if video_dir.exists():
            for vf in sorted(video_dir.glob("*.mp4")):
                size_mb = vf.stat().st_size / (1024 * 1024)
                print(f"   python quick_profile.py {vf} [frames]  ({size_mb:.1f}MB)")
        sys.exit(1)
    
    quick_profile(video, frames)
