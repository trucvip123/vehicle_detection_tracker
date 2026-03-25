#!/usr/bin/env python3
"""
Module Profiling Script - Đo lường thời gian xử lý của từng module
Identifies performance bottlenecks in the detection pipeline.

Usage:
    python profile_modules.py [video_file] [num_frames]
    
Example:
    python profile_modules.py video/h.mp4 100
"""

import cv2
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import os
import sys

# Import our modules
from VehicleDetectionTracker.frame_processor import FrameProcessor
from VehicleDetectionTracker.plate_processor import PlateProcessor
from VehicleDetectionTracker.device_utils import get_device
from VehicleDetectionTracker.config_loader import get_config
from VehicleDetectionTracker.plate_utils import (
    initialize_plate_detector,
    detect_license_plate_sync,
)
from VehicleDetectionTracker.utils.ocr import OCRReader
from concurrent.futures import ThreadPoolExecutor
import threading


class ModuleProfiler:
    """Profile execution time of each module in the pipeline."""
    
    def __init__(self, video_path, num_frames=100):
        """Initialize profiler with video source.
        
        Args:
            video_path: Path to video file or RTSP stream
            num_frames: Number of frames to process for profiling
        """
        self.video_path = video_path
        self.num_frames = num_frames
        
        # Timing results
        self.timings = {
            'vehicle_detection': [],      # YOLOv8 vehicle tracking
            'plate_detection': [],        # YOLOv8 plate detection per vehicle
            'plate_ocr': [],              # OCR processing per plate
        }
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'total_vehicles': 0,
            'total_plates': 0,
            'total_ocr_calls': 0,
        }
        
        self.log_func = self._log
        
        # Initialize components
        print("[PROFILER] Initializing detection components...")
        self._init_components()
        print("[PROFILER] ✓ Components initialized")
    
    def _log(self, message):
        """Simple logging function."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
    
    def _init_components(self):
        """Initialize all detection components."""
        try:
            # Device
            self.device, _ = get_device(self._log)
            self._log(f"Device: {self.device}")
            
            # Load configuration
            config = get_config()
            detection_config = config.get_detection_config()
            
            # Initialize frame processor
            self.frame_processor = FrameProcessor(
                model_path="yolov8n.pt",
                config=config,
                device=self.device,
                log_func=self._log
            )
            self._log("✓ FrameProcessor initialized")
            
            # Initialize plate processor (minimal setup for profiling)
            executor = ThreadPoolExecutor(max_workers=1)  # Single worker for profiling
            
            # Initialize plate detector
            self.plate_model = initialize_plate_detector()
            self._log("✓ Plate detector initialized")
            
            # Initialize OCR reader
            self.ocr_reader = OCRReader(
                model_type="paddleocr",
                use_gpu=(self.device == "cuda"),
                lang=['en'],
                log_func=self._log
            )
            self._log("✓ OCR reader initialized")
            
            # Create minimal plate processor
            self.plate_processor = PlateProcessor(
                plate_model=self.plate_model,
                ocr_reader=self.ocr_reader,
                executor=executor,
                log_func=self._log
            )
            
            self.model_lock = threading.Lock()
        
        except Exception as e:
            self._log(f"❌ Error initializing components: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def profile_vehicle_detection(self, frame):
        """Profile vehicle detection on a frame.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            tuple: (elapsed_ms, results)
        """
        start = time.perf_counter()
        
        try:
            # Run vehicle detection
            results = self.frame_processor.model.track(
                frame,
                persist=True,
                tracker="trackers/bytetrack.yaml",
                classes=[5, 7, 8],  # bus, truck, boat
                verbose=False,
                conf=0.4,
                iou=0.4,
            )
            
            elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
            return elapsed, results
        
        except Exception as e:
            self._log(f"❌ Error in vehicle detection: {e}")
            return 0, None
    
    def profile_plate_detection(self, vehicle_frame):
        """Profile plate detection on a vehicle frame.
        
        Args:
            vehicle_frame: Cropped vehicle frame
            
        Returns:
            float: elapsed_ms
        """
        start = time.perf_counter()
        
        try:
            from VehicleDetectionTracker.plate_utils import _sync_plate_inference
            
            with self.model_lock:
                results = _sync_plate_inference(
                    self.frame_processor.model if hasattr(self, 'plate_model') and not self.plate_model else None,
                    vehicle_frame,
                    self.model_lock,
                    size=640
                )
            
            elapsed = (time.perf_counter() - start) * 1000
            return elapsed
        
        except Exception as e:
            self._log(f"⚠ Error in plate detection: {e}")
            return 0
    
    def profile_ocr(self, plate_image):
        """Profile OCR processing on plate image.
        
        Args:
            plate_image: Cropped plate image
            
        Returns:
            tuple: (elapsed_ms, text)
        """
        if plate_image.size == 0:
            return 0, None
        
        start = time.perf_counter()
        
        try:
            with self.model_lock:
                text = self.ocr_reader.read_license_plate(plate_image)
            
            elapsed = (time.perf_counter() - start) * 1000
            return elapsed, text
        
        except Exception as e:
            self._log(f"⚠ Error in OCR: {e}")
            return 0, None
    
    def run_profiling(self):
        """Run profiling on video stream."""
        self._log(f"Starting profiling: video={self.video_path}, frames={self.num_frames}")
        print("=" * 80)
        
        try:
            # Open video
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self._log(f"❌ Cannot open video: {self.video_path}")
                return
            
            frame_count = 0
            
            while frame_count < self.num_frames:
                ret, frame = cap.read()
                if not ret:
                    self._log(f"⚠ Reached end of video at frame {frame_count}")
                    break
                
                frame_count += 1
                self._log(f"\n--- Frame {frame_count}/{self.num_frames} ---")
                
                # ===== VEHICLE DETECTION PROFILING =====
                vehicle_time, results = self.profile_vehicle_detection(frame)
                self.timings['vehicle_detection'].append(vehicle_time)
                self._log(f"  Vehicle Detection: {vehicle_time:.2f}ms")
                
                if results is None or results[0] is None or results[0].boxes is None:
                    self._log(f"  ⚠ No vehicles detected")
                    self.stats['total_frames'] += 1
                    continue
                
                # Extract vehicle information
                boxes = results[0].boxes.xywh.cpu() if results[0].boxes else []
                track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []
                num_vehicles = len(track_ids)
                self.stats['total_frames'] += 1
                self.stats['total_vehicles'] += num_vehicles
                self._log(f"  Detected {num_vehicles} vehicles")
                
                # ===== PLATE DETECTION PROFILING PER VEHICLE =====
                frame_plate_times = []
                for idx, (box, track_id) in enumerate(zip(boxes, track_ids)):
                    try:
                        x, y, w, h = box
                        vehicle_frame = frame[
                            int(y - h / 2 + 200) : int(y + h / 2 + 40),
                            int(x - w / 2) : int(x + w / 2),
                        ]
                        
                        if vehicle_frame.size == 0:
                            continue
                        
                        # Profile plate detection
                        plate_time = self.profile_plate_detection(vehicle_frame)
                        frame_plate_times.append(plate_time)
                        self.timings['plate_detection'].append(plate_time)
                        self.stats['total_plates'] += 1
                        
                        # Simple plate detection for OCR profiling
                        # (Simulate with gaussian blur detection)
                        gray = cv2.cvtColor(vehicle_frame, cv2.COLOR_BGR2GRAY)
                        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                        edges = cv2.Canny(blurred, 100, 200)
                        
                        # Find contours
                        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            # Use largest contour as plate
                            cnt = max(contours, key=cv2.contourArea)
                            x_plate, y_plate, w_plate, h_plate = cv2.boundingRect(cnt)
                            
                            if w_plate > 30 and h_plate > 15:
                                plate_crop = vehicle_frame[y_plate:y_plate+h_plate, x_plate:x_plate+w_plate]
                                
                                # ===== OCR PROFILING =====
                                ocr_time, text = self.profile_ocr(plate_crop)
                                self.timings['plate_ocr'].append(ocr_time)
                                self.stats['total_ocr_calls'] += 1
                                self._log(f"    Vehicle {track_id} - Plate OCR: {ocr_time:.2f}ms, Text: '{text}'")
                    
                    except Exception as e:
                        self._log(f"    ⚠ Error processing vehicle {idx}: {e}")
                        continue
                
                if frame_plate_times:
                    avg_plate_time = sum(frame_plate_times) / len(frame_plate_times)
                    max_plate_time = max(frame_plate_times)
                    self._log(f"  Plate Detection - Avg: {avg_plate_time:.2f}ms, Max: {max_plate_time:.2f}ms")
            
            cap.release()
            self._log(f"\n✓ Profiling complete ({frame_count} frames processed)")
            self._print_summary()
        
        except Exception as e:
            self._log(f"❌ Error during profiling: {e}")
            import traceback
            traceback.print_exc()
    
    def _print_summary(self):
        """Print profiling summary."""
        print("\n" + "=" * 80)
        print("PROFILING SUMMARY")
        print("=" * 80)
        
        # Statistics
        print(f"\nStatistics:")
        print(f"  Total frames: {self.stats['total_frames']}")
        print(f"  Total vehicles: {self.stats['total_vehicles']}")
        print(f"  Total plates: {self.stats['total_plates']}")
        print(f"  Total OCR calls: {self.stats['total_ocr_calls']}")
        
        if self.stats['total_frames'] > 0:
            print(f"  Avg vehicles/frame: {self.stats['total_vehicles'] / self.stats['total_frames']:.2f}")
        
        # Timing analysis
        print(f"\n--- Module Timing ---")
        
        # Vehicle Detection
        if self.timings['vehicle_detection']:
            vd_times = self.timings['vehicle_detection']
            print(f"\n1. VEHICLE DETECTION (YOLOv8 Tracking):")
            print(f"   Samples: {len(vd_times)}")
            print(f"   Min: {min(vd_times):.2f}ms")
            print(f"   Max: {max(vd_times):.2f}ms")
            print(f"   Avg: {sum(vd_times)/len(vd_times):.2f}ms")
            print(f"   Total: {sum(vd_times):.2f}ms")
        
        # Plate Detection
        if self.timings['plate_detection']:
            pd_times = self.timings['plate_detection']
            print(f"\n2. PLATE DETECTION (YOLOv8 Plate):")
            print(f"   Samples: {len(pd_times)}")
            print(f"   Min: {min(pd_times):.2f}ms")
            print(f"   Max: {max(pd_times):.2f}ms")
            print(f"   Avg: {sum(pd_times)/len(pd_times):.2f}ms")
            print(f"   Total: {sum(pd_times):.2f}ms")
        
        # OCR
        if self.timings['plate_ocr']:
            ocr_times = self.timings['plate_ocr']
            print(f"\n3. OCR (PaddleOCR):")
            print(f"   Samples: {len(ocr_times)}")
            print(f"   Min: {min(ocr_times):.2f}ms")
            print(f"   Max: {max(ocr_times):.2f}ms")
            print(f"   Avg: {sum(ocr_times)/len(ocr_times):.2f}ms")
            print(f"   Total: {sum(ocr_times):.2f}ms")
        
        # Bottleneck Analysis
        print(f"\n--- BOTTLENECK ANALYSIS ---")
        
        total_times = {}
        if self.timings['vehicle_detection']:
            total_times['Vehicle Detection'] = sum(self.timings['vehicle_detection'])
        if self.timings['plate_detection']:
            total_times['Plate Detection'] = sum(self.timings['plate_detection'])
        if self.timings['plate_ocr']:
            total_times['OCR'] = sum(self.timings['plate_ocr'])
        
        if total_times:
            total = sum(total_times.values())
            print(f"\nTotal Processing Time: {total:.2f}ms\n")
            
            # Sort by time
            sorted_modules = sorted(total_times.items(), key=lambda x: x[1], reverse=True)
            for idx, (module, module_time) in enumerate(sorted_modules, 1):
                percentage = (module_time / total) * 100 if total > 0 else 0
                bar_length = int(percentage / 5)
                bar = "█" * bar_length
                print(f"{idx}. {module:.<30} {module_time:>10.2f}ms ({percentage:>5.1f}%) {bar}")
        
        print("\n" + "=" * 80)


def main():
    """Main entry point."""
    # Parse arguments
    video_path = sys.argv[1] if len(sys.argv) > 1 else "video/h.mp4"
    num_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    
    print(f"Module Profiler")
    print(f"Video: {video_path}")
    print(f"Frames to process: {num_frames}")
    print()
    
    # Check if video exists
    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        print(f"Available videos:")
        video_dir = Path("video")
        if video_dir.exists():
            for vf in video_dir.glob("*.mp4"):
                print(f"  - {vf}")
        return
    
    # Run profiler
    profiler = ModuleProfiler(video_path, num_frames)
    profiler.run_profiling()


if __name__ == "__main__":
    main()
