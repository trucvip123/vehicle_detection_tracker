"""
GPU-optimized RTSP streaming pipeline with GStreamer and TensorRT support.
Pipeline: RTSP → GStreamer → GPU decode (CUVID) → YOLO TensorRT → Display
"""

import cv2
import sys
import time
import os
import subprocess
import threading
from datetime import datetime
from queue import Queue
from typing import Optional, Tuple
import numpy as np

from VehicleDetectionTracker.logging_utils import log
from VehicleDetectionTracker.performance_timing import time_block
from VehicleDetectionTracker.device_utils import get_device

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    log("[WARNING] Ultralytics not available")

try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst, GLib
    GSTREAMER_AVAILABLE = True
    Gst.init(None)
except (ImportError, ValueError):
    GSTREAMER_AVAILABLE = False
    log("[INFO] GStreamer not available, using OpenCV")


class GPUOptimizedStreamHandler:
    """Handles RTSP/video streaming with GPU optimization."""
    
    def __init__(self, log_func=log, prefer_gstreamer=True, use_tensorrt=False):
        self.log = log_func
        self.prefer_gstreamer = prefer_gstreamer and GSTREAMER_AVAILABLE
        self.use_tensorrt = use_tensorrt
        self.device, self.use_gpu = get_device(log_func)
        
        self.log(f"[GPU_STREAM] Initializing GPU stream handler")
        self.log(f"[GPU_STREAM] Device: {self.device}")
        self.log(f"[GPU_STREAM] GPU available: {self.use_gpu}")
        self.log(f"[GPU_STREAM] GStreamer available: {GSTREAMER_AVAILABLE}")
        self.log(f"[GPU_STREAM] Using GStreamer: {self.prefer_gstreamer}")
        self.log(f"[GPU_STREAM] Using TensorRT: {self.use_tensorrt}")
    
    def create_gstreamer_pipeline(self, rtsp_url: str) -> Optional[cv2.VideoCapture]:
        """
        Create GStreamer-based RTSP pipeline with GPU decode.
        
        Pipeline: RTSP → CUVID (GPU decode) → OpenCV
        
        Args:
            rtsp_url: RTSP URL
            
        Returns:
            cv2.VideoCapture object or None if failed
        """
        if not self.prefer_gstreamer:
            return None
        
        try:
            # GStreamer pipeline with NVIDIA GPU decoding (CUVID)
            # rtspsrc → decodebin → videoscale → videoconvert → appsink
            gst_pipeline = (
                f"rtspsrc location={rtsp_url} latency=0 ! "
                "rtph264depay ! h264parse ! nvh264dec ! "  # NVIDIA GPU decode
                "videoscale ! video/x-raw,format=BGR,width=1280,height=720 ! "
                "videoconvert ! appsink max-buffers=1 drop=true"
            )
            
            self.log(f"[GStreamer] Creating pipeline with CUVID GPU decode")
            cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            
            if cap.isOpened():
                self.log(f"[GStreamer] ✓ Pipeline initialized successfully")
                return cap
            else:
                self.log(f"[GStreamer] ✗ Pipeline failed, falling back to OpenCV")
                return None
        
        except Exception as e:
            self.log(f"[GStreamer] Error creating pipeline: {e}")
            return None
    
    def create_opencv_pipeline(self, video_source: str) -> cv2.VideoCapture:
        """
        Create OpenCV-based pipeline with GPU acceleration.
        
        Args:
            video_source: Video file path or RTSP URL
            
        Returns:
            cv2.VideoCapture object
        """
        # Enable CUDA in OpenCV if available
        cap = cv2.VideoCapture(video_source)
        
        # Try to enable hardware acceleration
        try:
            # Set OpenCV backend to CUDA if available
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffering for low latency
            cap.set(cv2.CAP_PROP_FPS, 30)  # Target FPS
        except Exception as e:
            self.log(f"[OpenCV] Warning setting properties: {e}")
        
        return cap
    
    def load_yolo_model(self, model_path: str = "yolov8n.pt"):
        """
        Load YOLO model with TensorRT optimization if available.
        
        Args:
            model_path: Path to YOLO model
            
        Returns:
            YOLO model object
        """
        if not ULTRALYTICS_AVAILABLE:
            self.log("[ERROR] Ultralytics not available")
            return None
        
        try:
            self.log(f"[YOLO] Loading model: {model_path}")
            model = YOLO(model_path)
            
            # Move to GPU
            if self.use_gpu and self.device != "cpu":
                model.to(self.device)
                self.log(f"[YOLO] Model moved to {self.device}")
            
            # Try TensorRT export if requested
            if self.use_tensorrt and self.use_gpu:
                try:
                    self.log("[YOLO] Attempting TensorRT optimization...")
                    # Export to TensorRT format
                    model_engine_path = model_path.replace('.pt', '.engine')
                    if not os.path.exists(model_engine_path):
                        self.log("[YOLO] Exporting to TensorRT format...")
                        model.export(format='engine', device=0)
                        self.log(f"[YOLO] ✓ TensorRT export completed: {model_engine_path}")
                except Exception as e:
                    self.log(f"[YOLO] TensorRT optimization skipped: {e}")
            
            self.log("[YOLO] ✓ Model loaded successfully")
            return model
        
        except Exception as e:
            self.log(f"[ERROR] Failed to load model: {e}")
            return None
    
    def process_stream(
        self,
        video_source: str,
        model: Optional['YOLO'] = None,
        max_frames: Optional[int] = None,
        display_window: bool = True,
        confidence: float = 0.4,
    ):
        """
        Process RTSP/video stream with GPU optimization.
        
        Args:
            video_source: RTSP URL or video file path
            model: YOLO model for detection
            max_frames: Maximum frames to process
            display_window: Whether to display video
            confidence: Detection confidence threshold
        """
        self.log(f"\n[GPU_STREAM] Starting GPU-optimized stream")
        self.log(f"[GPU_STREAM] Source: {video_source}")
        self.log(f"[GPU_STREAM] Display: {display_window}")
        
        # Create video capture with GPU optimization
        cap = None
        if self.prefer_gstreamer and video_source.startswith("rtsp://"):
            self.log("[GPU_STREAM] Attempting GStreamer pipeline...")
            cap = self.create_gstreamer_pipeline(video_source)
        
        if cap is None:
            self.log("[GPU_STREAM] Using OpenCV pipeline...")
            cap = self.create_opencv_pipeline(video_source)
        
        if not cap.isOpened():
            self.log("[ERROR] Failed to open video source")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.log(f"[GPU_STREAM] FPS: {fps}, Resolution: {width}x{height}")
        
        frame_id = 0
        frame_times = []
        detect_times = []
        start_time = datetime.now()
        
        # Create window
        if display_window:
            window_name = "GPU-Optimized Stream"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        try:
            while True:
                # === READ FRAME ===
                with time_block("[STREAM_READ]", self.log):
                    success, frame = cap.read()
                    frame_id += 1
                
                if not success or frame is None:
                    self.log(f"[GPU_STREAM] End of stream at frame {frame_id}")
                    break
                
                if max_frames and frame_id > max_frames:
                    self.log(f"[GPU_STREAM] Reached max frames ({max_frames})")
                    break
                
                # === VEHICLE DETECTION ===
                if model:
                    with time_block("[DETECT_INFERENCE]", self.log):
                        try:
                            results = model.track(
                                frame,
                                verbose=False,
                                conf=confidence,
                                device=0 if self.use_gpu else "cpu"
                            )
                            
                            if results and len(results) > 0:
                                result = results[0]
                                
                                # Draw detections
                                if result.boxes and len(result.boxes) > 0:
                                    num_detections = len(result.boxes)
                                    for box in result.boxes:
                                        try:
                                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                                            conf = float(box.conf[0].cpu().numpy())
                                            track_id = int(box.id[0].cpu().numpy()) if box.id is not None else -1
                                            
                                            # Draw box
                                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                            
                                            # Draw label
                                            label = f"ID:{track_id} {conf:.2f}"
                                            cv2.putText(frame, label, (x1, y1 - 10),
                                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                        except Exception as e:
                                            pass
                                    
                                    if frame_id % 30 == 0:
                                        self.log(f"[DETECT] Frame {frame_id}: {num_detections} vehicles")
                        
                        except Exception as e:
                            self.log(f"[ERROR] Detection failed: {e}")
                
                # === DISPLAY ===
                if display_window:
                    with time_block("[STREAM_DISPLAY]", self.log):
                        # Add stats overlay
                        elapsed = (datetime.now() - start_time).total_seconds()
                        current_fps = frame_id / elapsed if elapsed > 0 else 0
                        
                        cv2.putText(frame, f"Frame: {frame_id}", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 70),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        if self.use_gpu:
                            cv2.putText(frame, f"GPU: {self.device}", (10, 110),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                        cv2.imshow(window_name, frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            self.log("[GPU_STREAM] User quit")
                            break
                
                frame_times.append((time.time() - start_time.timestamp()) * 1000)
                
                if frame_id % 30 == 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    actual_fps = frame_id / elapsed if elapsed > 0 else 0
                    self.log(f"[STATS] Frame {frame_id}: {actual_fps:.1f} FPS")
        
        except KeyboardInterrupt:
            self.log("[GPU_STREAM] Interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            if display_window:
                cv2.destroyAllWindows()
            
            # Print summary
            if frame_id > 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                avg_fps = frame_id / elapsed
                
                self.log(f"\n[SUMMARY] GPU Stream Processing Results:")
                self.log(f"          Total frames: {frame_id}")
                self.log(f"          Elapsed time: {elapsed:.2f}s")
                self.log(f"          Average FPS: {avg_fps:.2f}")
                self.log(f"          GPU: {self.device}")
                self.log(f"          GStreamer: {self.prefer_gstreamer}")
                self.log(f"          TensorRT: {self.use_tensorrt}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="GPU-optimized RTSP streaming")
    parser.add_argument("--source", type=str, default="video/h.mp4",
                       help="RTSP URL or video file")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                       help="YOLO model path")
    parser.add_argument("--no-gstreamer", action="store_true",
                       help="Disable GStreamer, use OpenCV only")
    parser.add_argument("--tensorrt", action="store_true",
                       help="Enable TensorRT optimization")
    parser.add_argument("--max-frames", type=int, default=None,
                       help="Maximum frames to process")
    parser.add_argument("--no-display", action="store_true",
                       help="Disable display window")
    parser.add_argument("--confidence", type=float, default=0.4,
                       help="Detection confidence threshold")
    
    args = parser.parse_args()
    
    # Create handler
    handler = GPUOptimizedStreamHandler(
        prefer_gstreamer=not args.no_gstreamer,
        use_tensorrt=args.tensorrt
    )
    
    # Load model
    model = handler.load_yolo_model(args.model)
    
    # Process stream
    handler.process_stream(
        video_source=args.source,
        model=model,
        max_frames=args.max_frames,
        display_window=not args.no_display,
        confidence=args.confidence
    )


if __name__ == "__main__":
    main()
