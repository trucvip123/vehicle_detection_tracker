"""
GPU-Optimized Pipeline (Pure OpenCV + YOLO on GPU)
No GStreamer Python bindings required - works with system GStreamer CLI or pure OpenCV

This is a working alternative to stream_gpu_optimized.py that avoids conda/PyGObject issues
"""

import cv2
import os
from pathlib import Path
from ultralytics import YOLO
import numpy as np
import time
from datetime import datetime
import torch

print(f"[INIT] PyTorch CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[INIT] GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"[INIT] CUDA Version: {torch.version.cuda}")
    torch.cuda.empty_cache()  # Clear cache


class GPUPipelineSimple:
    """Simplified GPU pipeline without GStreamer Python bindings dependency"""
    
    def __init__(self, model_path="yolov8n.pt", confidence=0.5, use_gpu=True, inference_resolution=None):
        """
        Initialize GPU pipeline
        
        Args:
            model_path: Path to YOLO model
            confidence: Confidence threshold
            use_gpu: Whether to use GPU for inference
            inference_resolution: Tuple (width, height) for inference. None = use full resolution
        """
        self.confidence = confidence
        self.use_gpu = use_gpu
        self.inference_resolution = inference_resolution  # e.g., (1280, 720) or (640, 360)
        
        # Load YOLO model with GPU support
        print(f"[INIT] Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        
        # Force GPU device
        if use_gpu and torch.cuda.is_available():
            print(f"[INIT] Moving model to GPU...")
            device = 0  # CUDA device 0
            # Run inference on GPU device
            _ = self.model.to(device)
            self.device = device
            print(f"[INIT] ✓ Model on GPU (CUDA:{device})")
        else:
            print(f"[INIT] Using CPU (GPU not available or use_gpu=False)")
            self.device = 'cpu'
    
    def process_frame_gpu(self, frame, verbose=False):
        """
        Process single frame on GPU with YOLO
        
        Args:
            frame: Input frame (numpy array BGR)
            verbose: Print timing details
            
        Returns:
            Annotated frame, detections, inference_time_ms
        """
        # Resize frame for inference if specified
        inference_frame = frame
        scale_x, scale_y = 1.0, 1.0
        
        if self.inference_resolution:
            orig_height, orig_width = frame.shape[:2]
            target_width, target_height = self.inference_resolution
            inference_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            scale_x = orig_width / target_width
            scale_y = orig_height / target_height
        
        start = time.time()
        
        # YOLO inference on GPU (specify device=0 for CUDA)
        results = self.model(inference_frame, conf=self.confidence, verbose=False, device=self.device)
        
        inference_time = (time.time() - start) * 1000
        
        # Get detections and scale back to original resolution if needed
        detections = []
        if len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Scale back to original resolution if inference was downsampled
                if self.inference_resolution:
                    x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                    y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                else:
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf,
                    'class': cls,
                    'class_name': results[0].names[cls]
                })
        
        # Annotate frame
        annotated = results[0].plot()
        
        if verbose:
            print(f"[YOLO] Inference: {inference_time:.2f}ms | Detections: {len(detections)}")
        
        return annotated, detections, inference_time
    
    def process_stream_file(self, video_path, max_frames=None, display=True, inference_resolution=None):
        """
        Process video file on GPU
        
        Args:
            video_path: Path to video file or RTSP URL
            max_frames: Maximum frames to process (None = all)
            display: Show video with detections
            inference_resolution: Override resolution for this stream. Tuple (width, height) or None
        """
        # Use provided resolution or class default
        if inference_resolution:
            original_inference_res = self.inference_resolution
            self.inference_resolution = inference_resolution
        print(f"\n[STREAM] Opening: {video_path}")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video source: {video_path}")
            return None
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[VIDEO] Resolution: {width}x{height} | FPS: {fps:.1f} | Total frames: {total_frames}")
        
        # Statistics
        frame_count = 0
        total_inference_time = 0
        detection_count = 0
        read_times = []
        display_times = []
        
        start_time = time.time()
        
        while True:
            # Read frame
            read_start = time.time()
            ret, frame = cap.read()
            read_time = (time.time() - read_start) * 1000
            
            if not ret:
                break
            
            frame_count += 1
            read_times.append(read_time)
            
            # Process on GPU
            annotated, detections, inference_time = self.process_frame_gpu(frame)
            total_inference_time += inference_time
            detection_count += len(detections)
            
            # Display
            if display:
                display_start = time.time()
                
                # Add FPS counter
                current_fps = frame_count / max(time.time() - start_time, 0.001)
                cv2.putText(annotated, f"FPS: {current_fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated, f"Inf: {inference_time:.1f}ms", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated, f"Detections: {len(detections)}", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow("GPU Pipeline", annotated)
                
                display_time = (time.time() - display_start) * 1000
                display_times.append(display_time)
                
                # Exit on 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Restore original resolution if override was used
        if inference_resolution:
            self.inference_resolution = original_inference_res
        
        # Print statistics
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed
        avg_inference = total_inference_time / max(frame_count, 1)
        avg_read = np.mean(read_times)
        avg_display = np.mean(display_times) if display_times else 0
        
        print(f"\n{'='*70}")
        print(f"[RESULTS] GPU Pipeline Performance")
        print(f"{'='*70}")
        if self.inference_resolution:
            print(f"Inference Resolution: {self.inference_resolution[0]}x{self.inference_resolution[1]} (downscaled)")
        print(f"Frames Processed: {frame_count}")
        print(f"Total Time: {elapsed:.2f}s")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Average Inference: {avg_inference:.2f}ms (GPU YOLO)")
        print(f"Average Frame Read: {avg_read:.2f}ms")
        print(f"Average Display: {avg_display:.2f}ms")
        print(f"Total Detections: {detection_count}")
        print(f"{'='*70}\n")
        
        return {
            'frames': frame_count,
            'time': elapsed,
            'fps': avg_fps,
            'inference_time_ms': avg_inference,
            'total_inference_time': total_inference_time
        }


def test_gpu_pipeline_local():
    """Test GPU pipeline with local video"""
    
    print(f"\n{'='*70}")
    print("GPU Pipeline Test - Local Video (Pure OpenCV + YOLO on GPU)")
    print(f"{'='*70}\n")
    
    # Try different video sources
    test_sources = [
        "video/h.mp4",  # Local video
        "test.mp4",      # Alternative local
    ]
    
    video_source = None
    for source in test_sources:
        if os.path.exists(source):
            video_source = source
            break
    
    if not video_source:
        print("[INFO] No local video found. Using live camera test (5 frames):")
        video_source = 0  # Webcam
    
    print(f"[SOURCE] Using: {video_source}")
    
    # Initialize pipeline
    try:
        print(f"[TORCH] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
             print(f"[TORCH] CUDA device: {torch.cuda.get_device_name(0)}")
    except:
        pass
    
    pipeline = GPUPipelineSimple(
        model_path="yolov8n.pt",
        confidence=0.5,
        use_gpu=True
    )
    
    # Process stream
    max_frames = 50 if isinstance(video_source, str) else 5
    stats = pipeline.process_stream_file(
        video_source,
        max_frames=max_frames,
        display=True
    )
    
    return stats


def test_gpu_pipeline_rtsp():
    """Test GPU pipeline with RTSP stream - With resolution scaling"""
    
    print(f"\n{'='*70}")
    print("GPU Pipeline Test - RTSP Stream (Pure OpenCV + YOLO on GPU)")
    print(f"{'='*70}\n")
    
    # RTSP Configuration
    RTSP_URL = "rtsp://aicamera.serveminecraft.net:554/"
    RTSP_USER = "admin"
    RTSP_PASS = "MOVYKV"
    
    # Build RTSP URI with credentials
    rtsp_uri = f"rtsp://{RTSP_USER}:{RTSP_PASS}@aicamera.serveminecraft.net:554/"
    
    # Inference resolution (for faster processing)
    # Options: (1280, 720), (960, 540), (640, 360)
    INFERENCE_RESOLUTION = (1280, 720)  # ← Reduced from 2880x1620 for faster inference
    
    print(f"[RTSP CONFIG]")
    print(f"  URL: {RTSP_URL}")
    print(f"  User: {RTSP_USER}")
    print(f"  Auth: Enabled")
    print(f"  Capture Resolution: 2880x1620 (stream)")
    print(f"  Inference Resolution: {INFERENCE_RESOLUTION[0]}x{INFERENCE_RESOLUTION[1]} (downscaled for speed)")
    print(f"[RTSP URI] rtsp://{RTSP_USER}:****@aicamera.serveminecraft.net:554/")
    
    # Print GPU info
    try:
        print(f"[TORCH] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[TORCH] CUDA device: {torch.cuda.get_device_name(0)}")
    except:
        pass
    
    # Initialize pipeline without resolution (will use per-stream override)
    pipeline = GPUPipelineSimple(
        model_path="yolov8n.pt",
        confidence=0.5,
        use_gpu=True
    )
    
    # Process RTSP stream with resolution scaling
    print(f"\n[CONNECTING] Attempting RTSP connection...")
    print(f"[INFO] Frames will be downscaled to {INFERENCE_RESOLUTION[0]}x{INFERENCE_RESOLUTION[1]} for inference")
    stats = pipeline.process_stream_file(
        rtsp_uri,
        max_frames=100,  # Process 100 frames from RTSP
        display=True,
        inference_resolution=INFERENCE_RESOLUTION  # ← Pass resolution here
    )
    
    return stats


def test_gpu_pipeline(mode='local'):
    """Test GPU pipeline with local video or RTSP stream
    
    Args:
        mode: 'local' (default) or 'rtsp'
    """
    
    if mode.lower() == 'rtsp':
        return test_gpu_pipeline_rtsp()
    else:
        return test_gpu_pipeline_local()


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    mode = 'local'  # Default to local
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    
    # Display menu
    print(f"\n{'='*70}")
    print("GPU Pipeline - Test Mode Selection")
    print(f"{'='*70}")
    print("Usage: python gpu_pipeline_working.py [mode]")
    print("\nAvailable modes:")
    print("  local    - Test with local video file (default)")
    print("  rtsp     - Test with RTSP camera stream")
    print(f"\nSelected mode: {mode.upper()}")
    print(f"{'='*70}\n")
    
    # Test the GPU pipeline
    if mode == 'rtsp':
        print("[TEST] Starting RTSP stream test...\n")
        stats = test_gpu_pipeline_rtsp()
    else:
        print("[TEST] Starting local video test...\n")
        stats = test_gpu_pipeline_local()
    
    if stats:
        print("\n[SUCCESS] GPU Pipeline working!")
        print(f"Achieved {stats['fps']:.2f} FPS with {stats['inference_time_ms']:.2f}ms inference time")
        print(f"Total frames processed: {stats['frames']}")
        print(f"Total time: {stats['time']:.2f}s")
    else:
        print("\n[ERROR] GPU Pipeline test failed")
