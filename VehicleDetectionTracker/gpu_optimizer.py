"""
GPU Performance Optimization Module for VehicleDetectionTracker
Integrates features from GPUPipelineSimple for improved inference performance
"""

import torch
import cv2
import numpy as np
import time
from typing import Tuple, Optional, List, Dict, Any


class GPUOptimizer:
    """Optimize GPU inference with resolution scaling and memory management"""
    
    def __init__(self, device: str = 'cuda:0', use_gpu: bool = True):
        """
        Initialize GPU optimizer
        
        Args:
            device: GPU device (e.g., 'cuda:0')
            use_gpu: Whether to use GPU
        """
        self.device = device
        self.use_gpu = use_gpu
        self.inference_resolution = None  # None = full resolution
        
        if use_gpu and torch.cuda.is_available():
            self._cleanup_gpu_cache()
    
    def _cleanup_gpu_cache(self):
        """Clear GPU cache for better memory management"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    
    def set_inference_resolution(self, width: int, height: int):
        """
        Set inference resolution for downscaling
        
        Performance impact:
        - (1280, 720): ~18 FPS on RTSP
        - (960, 540):  ~25 FPS on RTSP
        - (640, 360):  ~35 FPS on RTSP
        """
        self.inference_resolution = (width, height)
    
    def scale_frame_for_inference(self, frame: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Resize frame for faster inference while maintaining tracking quality
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Scaled frame, scale_x (orig_width/scaled_width), scale_y (orig_height/scaled_height)
        """
        if not self.inference_resolution:
            return frame, 1.0, 1.0
        
        orig_height, orig_width = frame.shape[:2]
        target_width, target_height = self.inference_resolution
        
        # Resize for faster inference
        scaled_frame = cv2.resize(
            frame, 
            (target_width, target_height),
            interpolation=cv2.INTER_LINEAR  # Fast linear interpolation
        )
        
        scale_x = orig_width / target_width
        scale_y = orig_height / target_height
        
        return scaled_frame, scale_x, scale_y
    
    def scale_detections_back(self, detections: List[Dict], scale_x: float, scale_y: float) -> List[Dict]:
        """
        Scale detection bounding boxes back to original frame resolution
        
        Args:
            detections: List of detection dicts with 'bbox' key
            scale_x: X-axis scale factor
            scale_y: Y-axis scale factor
            
        Returns:
            Detections with scaled bounding boxes
        """
        if scale_x == 1.0 and scale_y == 1.0:
            return detections
        
        scaled_detections = []
        for det in detections:
            scaled_det = det.copy()
            
            if 'bbox' in det:
                x1, y1, x2, y2 = det['bbox']
                # Scale back to original resolution
                scaled_det['bbox'] = (
                    int(x1 * scale_x),
                    int(y1 * scale_y),
                    int(x2 * scale_x),
                    int(y2 * scale_y)
                )
            
            scaled_detections.append(scaled_det)
        
        return scaled_detections
    
    def optimize_model_for_inference(self, model: Any):
        """
        Apply inference optimizations to YOLO model
        
        Args:
            model: YOLO model instance
        """
        if not self.use_gpu:
            return
        
        try:
            # Enable half precision (FP16) for faster inference on compatible GPUs
            # Note: Only works on GPUs with compute capability >= 5.3
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                if 'GTX' in gpu_name or 'RTX' in gpu_name:
                    # Most gaming GPUs support FP16
                    pass
        except Exception as e:
            print(f"[GPU] Could not apply optimizations: {e}")


class PerformanceMonitor:
    """Monitor and report performance metrics"""
    
    def __init__(self):
        """Initialize performance monitor"""
        self.frame_times = []
        self.inference_times = []
        self.read_times = []
        self.display_times = []
        self.detection_counts = []
        self.frame_count = 0
    
    def record_frame_processing(self, 
                               read_time_ms: float,
                               inference_time_ms: float,
                               display_time_ms: float,
                               detection_count: int):
        """Record processing times for a frame"""
        self.frame_count += 1
        self.read_times.append(read_time_ms)
        self.inference_times.append(inference_time_ms)
        self.display_times.append(display_time_ms)
        self.detection_counts.append(detection_count)
        self.frame_times.append(read_time_ms + inference_time_ms + display_time_ms)
    
    def get_statistics(self) -> Dict[str, float]:
        """Get performance statistics"""
        if not self.frame_count:
            return {}
        
        total_time = sum(self.frame_times) / 1000  # Convert to seconds
        fps = self.frame_count / total_time if total_time > 0 else 0
        
        return {
            'fps': fps,
            'avg_frame_time_ms': np.mean(self.frame_times),
            'avg_inference_ms': np.mean(self.inference_times),
            'avg_read_ms': np.mean(self.read_times),
            'avg_display_ms': np.mean(self.display_times),
            'avg_detections': np.mean(self.detection_counts),
            'total_frames': self.frame_count,
            'total_time_sec': total_time,
        }
    
    def print_summary(self):
        """Print performance summary"""
        stats = self.get_statistics()
        if not stats:
            print("[PERF] No statistics available")
            return
        
        print(f"\n{'='*70}")
        print("[PERFORMANCE SUMMARY]")
        print(f"{'='*70}")
        print(f"Frames Processed: {stats['total_frames']}")
        print(f"Total Time: {stats['total_time_sec']:.2f}s")
        print(f"Average FPS: {stats['fps']:.2f}")
        print(f"Average Inference: {stats['avg_inference_ms']:.2f}ms")
        print(f"Average Frame Read: {stats['avg_read_ms']:.2f}ms")
        print(f"Average Display: {stats['avg_display_ms']:.2f}ms")
        print(f"Average Detections: {stats['avg_detections']:.1f}")
        print(f"{'='*70}\n")


class OptimizedTracking:
    """Optimized tracking with GPU acceleration"""
    
    def __init__(self, gpu_optimizer: GPUOptimizer):
        """Initialize optimized tracking"""
        self.gpu_optimizer = gpu_optimizer
        self.performance_monitor = PerformanceMonitor()
    
    def process_frame_optimized(self, 
                               frame: np.ndarray,
                               model: Any,
                               detection_config: Dict,
                               tracking_config: Dict,
                               vehicle_classes: List[int]) -> Tuple[Any, List[Dict], float]:
        """
        Process frame with GPU optimizations
        
        Args:
            frame: Input frame
            model: YOLO model
            detection_config: Detection configuration
            tracking_config: Tracking configuration
            vehicle_classes: Vehicle class IDs to detect
            
        Returns:
            Tracking results, detections with scaled bboxes, inference_time_ms
        """
        # Scale frame for faster inference
        scaled_frame, scale_x, scale_y = self.gpu_optimizer.scale_frame_for_inference(frame)
        
        # Run tracking on scaled frame
        inference_start = time.time()
        results = model.track(
            scaled_frame,  # Use scaled frame for faster inference
            persist=True,
            tracker=tracking_config.get("tracker_type", "bytetrack.yaml"),
            classes=vehicle_classes,
            verbose=False,
            conf=detection_config.get("confidence", 0.5),
            iou=detection_config.get("iou", 0.4),
            imgsz=detection_config.get("image_size", 1280),
            device=self.gpu_optimizer.device if self.gpu_optimizer.use_gpu else 'cpu',
        )
        inference_time_ms = (time.time() - inference_start) * 1000
        
        # Extract detections and scale back to original frame size
        detections = []
        if len(results) > 0 and hasattr(results[0], 'boxes'):
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy() if self.gpu_optimizer.use_gpu else box.xyxy[0].numpy()
                
                # Scale back to original resolution
                x1, y1 = int(x1 * scale_x), int(y1 * scale_y)
                x2, y2 = int(x2 * scale_x), int(y2 * scale_y)
                
                conf = box.conf[0].item() if self.gpu_optimizer.use_gpu else box.conf[0]
                cls = int(box.cls[0].item()) if self.gpu_optimizer.use_gpu else int(box.cls[0])
                track_id = int(box.id[0].item()) if box.id is not None and self.gpu_optimizer.use_gpu else (int(box.id[0]) if box.id is not None else None)
                
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf,
                    'class': cls,
                    'track_id': track_id,
                })
        
        return results, detections, inference_time_ms
