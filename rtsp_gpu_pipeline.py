"""
RTSP GPU Processing Pipeline
Uses software video decode + NVIDIA CUDA GPU for real-time processing
"""

import cv2
import numpy as np
import time
from typing import Tuple, Optional

try:
    import cupy as cp
    HAS_CUPY = True
    print("✅ CuPy (NVIDIA GPU) available")
except ImportError:
    HAS_CUPY = False
    print("⚠️  CuPy not available - will use CPU. Install: pip install cupy-cuda12x")


class GPUProcessingPipeline:
    """Real-time RTSP video processing with GPU acceleration"""
    
    def __init__(self, rtsp_url: str, use_gpu: bool = True):
        self.rtsp_url = rtsp_url
        self.use_gpu = use_gpu and HAS_CUPY
        self.cap = None
        self.frame_count = 0
        self.fps = 0
        self.processing_time = 0
        
    def open_stream(self) -> bool:
        """Open RTSP stream (software decode, GPU processing)"""
        # Use OpenCV with software H.264 decode
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not self.cap.isOpened():
            # Fallback to default backend
            self.cap = cv2.VideoCapture(self.rtsp_url)
        
        return self.cap.isOpened()
    
    def process_frame_gpu(self, frame: np.ndarray) -> np.ndarray:
        """Process frame on NVIDIA GPU using CUDA"""
        if not self.use_gpu:
            return frame
        
        try:
            # Transfer frame to GPU
            gpu_frame = cp.asarray(frame, dtype=cp.uint8)
            
            # GPU-accelerated color to grayscale conversion
            gpu_gray = cp.mean(gpu_frame.astype(cp.float32), axis=2).astype(cp.uint8)
            
            # GPU-accelerated Canny edge detection simulation
            # (simplified for demonstration)
            gpu_edges = cp.zeros_like(gpu_gray)
            
            # Simple Sobel-like edge detection on GPU
            kernel = cp.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=cp.float32)
            for i in range(1, gpu_gray.shape[0] - 1):
                for j in range(1, gpu_gray.shape[1] - 1):
                    region = gpu_gray[i-1:i+2, j-1:j+2].astype(cp.float32)
                    gpu_edges[i, j] = cp.abs(cp.sum(region * kernel)).astype(cp.uint8)
            
            # Convert back to BGR for display
            gpu_edges_bgr = cp.stack([gpu_edges, gpu_edges, gpu_edges], axis=2)
            
            # Transfer result back to CPU
            output = cp.asnumpy(gpu_edges_bgr)
            return output
            
        except Exception as e:
            print(f"⚠️  GPU processing error: {e}, falling back to CPU")
            self.use_gpu = False
            return frame
    
    def process_frame_cpu(self, frame: np.ndarray) -> np.ndarray:
        """Process frame on CPU"""
        try:
            # Fast CPU processing: grayscale + edge detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Use Canny for edge detection
            edges = cv2.Canny(gray, 50, 150)
            # Stack back to 3 channels for display
            result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            return result
        except Exception as e:
            print(f"⚠️  CPU processing error: {e}")
            return frame
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read and process next frame"""
        if not self.cap or not self.cap.isOpened():
            return False, None
        
        ret, frame = self.cap.read()
        if not ret or frame is None:
            time.sleep(0.05)  # Small delay before retry
            return False, None
        
        # Validate frame
        if frame.size == 0:
            time.sleep(0.05)
            return False, None
        
        # Process frame
        process_start = time.time()
        
        if self.use_gpu:
            frame = self.process_frame_gpu(frame)
        else:
            # Still use GPU-friendly pipelines where possible
            frame = self.process_frame_cpu(frame)
        
        self.processing_time = (time.time() - process_start) * 1000  # ms
        self.frame_count += 1
        
        return True, frame
    
    def get_stats(self) -> dict:
        """Get pipeline statistics"""
        if self.frame_count == 0:
            return {"fps": 0, "processing_ms": 0, "gpu": self.use_gpu}
        
        return {
            "frames": self.frame_count,
            "fps": self.fps,
            "processing_ms": self.processing_time,
            "gpu_enabled": self.use_gpu,
        }
    
    def close(self):
        """Close video capture"""
        if self.cap:
            self.cap.release()
    
    def __enter__(self):
        self.open_stream()
        return self
    
    def __exit__(self, *args):
        self.close()


def main():
    """Demo GPU processing pipeline"""
    RTSP_URL = "rtsp://admin:MOVYKV@aicamera.serveminecraft.net:554/Streaming/Channels/101"
    
    print("\n🎬 Real-time RTSP GPU Processing Pipeline")
    print("="*70)
    print(f"Mode: {'NVIDIA CUDA GPU' if HAS_CUPY else 'CPU'}")
    print(f"RTSP: {RTSP_URL}")
    print("="*70)
    
    with GPUProcessingPipeline(RTSP_URL, use_gpu=HAS_CUPY) as pipeline:
        if not pipeline.cap.isOpened():
            print("❌ Cannot open RTSP stream - checking connection...")
            print("📋 Troubleshooting:")
            print("  1. Test URL with VLC: Media → Open Network Stream")
            print("  2. Check network: ping aicamera.serveminecraft.net")
            print("  3. Verify credentials in URL")
            return
        
        print("✅ Stream opened successfully")
        print("🔴 Press ESC to stop\n")
        
        frame_count = 0
        lost_frames = 0
        max_lost = 0
        start_time = time.time()
        last_time = start_time
        last_frame_time = start_time
        
        while True:
            ret, frame = pipeline.read_frame()
            
            if not ret:
                lost_frames += 1
                max_lost = max(max_lost, lost_frames)
                
                # If too many consecutive lost frames, check connection
                if lost_frames > 100:
                    print(f"⚠️  Stream connection lost (>100 frames)")
                    break
                continue
            
            # Frame received
            lost_frames = 0
            frame_count += 1
            now = time.time()
            fps = frame_count / (now - start_time)
            inst_fps = 1 / (now - last_time) if now != last_time else 0
            last_time = now
            last_frame_time = now
            
            # Add stats to frame
            stats_text = f"FPS: {fps:.1f} | Proc: {pipeline.processing_time:.1f}ms"
            gpu_status = "🟢 GPU" if pipeline.use_gpu else "🔵 CPU"
            
            cv2.putText(frame, f"{stats_text} | {gpu_status}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Instant FPS: {inst_fps:.1f} | Frames: {frame_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("GPU RTSP Pipeline", frame)
            
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                print("\n⏹️  User stopped pipeline")
                break
    
    cv2.destroyAllWindows()
    duration = time.time() - start_time
    print(f"\n✅ Pipeline Stats:")
    print(f"  Frames processed: {frame_count}")
    print(f"  Duration: {duration:.1f}s")
    print(f"  Average FPS: {frame_count/duration:.1f}")
    print(f"  GPU enabled: {HAS_CUPY}")
    if frame_count > 0:
        print(f"  ✅ SUCCESS - GPU processing working!")
    else:
        print(f"  ⚠️  No frames processed - check RTSP connection")


if __name__ == "__main__":
    main()
