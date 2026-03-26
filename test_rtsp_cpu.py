"""
RTSP CPU Processing Pipeline Test (GPU fallback)
Simpler test to verify RTSP connection and processing work
"""

import cv2
import numpy as np
import time
from typing import Tuple, Optional


class CPUProcessingPipeline:
    """Simple CPU-based RTSP video processing"""
    
    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        self.cap = None
        self.frame_count = 0
        self.processing_time = 0
        
    def open_stream(self) -> bool:
        """Open RTSP stream"""
        print(f"🔌 Connecting to RTSP: {self.rtsp_url[:50]}...")
        
        # Try primary backend
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if self.cap.isOpened():
            print("✅ Connected with CAP_FFMPEG")
            return True
        
        # Try default backend
        print("⏳ Fallback to default backend...")
        self.cap = cv2.VideoCapture(self.rtsp_url)
        
        if self.cap.isOpened():
            print("✅ Connected with default backend")
            return True
        
        print("❌ Failed to connect")
        return False
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """CPU-based frame processing"""
        try:
            # Fast CPU processing: grayscale + edge detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            return result
        except Exception as e:
            print(f"⚠️ Processing error: {e}")
            return frame
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read and process next frame"""
        if not self.cap or not self.cap.isOpened():
            return False, None
        
        ret, frame = self.cap.read()
        if not ret or frame is None or frame.size == 0:
            time.sleep(0.01)
            return False, None
        
        # Process frame
        process_start = time.time()
        frame = self.process_frame(frame)
        self.processing_time = (time.time() - process_start) * 1000
        self.frame_count += 1
        
        return True, frame
    
    def close(self):
        """Close video capture"""
        if self.cap:
            self.cap.release()


def main():
    """Test RTSP pipeline"""
    RTSP_URL = "rtsp://admin:MOVYKV@aicamera.serveminecraft.net:554/Streaming/Channels/101"
    
    print("\n🎬 RTSP Pipeline Test (CPU Processing)")
    print("="*70)
    
    pipeline = CPUProcessingPipeline(RTSP_URL)
    
    try:
        if not pipeline.open_stream():
            print("\n📋 Connection Troubleshooting:")
            print("  1. Test URL manually with VLC:")
            print(f"     File → Open Network Stream → {RTSP_URL}")
            print("  2. Verify network connectivity:")
            print("     ping aicamera.serveminecraft.net")
            print("  3. Check camera credentials in URL")
            return
        
        print("🔴 Press ESC to stop\n")
        
        frame_count = 0
        lost_frames = 0
        start_time = time.time()
        last_time = start_time
        
        while True:
            ret, frame = pipeline.read_frame()
            
            if not ret:
                lost_frames += 1
                if lost_frames > 100:
                    print(f"\n⚠️  Connection lost after {frame_count} frames")
                    break
                continue
            
            lost_frames = 0
            frame_count += 1
            now = time.time()
            
            avg_fps = frame_count / (now - start_time)
            inst_fps = 1 / (now - last_time) if now != last_time else 0
            last_time = now
            
            # Show stats on frame
            stats = f"FPS: {avg_fps:.1f} | Proc: {pipeline.processing_time:.1f}ms"
            cv2.putText(frame, stats, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Frames: {frame_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("RTSP Test", frame)
            
            if cv2.waitKey(1) == 27:  # ESC
                break
        
        cv2.destroyAllWindows()
        
        # Results
        print(f"\n✅ TEST COMPLETE")
        print(f"  Frames processed: {frame_count}")
        print(f"  Duration: {(time.time() - start_time):.1f}s")
        if frame_count > 0:
            print(f"  Average FPS: {frame_count / (time.time() - start_time):.1f}")
            print(f"  ✅ RTSP connection & processing WORKING!")
        
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
