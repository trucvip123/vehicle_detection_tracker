"""
Direct NVIDIA GPU H.264 decoding using NVDEC library
Uses FFmpeg with GPU acceleration backend
"""

import cv2
import numpy as np
import subprocess
import threading
from queue import Queue
import time


class GPUVideoDecoder:
    """Real-time GPU-accelerated RTSP video decoder using FFmpeg NVDEC"""
    
    def __init__(self, rtsp_url: str, buffer_size: int = 2):
        self.rtsp_url = rtsp_url
        self.buffer_size = buffer_size
        self.frame_queue = Queue(maxsize=buffer_size)
        self.is_running = False
        self.thread = None
        self.process = None
        self.frame_count = 0
        self.fps = 0
        
    def _build_ffmpeg_command(self) -> list:
        """Build FFmpeg command with NVIDIA GPU hardware acceleration"""
        # Try to use NVIDIA GPU decoder if available
        return [
            'ffmpeg',
            '-rtsp_transport', 'tcp',
            '-i', self.rtsp_url,
            '-c:v', 'h264_cuvid',  # NVIDIA GPU decoder
            '-gpu', '0',             # GPU device ID
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-',
        ]
    
    def _build_ffmpeg_fallback_command(self) -> list:
        """Build FFmpeg command with software fallback"""
        return [
            'ffmpeg',
            '-rtsp_transport', 'tcp',
            '-i', self.rtsp_url,
            '-c:v', 'libx264',  # Software decoder
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-',
        ]
    
    def _decode_thread(self, use_gpu: bool = True):
        """Thread for decoding video frames"""
        cmd = self._build_ffmpeg_command() if use_gpu else self._build_ffmpeg_fallback_command()
        
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10 * 1024 * 1024,
            )
            
            print(f"✅ FFmpeg process started (GPU: {use_gpu})")
            
            # Get frame size (assuming 1920x1080) - adjust if needed
            frame_width, frame_height = 1920, 1080
            frame_size = frame_width * frame_height * 3
            
            while self.is_running:
                raw_frame = self.process.stdout.read(frame_size)
                if not raw_frame:
                    print("⚠️ Lost connection to stream")
                    break
                
                frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(
                    (frame_height, frame_width, 3)
                )
                
                try:
                    self.frame_queue.put_nowait(frame)
                except:
                    pass  # Drop frames if queue is full
                
                self.frame_count += 1
                
        except Exception as e:
            print(f"❌ Decoding error: {e}")
        finally:
            if self.process:
                self.process.kill()
    
    def start(self):
        """Start GPU video decoding"""
        self.is_running = True
        self.thread = threading.Thread(target=self._decode_thread, args=(True,), daemon=True)
        self.thread.start()
        time.sleep(1)  # Wait for thread to start
        
        # Check if GPU decoder worked, fallback to software if not
        if self.frame_queue.empty():
            print("⚠️ GPU decoder failed, falling back to software...")
            self.stop()
            self.is_running = True
            self.thread = threading.Thread(target=self._decode_thread, args=(False,), daemon=True)
            self.thread.start()
    
    def get_frame(self, timeout: float = 1.0):
        """Get next decoded frame"""
        try:
            frame = self.frame_queue.get(timeout=timeout)
            return True, frame
        except:
            return False, None
    
    def stop(self):
        """Stop decoding"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=5)
        if self.process:
            self.process.kill()
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()


def test_gpu_decoder(rtsp_url: str):
    """Test GPU video decoder"""
    print("🎬 Testing GPU Video Decoder...")
    
    with GPUVideoDecoder(rtsp_url) as decoder:
        frame_count = 0
        start_time = time.time()
        last_time = start_time
        
        while True:
            ret, frame = decoder.get_frame(timeout=2.0)
            if not ret:
                print("⚠️ Timeout getting frame")
                continue
            
            frame_count += 1
            now = time.time()
            avg_fps = frame_count / (now - start_time)
            inst_fps = 1 / (now - last_time) if now != last_time else 0
            last_time = now
            
            # Add FPS text
            cv2.putText(frame, f"GPU FPS(avg): {avg_fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"GPU FPS(inst): {inst_fps:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("GPU RTSP Decoder", frame)
            
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    RTSP_URL = "rtsp://admin:MOVYKV@aicamera.serveminecraft.net:554/Streaming/Channels/101"
    test_gpu_decoder(RTSP_URL)
