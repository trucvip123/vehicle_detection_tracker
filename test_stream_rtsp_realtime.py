"""
Test script for RTSP/video stream with real-time display and performance timing.
Includes vehicle detection with timing logs.
"""

import cv2
import sys
import time
from datetime import datetime
from VehicleDetectionTracker.logging_utils import log
from VehicleDetectionTracker.performance_timing import time_block
from ultralytics import YOLO
from VehicleDetectionTracker.device_utils import get_device
import numpy as np

def test_stream_imshow(video_source="video/h.mp4", stream_frame_size=(1280, 720), max_frames=None, detect_vehicles=True, model_path="yolov8n.pt"):
    """
    Test streaming with imshow display and optional vehicle detection.
    
    Args:
        video_source: Path to video file or RTSP URL
        stream_frame_size: Size to resize frames to (width, height)
        max_frames: Maximum frames to process (None = all)
        detect_vehicles: Whether to run vehicle detection
        model_path: Path to YOLO model
    """
    log(f"[TEST] Starting stream test: {video_source}")
    log(f"[TEST] Frame size: {stream_frame_size}")
    log(f"[TEST] Vehicle detection: {detect_vehicles}")
    
    # Load YOLO model if detection enabled
    model = None
    device = None
    if detect_vehicles:
        log("[TEST] Loading YOLO vehicle detection model...")
        try:
            device, use_gpu = get_device(log)
            model = YOLO(model_path)
            if use_gpu and device != "cpu":
                model.to(device)
            log(f"[TEST] Model loaded on {device}")
        except Exception as e:
            log(f"[ERROR] Failed to load model: {e}")
            detect_vehicles = False
    
    # Open video source
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        log(f"[ERROR] Cannot open video source: {video_source}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    log(f"[INFO] Video properties:")
    log(f"       FPS: {fps}")
    log(f"       Total frames: {frame_count}")
    log(f"       Resolution: {width}x{height}")
    
    frame_id = 0
    frame_times = []
    detect_times = []
    display_times = []
    start_time = datetime.now()
    
    # Create window
    window_name = f"Stream Test - {video_source}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    try:
        while True:
            # === TIME BLOCK: READ FRAME ===
            with time_block("[STREAM_READ]", log):
                success, frame = cap.read()
                frame_id += 1
            
            if not success or frame is None:
                log(f"[INFO] End of stream at frame {frame_id}")
                break
            
            # Check max frames limit
            if max_frames and frame_id > max_frames:
                log(f"[INFO] Reached max frames limit ({max_frames})")
                break
            
            # === TIME BLOCK: RESIZE FRAME ===
            with time_block("[STREAM_RESIZE]", log):
                if stream_frame_size and frame is not None:
                    try:
                        frame = cv2.resize(frame, stream_frame_size, interpolation=cv2.INTER_AREA)
                    except Exception as e:
                        log(f"[WARNING] Resize failed: {e}")
            
            # === TIME BLOCK: VEHICLE DETECTION ===
            if detect_vehicles and model:
                detect_start = time.time()
                with time_block("[VEHICLE_DETECT]", log):
                    try:
                        results = model.track(frame, verbose=False, conf=0.4)
                        if results and len(results) > 0:
                            result = results[0]
                            
                            # Draw detections
                            if result.boxes and len(result.boxes) > 0:
                                num_detections = len(result.boxes)
                                for box in result.boxes:
                                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                                    conf = float(box.conf[0].cpu().numpy())
                                    track_id = int(box.id[0].cpu().numpy()) if box.id is not None else -1
                                    
                                    # Draw bounding box
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    
                                    # Draw label
                                    label = f"ID:{track_id} {conf:.2f}" if track_id >= 0 else f"{conf:.2f}"
                                    cv2.putText(frame, label, (x1, y1 - 10),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                
                                log(f"[DETECT] Frame {frame_id}: {num_detections} vehicles detected")
                    except Exception as e:
                        log(f"[ERROR] Detection failed: {e}")
                
                detect_time = (time.time() - detect_start) * 1000
                detect_times.append(detect_time)
            
            # === TIME BLOCK: DISPLAY FRAME ===
            frame_start = time.time()
            with time_block("[STREAM_DISPLAY]", log):
                # Add frame number and stats
                cv2.putText(frame, f"Frame: {frame_id}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if detect_vehicles and detect_times:
                    avg_detect = sum(detect_times[-30:]) / min(30, len(detect_times))
                    cv2.putText(frame, f"Detect: {avg_detect:.1f}ms", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                cv2.imshow(window_name, frame)
                # Press 'q' to quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    log(f"[INFO] User pressed 'q', exiting...")
                    break
            
            frame_time = (time.time() - frame_start) * 1000  # Convert to ms
            frame_times.append(frame_time)
            
            # Log every 30 frames
            if frame_id % 30 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                current_fps = frame_id / elapsed if elapsed > 0 else 0
                avg_frame_time = sum(frame_times[-30:]) / 30 if len(frame_times) >= 30 else 0
                
                log(f"[STATS] Frame {frame_id}: FPS={current_fps:.1f}, Avg frame time={avg_frame_time:.2f}ms", )
                if detect_vehicles and detect_times:
                    avg_detect = sum(detect_times[-30:]) / min(30, len(detect_times))
                    log(f"        Avg detect time={avg_detect:.2f}ms")
    
    except KeyboardInterrupt:
        log(f"[INFO] Interrupted by user at frame {frame_id}")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        if frame_times:
            elapsed = (datetime.now() - start_time).total_seconds()
            avg_fps = frame_id / elapsed if elapsed > 0 else 0
            avg_frame_time = sum(frame_times) / len(frame_times)
            min_frame_time = min(frame_times)
            max_frame_time = max(frame_times)
            
            log(f"\n[SUMMARY] Stream Test Results:")
            log(f"          Total frames: {frame_id}")
            log(f"          Elapsed time: {elapsed:.2f}s")
            log(f"          Average FPS: {avg_fps:.2f}")
            log(f"          Avg frame time: {avg_frame_time:.2f}ms")
            log(f"          Min frame time: {min_frame_time:.2f}ms")
            log(f"          Max frame time: {max_frame_time:.2f}ms")
            
            if detect_vehicles and detect_times:
                avg_detect = sum(detect_times) / len(detect_times)
                min_detect = min(detect_times)
                max_detect = max(detect_times)
                log(f"          Avg detection time: {avg_detect:.2f}ms")
                log(f"          Min detection time: {min_detect:.2f}ms")
                log(f"          Max detection time: {max_detect:.2f}ms")
                log(f"          Detection overhead: {(avg_detect/avg_frame_time)*100:.1f}% of frame time")


def test_rtsp_stream(rtsp_url, stream_frame_size=(1280, 720), max_frames=None, detect_vehicles=True, model_path="yolov8n.pt"):
    """
    Test RTSP stream specifically.
    
    Args:
        rtsp_url: RTSP URL (e.g., "rtsp://ip:port/stream")
        stream_frame_size: Size to resize frames to
        max_frames: Maximum frames to process
        detect_vehicles: Whether to run vehicle detection
        model_path: Path to YOLO model
    """
    log(f"[TEST] Testing RTSP stream: {rtsp_url}")
    test_stream_imshow(rtsp_url, stream_frame_size, max_frames, detect_vehicles, model_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test stream with real-time display and vehicle detection")
    parser.add_argument("--source", type=str, default="video/h.mp4", 
                       help="Video file or RTSP URL")
    parser.add_argument("--width", type=int, default=1280, 
                       help="Frame width")
    parser.add_argument("--height", type=int, default=720, 
                       help="Frame height")
    parser.add_argument("--max-frames", type=int, default=None, 
                       help="Maximum frames to process")
    parser.add_argument("--rtsp", action="store_true", 
                       help="Test RTSP stream")
    parser.add_argument("--no-detect", action="store_true", 
                       help="Disable vehicle detection")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                       help="Path to YOLO model")
    
    args = parser.parse_args()
    
    stream_size = (args.width, args.height)
    detect_vehicles = not args.no_detect
    
    if args.rtsp:
        test_rtsp_stream(args.source, stream_size, args.max_frames, detect_vehicles, args.model)
    else:
        test_stream_imshow(args.source, stream_size, args.max_frames, detect_vehicles, args.model)
