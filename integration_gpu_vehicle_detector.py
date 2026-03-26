"""
Vehicle Detection with GPU Pipeline - Integration Guide
Ready-to-use example for running license plate detection on RTSP streams
"""

from gpu_pipeline_working import GPUPipelineSimple
import cv2
import time


def integrate_vehicle_detector():
    """
    Example: Run vehicle detector with GPU pipeline on RTSP stream
    """
    
    print("\n" + "="*70)
    print("Vehicle Detection - GPU Pipeline Integration")
    print("="*70 + "\n")
    
    # Configuration
    RTSP_URL = "rtsp://admin:MOVYKV@aicamera.serveminecraft.net:554/"
    DETECTOR_MODEL = "model/license_plate_detector.pt"  # Your vehicle detector
    CONFIDENCE_THRESHOLD = 0.5
    INFERENCE_RESOLUTION = (1280, 720)

    print(f"[CONFIG] RTSP Stream: {RTSP_URL}")
    print(f"[CONFIG] Model: {DETECTOR_MODEL}")
    print(f"[CONFIG] Confidence: {CONFIDENCE_THRESHOLD}")
    print(f"[CONFIG] Inference Resolution: {INFERENCE_RESOLUTION[0]}x{INFERENCE_RESOLUTION[1]}\n")


    
    # Create GPU pipeline with your vehicle detector
    print("[INIT] Initializing GPU Pipeline...")
    pipeline = GPUPipelineSimple(
        model_path=DETECTOR_MODEL,
        confidence=CONFIDENCE_THRESHOLD,
        use_gpu=True,
        inference_resolution=INFERENCE_RESOLUTION
    )
    
    # Process RTSP stream
    print("[STREAM] Starting RTSP stream processing...")
    print("[INFO] Processing frames with GPU inference...")
    print("[INFO] Press 'q' in display window to stop\n")
    
    stats = pipeline.process_stream_file(
        video_path=RTSP_URL,
        max_frames=500,  # Process 500 frames (adjust as needed)
        display=True  # Show detections in real-time
    )
    
    # Print results
    if stats:
        print("\n[COMPLETE] Vehicle Detection Test Results")
        print("="*70)
        print(f"Frames Processed: {stats['frames']}")
        print(f"Processing Time: {stats['time']:.2f}s")
        print(f"Average FPS: {stats['fps']:.2f}")
        print(f"Inference Time/Frame: {stats['inference_time_ms']:.2f}ms")
        print(f"Total Inference Time: {stats['total_inference_time']:.2f}s")
        print("="*70)
        
        # Check if performance meets requirements
        if stats['fps'] >= 10:
            print("✅ Performance EXCELLENT - Meets production requirements")
        elif stats['fps'] >= 5:
            print("✅ Performance GOOD - Suitable for monitoring")
        else:
            print("⚠️  Performance lower than expected - Check network")
    
    return stats


def integrate_with_tracking():
    """
    Advanced: Vehicle detection with tracking
    Maintains vehicle IDs across frames
    """
    
    print("\n" + "="*70)
    print("Vehicle Detection with Tracking - GPU Pipeline")
    print("="*70 + "\n")
    
    # Your existing tracker initialization
    # from VehicleDetectionTracker import VehicleDetectionTracker
    # tracker = VehicleDetectionTracker(config_path="config.yaml")
    
    RTSP_URL = "rtsp://admin:MOVYKV@aicamera.serveminecraft.net:554/"
    DETECTOR_MODEL = "model/license_plate_detector.pt"
    
    print("[INIT] Initializing GPU Pipeline with tracker...")
    pipeline = GPUPipelineSimple(
        model_path=DETECTOR_MODEL,
        confidence=0.5,
        use_gpu=True,
        inference_resolution=(1280, 720)
    )
    
    cap = cv2.VideoCapture(RTSP_URL)
    
    if not cap.isOpened():
        print("[ERROR] Cannot open RTSP stream")
        return None
    
    frame_count = 0
    tracks_detected = {}  # vehicle_id -> plate info
    
    print("[STREAM] Processing with tracking...")
    
    while cap.isOpened() and frame_count < 100:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # GPU inference
        start = time.time()
        results = pipeline.model(frame, conf=0.5, verbose=False, device=pipeline.device)
        inference_time = time.time() - start
        
        # Extract detections
        detections = []
        if len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].item()
                detection = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(conf),
                    'timestamp': time.time()
                }
                detections.append(detection)
        
        # Update tracker with detections
        # tracker.update(frame, detections)  # Your tracking logic
        
        # Display
        annotated = results[0].plot()
        cv2.putText(annotated, 
                   f"GPU FPS: {1/max(inference_time, 0.001):.1f} | Detections: {len(detections)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Vehicle Detection + Tracking", annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        if frame_count % 10 == 0:
            print(f"  Frame {frame_count}: {len(detections)} detections | GPU: {inference_time*1000:.1f}ms")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n[COMPLETE] Processed {frame_count} frames with tracking")
    return tracks_detected


def benchmark_detector_model():
    """
    Test different detector models and compare performance
    """
    
    print("\n" + "="*70)
    print("Detector Model Comparison - GPU Pipeline")
    print("="*70 + "\n")
    
    models = [
        ("model/license_plate_detector.pt", "License Plate Detector"),
        ("yolov8n.pt", "YOLOv8 Nano"),
        ("yolov8s.pt", "YOLOv8 Small (if available)"),
    ]
    
    RTSP_URL = "rtsp://admin:MOVYKV@aicamera.serveminecraft.net:554/"
    
    results = {}
    
    for model_path, model_name in models:
        print(f"\n[TEST] Testing: {model_name} ({model_path})")
        
        try:
            pipeline = GPUPipelineSimple(
                model_path=model_path,
                confidence=0.5,
                use_gpu=True,
                inference_resolution=(1280, 720)
            )
            
            # Test on 10 frames
            stats = pipeline.process_stream_file(
                RTSP_URL,
                max_frames=10,
                display=False
            )
            
            if stats:
                results[model_name] = {
                    'fps': stats['fps'],
                    'inference_ms': stats['inference_time_ms']
                }
                print(f"  ✓ {stats['fps']:.2f} FPS | Inference: {stats['inference_time_ms']:.2f}ms")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    # Summary
    print(f"\n{'='*70}")
    print("Model Comparison Summary")
    print(f"{'='*70}")
    for model_name, stats in results.items():
        print(f"{model_name:30} | FPS: {stats['fps']:6.2f} | Inference: {stats['inference_ms']:6.2f}ms")
    print(f"{'='*70}\n")
    
    return results


if __name__ == "__main__":
    import sys
    
    print("\nVehicle Detection - GPU Pipeline Integration Examples")
    print("Usage: python integration_gpu_vehicle_detector.py [mode]")
    print("\nModes:")
    print("  basic     - Basic vehicle detection (default)")
    print("  tracking  - With vehicle tracking")
    print("  benchmark - Compare different models\n")
    
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "basic"
    
    if mode == "tracking":
        integrate_with_tracking()
    elif mode == "benchmark":
        benchmark_detector_model()
    else:
        integrate_vehicle_detector()
