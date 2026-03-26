#!/usr/bin/env python3
"""
NVIDIA GPU Optimized YOLOv8 Quantization
Uses TensorRT for int8 quantization on NVIDIA GPU
Much faster than CPU quantization, optimized for deployment

Usage:
    python quantize_tensorrt_gpu.py [input_model] [output_model]
    
Example:
    python quantize_tensorrt_gpu.py yolov8n.pt yolov8n_int8.engine
"""

import os
import sys
from pathlib import Path
import time
import cv2
import numpy as np
from ultralytics import YOLO
import torch

def check_gpu_setup():
    """Check if NVIDIA GPU is properly set up."""
    print("🔍 Checking NVIDIA GPU Setup...")
    print("-" * 80)
    
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"✓ CUDA Available: {cuda_available}")
    
    if not cuda_available:
        print("❌ CUDA not available!")
        print("\n   Install NVIDIA drivers and CUDA toolkit:")
        print("   https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/")
        return False
    
    # GPU Info
    gpu_count = torch.cuda.device_count()
    print(f"✓ GPU Count: {gpu_count}")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    
    # CUDA Version
    cuda_version = torch.version.cuda
    print(f"✓ CUDA Version: {cuda_version}")
    
    # cuDNN
    cudnn_enabled = torch.backends.cudnn.enabled
    print(f"✓ cuDNN Enabled: {cudnn_enabled}")
    
    print("-" * 80 + "\n")
    return True


def quantize_with_tensorrt(input_model="yolov8n.pt", output_model="yolov8n_int8.engine"):
    """
    Quantize YOLOv8 to TensorRT int8 format for NVIDIA GPU.
    
    Args:
        input_model: Path to input YOLOv8 model (PT format)
        output_model: Path to output TensorRT engine (ENGINE format)
    
    Returns:
        bool: Success status
    """
    
    print("=" * 80)
    print("TENSORRT INT8 QUANTIZATION FOR NVIDIA GPU")
    print("=" * 80)
    
    # Check GPU
    if not check_gpu_setup():
        return False
    
    # Check input model
    if not Path(input_model).exists():
        print(f"❌ Input model not found: {input_model}")
        return False
    
    print(f"📥 Loading model: {input_model}")
    print("-" * 80)
    
    try:
        model = YOLO(input_model)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    print("\n🔧 Exporting to TensorRT int8 format...")
    print("-" * 80)
    print("   This may take 2-5 minutes on first run")
    print("   (Includes calibration on sample data)")
    print()
    
    try:
        # Get the output filename without extension for ultralytics
        engine_name = str(output_model).replace(".engine", "")
        
        # Export to TensorRT with int8 quantization
        # The model will use GPU for inference automatically
        engine_path = model.export(
            format="engine",           # TensorRT format
            imgsz=640,                 # Input image size
            half=False,                # Use full precision for quantization calibration
            int8=True,                 # Enable int8 quantization
            device=0,                  # GPU device ID (0 for first GPU)
            workspace=4,               # TensorRT workspace in GB
            dynamic=False,             # No dynamic shape (faster)
            simplify=True,             # Simplify model graph
        )
        
        print(f"\n✅ TensorRT Export Complete!")
        print(f"   Output: {engine_path}")
        
        # Check file size
        if Path(engine_path).exists():
            size_mb = Path(engine_path).stat().st_size / (1024 * 1024)
            original_size = Path(input_model).stat().st_size / (1024 * 1024)
            
            print(f"\n📊 Model Sizes:")
            print(f"   Original (PT):  {original_size:.1f}MB")
            print(f"   TensorRT (INT8): {size_mb:.1f}MB")
            print(f"   Reduction: {((original_size - size_mb) / original_size * 100):.0f}%")
            
            # Rename to requested output if different
            if str(engine_path) != str(output_model):
                import shutil
                shutil.move(str(engine_path), output_model)
                print(f"   Saved as: {output_model}")
            
            return True
        else:
            print(f"❌ Engine file not created")
            return False
    
    except Exception as e:
        print(f"❌ TensorRT export failed: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Make sure NVIDIA driver is up to date")
        print("   2. Install TensorRT: pip install tensorrt")
        print("   3. Check CUDA/cuDNN compatibility")
        return False


def compare_gpu_models(original_model="yolov8n.pt", tensorrt_model="yolov8n_int8.engine", 
                       test_video="video/h.mp4", num_frames=50, warmup_frames=5):
    """
    Compare performance of original vs TensorRT quantized model on GPU.
    
    Args:
        original_model: Original PT model
        tensorrt_model: TensorRT INT8 engine
        test_video: Video for testing
        num_frames: Frames to test
        warmup_frames: Frames for GPU warmup
    """
    
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON: ORIGINAL (PT) vs TensorRT (INT8)")
    print("=" * 80)
    
    if not Path(test_video).exists():
        print(f"❌ Test video not found: {test_video}")
        return
    
    # Test original PT model
    print(f"\n📊 Testing Original Model (PT): {original_model}")
    print("-" * 80)
    
    try:
        model_pt = YOLO(original_model)
        original_times = []
        
        cap = cv2.VideoCapture(test_video)
        frame_count = 0
        
        print(f"   Warmup: {warmup_frames} frames (GPU initialization)")
        
        while frame_count < (num_frames + warmup_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Warmup phase
            if frame_count <= warmup_frames:
                _ = model_pt.track(frame, persist=True, verbose=False, conf=0.4)
                print(f"   Warmup frame {frame_count}/{warmup_frames}", end="\r")
                continue
            
            # Test phase
            torch.cuda.synchronize()  # Wait for GPU to finish
            start = time.perf_counter()
            
            _ = model_pt.track(frame, persist=True, verbose=False, conf=0.4)
            
            torch.cuda.synchronize()  # Wait for GPU to finish
            elapsed = (time.perf_counter() - start) * 1000
            original_times.append(elapsed)
            
            if len(original_times) % 10 == 0:
                print(f"   Frame {len(original_times)}: {elapsed:.2f}ms")
        
        cap.release()
        
        if original_times:
            avg_original = sum(original_times) / len(original_times)
            min_original = min(original_times)
            max_original = max(original_times)
            fps_original = 1000 / avg_original
            
            print(f"\n✓ Original Model Results:")
            print(f"  Average: {avg_original:.2f}ms")
            print(f"  Min/Max: {min_original:.2f}ms / {max_original:.2f}ms")
            print(f"  FPS: {fps_original:.1f}")
        else:
            print("❌ No frames processed")
            return
    
    except Exception as e:
        print(f"❌ Failed: {e}")
        return
    
    # Test TensorRT model
    print(f"\n📊 Testing TensorRT Model (INT8): {tensorrt_model}")
    print("-" * 80)
    
    if not Path(tensorrt_model).exists():
        print(f"⚠️  TensorRT model not found: {tensorrt_model}")
        print("   Run quantization first: python quantize_tensorrt_gpu.py")
        return
    
    try:
        model_engine = YOLO(tensorrt_model)
        tensorrt_times = []
        
        cap = cv2.VideoCapture(test_video)
        frame_count = 0
        
        print(f"   Warmup: {warmup_frames} frames (GPU initialization)")
        
        while frame_count < (num_frames + warmup_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Warmup phase
            if frame_count <= warmup_frames:
                _ = model_engine.track(frame, persist=True, verbose=False, conf=0.4)
                print(f"   Warmup frame {frame_count}/{warmup_frames}", end="\r")
                continue
            
            # Test phase
            torch.cuda.synchronize()  # Wait for GPU to finish
            start = time.perf_counter()
            
            _ = model_engine.track(frame, persist=True, verbose=False, conf=0.4)
            
            torch.cuda.synchronize()  # Wait for GPU to finish
            elapsed = (time.perf_counter() - start) * 1000
            tensorrt_times.append(elapsed)
            
            if len(tensorrt_times) % 10 == 0:
                print(f"   Frame {len(tensorrt_times)}: {elapsed:.2f}ms")
        
        cap.release()
        
        if tensorrt_times:
            avg_tensorrt = sum(tensorrt_times) / len(tensorrt_times)
            min_tensorrt = min(tensorrt_times)
            max_tensorrt = max(tensorrt_times)
            fps_tensorrt = 1000 / avg_tensorrt
            
            print(f"\n✓ TensorRT Model Results:")
            print(f"  Average: {avg_tensorrt:.2f}ms")
            print(f"  Min/Max: {min_tensorrt:.2f}ms / {max_tensorrt:.2f}ms")
            print(f"  FPS: {fps_tensorrt:.1f}")
        else:
            print("❌ No frames processed")
            return
    
    except Exception as e:
        print(f"❌ Failed: {e}")
        return
    
    # Calculate and display improvement
    print("\n" + "=" * 80)
    print("PERFORMANCE IMPROVEMENT")
    print("=" * 80)
    
    speedup = avg_original / avg_tensorrt
    improvement = ((avg_original - avg_tensorrt) / avg_original) * 100
    fps_gain = fps_tensorrt - fps_original
    
    print(f"\nLatency:")
    print(f"  Original: {avg_original:.2f}ms → TensorRT: {avg_tensorrt:.2f}ms")
    print(f"  Speedup: {speedup:.2f}x ⚡")
    
    print(f"\nThroughput:")
    print(f"  Original: {fps_original:.1f} FPS → TensorRT: {fps_tensorrt:.1f} FPS")
    print(f"  Gain: +{fps_gain:.0f} FPS (+{improvement:.0f}%)")
    
    if speedup > 1.0:
        print(f"\n✅ SIGNIFICANT IMPROVEMENT - {speedup:.2f}x faster!")
        print(f"   👉 Use TensorRT model for production deployment")
    else:
        print(f"\n⚠️  Limited improvement observed")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("NVIDIA GPU Optimized YOLOv8 Quantization")
        print("\nUsage:")
        print("  python quantize_tensorrt_gpu.py [input_model] [output_model] [--compare] [--video video_file] [--frames num_frames]")
        print("\nExamples:")
        print("  # Quantize only:")
        print("  python quantize_tensorrt_gpu.py yolov8n.pt yolov8n_int8.engine")
        print("\n  # Quantize and compare:")
        print("  python quantize_tensorrt_gpu.py yolov8n.pt yolov8n_int8.engine --compare --video video/h.mp4 --frames 100")
        sys.exit(1)
    
    # Parse arguments
    input_model = sys.argv[1]
    output_model = sys.argv[2] if len(sys.argv) > 2 else "yolov8n_int8.engine"
    
    # Flags
    do_compare = "--compare" in sys.argv
    video_file = "video/h.mp4"
    num_frames = 100
    
    if "--video" in sys.argv:
        idx = sys.argv.index("--video")
        if idx + 1 < len(sys.argv):
            video_file = sys.argv[idx + 1]
    
    if "--frames" in sys.argv:
        idx = sys.argv.index("--frames")
        if idx + 1 < len(sys.argv):
            num_frames = int(sys.argv[idx + 1])
    
    # Run quantization
    success = quantize_with_tensorrt(input_model, output_model)
    
    # Compare if requested
    if success and do_compare:
        compare_gpu_models(input_model, output_model, video_file, num_frames)
        
        # Deployment instructions
        print("\n" + "=" * 80)
        print("DEPLOYMENT INSTRUCTIONS")
        print("=" * 80)
        print(f"\n1. Update config to use TensorRT model:")
        print(f"   model_path = \"{output_model}\"")
        print(f"\n2. Update StreamingCameraTest.py or config.yaml")
        print(f"\n3. Run with TensorRT:")
        print(f"   python StreamingCameraTest.py")
        print(f"\n4. Monitor GPU usage:")
        print(f"   nvidia-smi -l 1")
        print("\n" + "=" * 80)
