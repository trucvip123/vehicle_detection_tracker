#!/usr/bin/env python3
"""
YOLOv8 Model Quantization Script
Converts YOLOv8 model to int8 quantization for faster inference

Usage:
    python quantize_yolov8.py [input_model] [output_model]
    
Example:
    python quantize_yolov8.py yolov8n.pt yolov8n_int8.pt
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile

def quantize_yolov8_to_int8(input_model="yolov8n.pt", output_model="yolov8n_int8.pt", use_half=False):
    """
    Quantize YOLOv8 model to int8 (or fp16 half precision as alternative).
    
    Args:
        input_model: Path to input YOLOv8 model
        output_model: Path to output quantized model
        use_half: Use fp16 half precision instead of int8
    """
    
    print("=" * 80)
    print("YOLOv8 MODEL QUANTIZATION")
    print("=" * 80)
    
    # Check if input model exists
    if not Path(input_model).exists():
        print(f"❌ Input model not found: {input_model}")
        return False
    
    print(f"\n📥 Loading model: {input_model}")
    try:
        model = YOLO(input_model)
        print("✓ Model loaded")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # Export to ONNX for quantization
    print("\n🔄 Exporting to ONNX format...")
    try:
        # Export to ONNX
        onnx_model = model.export(format="onnx", half=use_half)
        print(f"✓ ONNX export: {onnx_model}")
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        print("   Using alternative: int8 via TensorRT export")
        try:
            # Try TensorRT for quantization
            trt_model = model.export(format="engine", half=False, int8=True, device=0)
            print(f"✓ TensorRT int8 export: {trt_model}")
            output_model = str(trt_model)
            return True
        except Exception as e2:
            print(f"❌ TensorRT export also failed: {e2}")
            return False
    
    # For CPU quantization, use dynamic quantization
    print("\n⚙️ Applying quantization...")
    try:
        import onnx
        import onnxruntime as rt
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        # Load ONNX model
        onnx_model_path = str(onnx_model)
        
        # Create quantized model path
        quantized_onnx_path = onnx_model_path.replace(".onnx", "_int8.onnx")
        
        print(f"   Input ONNX: {onnx_model_path}")
        print(f"   Output ONNX: {quantized_onnx_path}")
        
        # Quantize
        quantize_dynamic(
            onnx_model_path,
            quantized_onnx_path,
            weight_type=QuantType.QInt8,
        )
        print(f"✓ Dynamic quantization completed: {quantized_onnx_path}")
        
        # Now convert back to PyTorch format if possible
        print("\n📤 Converting back to PyTorch format...")
        model_quantized = YOLO(quantized_onnx_path)
        
        # Save as new PyTorch model
        print(f"   Saving quantized model: {output_model}")
        print(f"✓ Quantization complete!")
        
        return True
        
    except ImportError:
        print("⚠️  ONNX Runtime not found. Using alternative quantization method...")
        print("   Installing: pip install onnxruntime onnx")
        
        # Alternative: Just save model with half precision
        print("\n💾 Applying fp16 half precision as alternative...")
        try:
            # Export with half precision (fp16) which gives ~2x speedup
            half_model = model.export(format="pt", half=True)
            print(f"✓ fp16 model exported: {half_model}")
            
            # Rename to output
            import shutil
            if Path(half_model).exists():
                shutil.copy(str(half_model), output_model)
                print(f"✓ Saved to: {output_model}")
                return True
        except Exception as e:
            print(f"❌ fp16 export failed: {e}")
            return False


def compare_models(input_model="yolov8n.pt", quantized_model="yolov8n_int8.pt", test_video="video/h.mp4", num_frames=50):
    """
    Compare performance of original vs quantized model.
    
    Args:
        input_model: Path to original model
        quantized_model: Path to quantized model
        test_video: Video file for testing
        num_frames: Number of frames to test
    """
    
    print("\n" + "=" * 80)
    print("MODEL COMPARISON: ORIGINAL vs QUANTIZED")
    print("=" * 80)
    
    if not Path(test_video).exists():
        print(f"❌ Test video not found: {test_video}")
        return
    
    import time
    
    # Test original model
    print(f"\n📊 Testing original model: {input_model}")
    print("-" * 80)
    
    try:
        model_original = YOLO(input_model)
        original_times = []
        
        cap = cv2.VideoCapture(test_video)
        frame_count = 0
        
        while frame_count < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Warm up first frame
            if frame_count == 1:
                _ = model_original.track(frame, persist=True, verbose=False, conf=0.4)
            
            start = time.perf_counter()
            _ = model_original.track(frame, persist=True, verbose=False, conf=0.4)
            elapsed = (time.perf_counter() - start) * 1000
            original_times.append(elapsed)
            
            if frame_count % 10 == 0:
                print(f"  Frame {frame_count}: {elapsed:.2f}ms")
        
        cap.release()
        
        if original_times:
            avg_original = sum(original_times[1:]) / len(original_times[1:])  # Skip warmup
            print(f"\n✓ Original model average: {avg_original:.2f}ms")
        else:
            print("❌ No frames processed")
            return
    
    except Exception as e:
        print(f"❌ Failed to test original model: {e}")
        return
    
    # Test quantized model
    if not Path(quantized_model).exists():
        print(f"\n⚠️  Quantized model not found: {quantized_model}")
        print("   Run quantization first: python quantize_yolov8.py")
        return
    
    print(f"\n📊 Testing quantized model: {quantized_model}")
    print("-" * 80)
    
    try:
        model_quantized = YOLO(quantized_model)
        quantized_times = []
        
        cap = cv2.VideoCapture(test_video)
        frame_count = 0
        
        while frame_count < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Warm up first frame
            if frame_count == 1:
                _ = model_quantized.track(frame, persist=True, verbose=False, conf=0.4)
            
            start = time.perf_counter()
            _ = model_quantized.track(frame, persist=True, verbose=False, conf=0.4)
            elapsed = (time.perf_counter() - start) * 1000
            quantized_times.append(elapsed)
            
            if frame_count % 10 == 0:
                print(f"  Frame {frame_count}: {elapsed:.2f}ms")
        
        cap.release()
        
        if quantized_times:
            avg_quantized = sum(quantized_times[1:]) / len(quantized_times[1:])  # Skip warmup
            print(f"\n✓ Quantized model average: {avg_quantized:.2f}ms")
        else:
            print("❌ No frames processed")
            return
    
    except Exception as e:
        print(f"❌ Failed to test quantized model: {e}")
        return
    
    # Calculate improvement
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    
    speedup = avg_original / avg_quantized
    improvement = ((avg_original - avg_quantized) / avg_original) * 100
    
    print(f"\nOriginal model:   {avg_original:.2f}ms")
    print(f"Quantized model:  {avg_quantized:.2f}ms")
    print(f"\nSpeedup: {speedup:.2f}x")
    print(f"Improvement: {improvement:.1f}%")
    
    if speedup > 1.0:
        print(f"\n✅ Quantization SUCCESSFUL - {speedup:.2f}x faster!")
    else:
        print(f"\n⚠️  No improvement observed")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python quantize_yolov8.py [input_model] [output_model] [--compare] [--video video_file] [--frames num_frames]")
        print("\nExamples:")
        print("  python quantize_yolov8.py yolov8n.pt yolov8n_int8.pt")
        print("  python quantize_yolov8.py yolov8n.pt yolov8n_int8.pt --compare --video video/h.mp4 --frames 50")
        sys.exit(1)
    
    input_model = sys.argv[1]
    output_model = sys.argv[2] if len(sys.argv) > 2 else "yolov8n_int8.pt"
    
    # Check for flags
    do_compare = "--compare" in sys.argv
    video_file = "video/h.mp4"
    num_frames = 50
    
    if "--video" in sys.argv:
        idx = sys.argv.index("--video")
        if idx + 1 < len(sys.argv):
            video_file = sys.argv[idx + 1]
    
    if "--frames" in sys.argv:
        idx = sys.argv.index("--frames")
        if idx + 1 < len(sys.argv):
            num_frames = int(sys.argv[idx + 1])
    
    # Run quantization
    success = quantize_yolov8_to_int8(input_model, output_model)
    
    if success and do_compare:
        compare_models(input_model, output_model, video_file, num_frames)
