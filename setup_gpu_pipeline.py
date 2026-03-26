"""
Setup guide for GPU-optimized RTSP streaming pipeline.
"""

import subprocess
import sys
import platform
from VehicleDetectionTracker.logging_utils import log

def check_gstreamer():
    """Check if GStreamer is installed."""
    try:
        result = subprocess.run(["gst-inspect-1.0", "--version"], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            log("[✓] GStreamer is installed")
            log(f"    {result.stdout.strip()}")
            return True
    except:
        pass
    
    log("[✗] GStreamer is NOT installed")
    return False


def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            log(f"[✓] CUDA is available")
            log(f"    Device: {torch.cuda.get_device_name(0)}")
            log(f"    CUDA Version: {torch.version.cuda}")
            return True
    except:
        pass
    
    log("[✗] CUDA is NOT available")
    return False


def check_tensorrt():
    """Check TensorRT availability."""
    try:
        import tensorrt
        log(f"[✓] TensorRT is available")
        log(f"    Version: {tensorrt.__version__}")
        return True
    except:
        pass
    
    log("[✗] TensorRT is NOT installed")
    return False


def install_gstreamer_windows():
    """
    Install GStreamer on Windows.
    Requires manual download from: https://gstreamer.freedesktop.org/download/
    """
    log("\n[INFO] GStreamer Installation Guide (Windows):")
    log("=" * 70)
    log("1. Download GStreamer from:")
    log("   https://gstreamer.freedesktop.org/download/")
    log("")
    log("2. Download these packages:")
    log("   - gstreamer-1.0-msvc-x86_64-VERSION.msi (runtime)")
    log("   - gstreamer-1.0-devel-msvc-x86_64-VERSION.msi (development)")
    log("")
    log("3. Install in order (runtime first, then development)")
    log("")
    log("4. Install Python bindings:")
    log("   pip install PyGObject")
    log("")
    log("5. Verify installation:")
    log("   gst-inspect-1.0 --version")
    log("=" * 70)


def install_gstreamer_linux():
    """Install GStreamer on Linux."""
    log("\n[INFO] GStreamer Installation Guide (Linux):")
    log("=" * 70)
    log("Ubuntu/Debian:")
    log("  sudo apt-get install -y gstreamer1.0 gstreamer1.0-dev gstreamer1.0-plugins-base")
    log("  sudo apt-get install -y gstreamer1.0-plugins-good gstreamer1.0-plugins-bad")
    log("  sudo apt-get install -y python3-gi")
    log("")
    log("Fedora/RHEL:")
    log("  sudo dnf install gstreamer1-devel gstreamer1-plugins-base")
    log("  sudo dnf install gstreamer1-plugins-good gstreamer1-plugins-bad")
    log("=" * 70)


def install_tensorrt():
    """
    Install TensorRT.
    Requires CUDA and cuDNN already installed.
    """
    log("\n[INFO] TensorRT Installation Guide:")
    log("=" * 70)
    log("1. TensorRT requires CUDA 12.x and cuDNN")
    log("")
    log("2. Download TensorRT from:")
    log("   https://developer.nvidia.com/tensorrt/download")
    log("   (Requires NVIDIA Developer account)")
    log("")
    log("3. Extract and setup Python package:")
    log("   pip install /path/to/TensorRT/python/tensorrt-*.whl")
    log("")
    log("4. Verify:")
    log("   python -c \"import tensorrt; print(tensorrt.__version__)\"")
    log("=" * 70)


def show_system_info():
    """Show system information."""
    log("\n[SYSTEM INFO]")
    log(f"Platform: {platform.system()} {platform.release()}")
    log(f"Python: {sys.version}")


def main():
    """Check all dependencies and show setup instructions."""
    log("\n" + "=" * 70)
    log("GPU-Optimized RTSP Streaming - Dependency Check")
    log("=" * 70)
    
    show_system_info()
    
    log("\n[CHECKING DEPENDENCIES]")
    log("-" * 70)
    
    has_gstreamer = check_gstreamer()
    has_cuda = check_cuda()
    has_tensorrt = check_tensorrt()
    
    log("\n[PIPELINE STATUS]")
    log("-" * 70)
    
    if has_gstreamer and has_cuda:
        log("[✓] Full GPU pipeline available: GStreamer + CUDA")
        log("    Can use: RTSP → GStreamer CUVID → GPU inference")
    elif has_cuda:
        log("[⚠] Partial GPU support: CUDA only (no GStreamer)")
        log("    Falls back to: RTSP → OpenCV → GPU inference")
    else:
        log("[✗] No GPU support detected")
        log("    Falls back to: RTSP → CPU inference")
    
    if has_tensorrt:
        log("[✓] TensorRT optimization available")
    else:
        log("[i] TensorRT not installed (optional, for maximum performance)")
    
    # Show installation guides
    system = platform.system()
    log("\n[INSTALLATION GUIDE]")
    log("-" * 70)
    
    if not has_gstreamer:
        if system == "Windows":
            install_gstreamer_windows()
        elif system == "Linux":
            install_gstreamer_linux()
        else:
            log("[INFO] Please install GStreamer from: https://gstreamer.freedesktop.org/")
    
    if not has_tensorrt and has_cuda:
        install_tensorrt()
    
    # Show usage examples
    log("\n[USAGE EXAMPLES]")
    log("-" * 70)
    log("")
    log("1. Basic stream test (auto GPU optimization):")
    log("   python stream_gpu_optimized.py --source video/h.mp4")
    log("")
    log("2. RTSP stream with GStreamer:")
    log("   python stream_gpu_optimized.py --source \"rtsp://ip:554/stream\"")
    log("")
    log("3. With TensorRT acceleration (if available):")
    log("   python stream_gpu_optimized.py --source video/h.mp4 --tensorrt")
    log("")
    log("4. Disable GStreamer (use OpenCV only):")
    log("   python stream_gpu_optimized.py --source video/h.mp4 --no-gstreamer")
    log("")
    log("5. Custom model and confidence:")
    log("   python stream_gpu_optimized.py --source video/h.mp4 \\")
    log("     --model yolov8s.pt --confidence 0.5")
    log("")
    log("6. Process limited frames:")
    log("   python stream_gpu_optimized.py --source video/h.mp4 --max-frames 100")
    log("")
    log("-" * 70)
    log("\n[PIPELINE ARCHITECTURE]")
    log("-" * 70)
    log("Pipeline: RTSP → GStreamer → GPU Decode (CUVID) → YOLO TensorRT → Display")
    log("")
    log("Components:")
    log("  • RTSP Source: Video streaming protocol")
    log("  • GStreamer: Video processing pipeline")
    log("  • CUVID: NVIDIA GPU H.264/H.265 hardware decoder")
    log("  • YOLO: Vehicle detection model")
    log("  • TensorRT: NVIDIA inference optimization (optional)")
    log("")
    log("-" * 70)


if __name__ == "__main__":
    main()
