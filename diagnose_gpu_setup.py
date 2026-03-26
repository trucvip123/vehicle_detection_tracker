"""
NVIDIA GPU Video Decoding Setup for RTSP Streams
Detailed instructions to enable NVIDIA NVDEC for hardware-accelerated H.264/H.265 decoding
"""

import subprocess
import sys
import json


def check_nvidia_driver():
    """Check if NVIDIA driver is installed"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            driver_version = result.stdout.strip()
            print(f"✅ NVIDIA Driver: {driver_version}")
            return True
        else:
            print("❌ NVIDIA SMI not found")
            return False
    except Exception as e:
        print(f"❌ Driver check failed: {e}")
        return False


def check_cuda_toolkit():
    """Check if CUDA toolkit is available"""
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✅ CUDA Toolkit: Available")
            return True
    except:
        pass
    print("⚠️ CUDA Toolkit: Not in PATH (may still work if installed)")
    return False


def check_gstreamer():
    """Check GStreamer installation"""
    try:
        result = subprocess.run(['gst-inspect-1.0', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✅ GStreamer: {result.stdout.strip()}")
            return True
    except:
        pass
    print("❌ GStreamer: Not installed")
    return False


def check_nvdec_plugin():
    """Check if NVIDIA NVDEC GStreamer plugin is available"""
    try:
        result = subprocess.run(['gst-inspect-1.0', 'nvh264dec'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✅ GStreamernvh264dec plugin available")
            return True
        else:
            print("❌ nvh264dec plugin NOT found")
            return False
    except:
        print("❌ gst-inspect-1.0 failed")
        return False


def install_nvidia_codecs_windows():
    """Installation instructions for Windows"""
    print("\n" + "="*70)
    print("NVIDIA NVDEC INSTALLATION FOR WINDOWS")
    print("="*70)
    
    print("""
📋 STEP 1: Install NVIDIA GStreamer Plugin (gst-nvcodec)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Option A: Download Pre-built Binary (EASIEST)
  1. Go to: https://github.com/GStreamer/gst-plugins-bad/releases
  2. Download: gst-plugins-bad-1.24.x-MSVC-x86_64.zip (or newer)
  3. Extract to: C:\\gstreamer\\1.0\\x86_64\\lib\\gstreamer-1.0\\
  4. Add to PATH: C:\\gstreamer\\1.0\\x86_64\\bin;C:\\gstreamer\\1.0\\x86_64\\lib

Option B: Build from Source (Advanced)
  1. Clone: git clone https://github.com/GStreamer/gst-plugins-bad.git
  2. Install: meson build --prefix=C:\\gstreamer
  3. Run: meson compile -C build && meson install -C build

📋 STEP 2: Verify Installation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Run: gst-inspect-1.0 nvh264dec
Expected: Plugin details for nvh264dec (if not installed, you'll see 'No such element')

📋 STEP  3: Verify NVIDIA GPU is detected by GStreamer
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Run: gst-inspect-1.0 | findstr nvdec

If you see: "nvh264dec  mux      H264 NVIDIA Decoder" ✅ GPU support is ready!
    """)


def print_gstreamer_pipelines():
    """Print example GStreamer pipelines"""
    print("\n" + "="*70)
    print("GSTREAMER PIPELINES FOR RTSP GPU DECODING")
    print("="*70)
    
    print("""
💎 GPU-Accelerated Pipeline (requires nvh264dec plugin):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pipeline = (
    f"rtspsrc location={rtsp_url} protocols=tcp latency=0 ! "
    f"rtph264depay ! h264parse ! nvh264dec ! "
    f"videoconvert ! appsink drop=true max-buffers=1 sync=false"
)
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

⭐ Benefits: GPU decoding → Lower CPU usage, Higher FPS

⚙️ Software Fallback Pipeline (always works):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pipeline = (
    f"rtspsrc location={rtsp_url} protocols=tcp latency=0 ! "
    f"rtph264depay ! h264parse ! avdec_h264 ! "
    f"videoconvert ! appsink drop=true max-buffers=1 sync=false"
)
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

⚠️ Trade-off: Higher CPU usage but more compatible
    """)


if __name__ == "__main__":
    print("\n🔍 NVIDIA GPU Video Decoding Diagnostic Tool")
    print("="*70)
    
    # Run checks
    has_driver = check_nvidia_driver()
    has_cuda = check_cuda_toolkit()
    has_gstreamer = check_gstreamer()
    has_nvdec = check_nvdec_plugin() if has_gstreamer else False
    
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    
    if has_driver and has_gstreamer and has_nvdec:
        print("""
✅ GPU VIDEO DECODING READY!

Your system has everything needed for NVIDIA GPU-accelerated H.264 video decoding.
You can use the GPU pipeline in verify_gstreamer.py with nvh264dec.
        """)
    elif has_driver and has_gstreamer:
        print("""
⚠️ PARTIAL SETUP

You have:
  ✅ NVIDIA Driver
  ✅ GStreamer
  ❌ NVIDIA NVDEC Plugin

NEXT STEPS:
  1. Install gst-plugins-bad with NVIDIA codec support
  2. See installation instructions above
  3. Then use nvh264dec pipeline for GPU acceleration
        """)
        install_nvidia_codecs_windows()
    else:
        print("""
❌ MISSING COMPONENTS

What's needed:
  {} NVIDIA Driver (v.460+)
  {} GStreamer + gst-plugins-bad
  {} NVIDIA NVDEC plugin

Please install missing components and run this diagnostic again.
        """.format(
            "✅" if has_driver else "❌",
            "✅" if has_gstreamer else "❌",
            "✅" if has_nvdec else "❌"
        ))
    
    print_gstreamer_pipelines()
