"""
Download and setup GStreamer with NVIDIA NVDEC support on Windows
"""

import os
import subprocess
import sys
import zipfile
import urllib.request
from pathlib import Path


def download_gstreamer_windows():
    """Download pre-built GStreamer for Windows with NVIDIA support"""
    
    print("📥 Downloading GStreamer 1.28.1 for Windows...")
    
    # GStreamer official binary for Windows
    urls = [
        # Official GStreamer project binaries
        "https://gstreamer.freedesktop.org/download/prebuilt-dependencies/bin/windows/x86_64/gstreamer-1.28.1-x86_64-setup.exe",
        # Alternative source (Bento4 / ffmpeg-python)
        "https://github.com/GStreamer/gstreamer/releases/download/1.28.1/GStreamer-1.28.1-devel-x86_64-w64-mingw32.msi",
    ]
    
    # Try first URL
    url = urls[0]
    filename = url.split('/')[-1]
    filepath = Path("C:/tmp") / filename
    
    filepath.parent.mkdir(exist_ok=True, parents=True)
    
    try:
        print(f"Downloading from: {url}")
        urllib.request.urlretrieve(url, filepath, show_progress)
        print(f"✅ Downloaded: {filepath}")
        return filepath
    except Exception as e:
        print(f"❌ Download failed ({e})\n")
        print("📋 MANUAL INSTALLATION:")
        print("  1. Visit: https://gstreamer.freedesktop.org/download/")
        print("  2. Download: GStreamer 1.28.1 (Windows x86_64)")
        print("  3. Install to: C:\\gstreamer")
        print("  4. Add to PATH: C:\\gstreamer\\1.0\\x86_64\\bin")
        print("  5. Add to PATH: C:\\gstreamer\\1.0\\x86_64\\lib\\gstreamer-1.0")
        return None


def setup_gstreamer_path():
    """Setup GStreamer environment variables"""
    
    gstreamer_path = Path("C:\\gstreamer\\1.0\\x86_64")
    
    if gstreamer_path.exists():
        print(f"✅ Found GStreamer at: {gstreamer_path}")
        
        # Add to PATH
        bin_path = str(gstreamer_path / "bin")
        lib_path = str(gstreamer_path / "lib")
        plugin_path = str(gstreamer_path / "lib" / "gstreamer-1.0")
        
        os.environ['PATH'] = f"{bin_path};{lib_path};{os.environ.get('PATH', '')}"
        os.environ['GST_PLUGIN_PATH'] = plugin_path
        
        print(f"✅ Updated GST_PLUGIN_PATH: {plugin_path}")
        return True
    else:
        print(f"❌ GStreamer not found at: {gstreamer_path}")
        return False


def install_gst_nvcodec_conda():
    """Try to install gst-nvcodec through conda-forge"""
    
    print("\n📦 Attempting to install gst-plugins-bad (contains NVIDIA codecs)...")
    
    try:
        # Install gst-plugins-bad which includes NVIDIA codec support
        result = subprocess.run(
            ['conda', 'install', '-c', 'conda-forge', 'gst-plugins-bad', '-y'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            print("✅ gst-plugins-bad installed successfully")
            return True
        else:
            print("⚠️ Conda installation had issues:")
            print(result.stderr[:500])
            return False
    except Exception as e:
        print(f"❌ Installation failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 GStreamer NVIDIA Codec Setup for Windows")
    print("="*70)
    
    # Try conda first
    if install_gst_nvcodec_conda():
        print("\n✅ GStreamer with NVIDIA codecs installed via conda!")
    else:
        print("\n📥 Falling back to manual download...")
        filepath = download_gstreamer_windows()
        
        if filepath and filepath.suffix == '.exe':
            print(f"\n⚠️  Please run the installer manually:")
            print(f"  {filepath}")
            input("\nPress Enter after installation complete...")
    
    # Setup environment
    if setup_gstreamer_path():
        print("\n✅ GStreamer environment variables configured!")
    else:
        print("\n⚠️  GStreamer path not configured - please install manually")
