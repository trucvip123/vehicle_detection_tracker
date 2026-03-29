"""
One-command setup for Docker GPU environment
Run this to automatically:
1. Check Docker/NVIDIA installation
2. Build Docker image
3. Run container with GPU support
"""

import subprocess
import sys
import platform
import os


def run_command(cmd, description):
    """Run command and report results"""
    print(f"\n{'='*70}")
    print(f"📋 {description}")
    print(f"{'='*70}")
    print(f"Command: {cmd}\n")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
        if result.returncode == 0:
            print(f"✅ Success")
            return True
        else:
            print(f"❌ Failed")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def check_docker():
    """Check if Docker is installed"""
    print("\n🔍 Checking Docker installation...")
    try:
        result = subprocess.run("docker --version", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Docker found: {result.stdout.strip()}")
            return True
        else:
            print("❌ Docker not found")
            print("📥 Download from: https://www.docker.com/products/docker-desktop")
            return False
    except:
        print("❌ Docker command failed")
        return False


def check_nvidia_runtime():
    """Check NVIDIA Container Toolkit"""
    print("\n🔍 Checking NVIDIA Container Runtime...")
    try:
        result = subprocess.run(
            "docker run --rm --gpus all nvidia/cuda:12.6.0-runtime-ubuntu22.04 nvidia-smi",
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            print("✅ NVIDIA Container Runtime available")
            return True
        else:
            print("❌ NVIDIA Container Runtime not configured")
            print("📋 Follow DOCKER_GPU_SETUP.md Step 2")
            return False
    except subprocess.TimeoutExpired:
        print("⏱️  Timeout (Docker might be pulling image)")
        return False
    except:
        print("❌ NVIDIA runtime check failed")
        return False


def build_image():
    """Build Docker image"""
    if not check_docker():
        print("\n❌ Cannot build: Docker not installed")
        return False
    
    cmd = "docker build -f Dockerfile.gpu -t vehicle-detector:gpu-latest ."
    return run_command(cmd, "Building Docker Image")


def run_container_test():
    """Run GPU test in container"""
    cmd = 'docker run --rm --gpus all -v %cd%:/app vehicle-detector:gpu-latest python quick_rtsp_test.py'
    return run_command(cmd, "Running GPU Test in Container")


def run_container_interactive():
    """Run interactive container"""
    cmd = 'docker run -it --gpus all -v %cd%:/app vehicle-detector:gpu-latest /bin/bash'
    return run_command(cmd, "Running Interactive Container (type 'exit' to quit)")


def main():
    """Main setup flow"""
    print("\n" + "="*70)
    print("🚀 Docker GPU Setup for Vehicle Detection")
    print("="*70)
    
    # Check prerequisites
    print("\n1️⃣  CHECKING PREREQUISITES...")
    
    if not check_docker():
        print("\n❌ Setup cannot continue without Docker")
        print("\n📋 Installation steps:")
        print("  1. Download Docker Desktop from: https://www.docker.com/products/docker-desktop")
        print("  2. Run the installer")
        print("  3. Restart your computer")
        print("  4. Open PowerShell and run this script again")
        return
    
    if not check_nvidia_runtime():
        print("\n⚠️  NVIDIA Container Runtime not ready")
        print("\n📋 Setup steps:")
        print("  1. Follow: DOCKER_GPU_SETUP.md - Step 2")
        print("  2. Or enable WSL2 integration: Docker Settings → Resources → WSL Integration")
        print("  3. Restart Docker Desktop")
        response = input("\n⏸️  Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Build image
    print("\n2️⃣  BUILDING DOCKER IMAGE...")
    if not build_image():
        print("\n❌ Image build failed")
        return
    
    # Test
    print("\n3️⃣  TESTING GPU IN CONTAINER...")
    print("\nOptions:")
    print("  1 - Quick RTSP test (recommended)")
    print("  2 - Interactive bash shell")
    print("  3 - Skip testing")
    
    choice = input("\nSelect (1/2/3): ").strip()
    
    if choice == "1":
        run_container_test()
    elif choice == "2":
        run_container_interactive()
    
    print("\n" + "="*70)
    print("✅ DOCKER GPU SETUP COMPLETE")
    print("="*70)
    print("\n📋 Next Steps:")
    print("  • Run detection: docker run --gpus all -v %cd%:/app vehicle-detector:gpu-latest python quick_rtsp_test.py")
    print("  • Interactive: docker run -it --gpus all -v %cd%:/app vehicle-detector:gpu-latest /bin/bash")
    print("  • Compose: docker-compose -f docker-compose.gpu.yml up")
    print("\n📖 Full guide: DOCKER_GPU_SETUP.md")


if __name__ == "__main__":
    main()
