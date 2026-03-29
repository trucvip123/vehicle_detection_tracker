# Docker GPU Setup Guide for Vehicle Detection

## 🎯 Overview

This guide will set up a **production-ready Docker environment** with full NVIDIA GPU support for real-time vehicle detection with RTSP streaming.

---

## 📋 Prerequisites

### **System Requirements**
- ✅ Windows 10/11 Pro, Enterprise, or Education (WSL2 support)
- ✅ NVIDIA GPU (confirmed: GTX 1650)
- ✅ NVIDIA Driver (confirmed: 560.94)
- ❌ Docker Desktop installed
- ❌ NVIDIA Container Toolkit installed

### **Current Status**
```
✅ NVIDIA Driver: 560.94
✅ NVIDIA GPU: GTX 1650
✅ CUDA: 12.6
❌ Docker: NOT YET INSTALLED
❌ NVIDIA Container Runtime: NOT YET INSTALLED
```

---

## 🚀 Installation Steps

### **Step 1: Install Docker Desktop**

**Windows:**

1. **Download Docker Desktop:**
   https://www.docker.com/products/docker-desktop

2. **Install:**
   - Run installer as Administrator
   - Enable "WSL 2" during installation
   - Restart computer

3. **Verify installation:**
   ```powershell
   docker --version
   docker run hello-world
   ```

Expected output:
```
Docker version 26.x.x, build xxxxxxx
Hello from Docker!
```

---

### **Step 2: Install NVIDIA Container Toolkit**

**Windows with Docker Desktop:**

1. **Enable WSL 2 integration in Docker:**
   - Open Docker Desktop
   - Settings → Resources → WSL Integration
   - Enable "Ubuntu" (or your WSL distribution)
   - Click "Apply & Restart"

2. **Inside WSL2 (Ubuntu terminal):**
   ```bash
   # Install NVIDIA Container Toolkit
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
     sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   
   sudo apt-get update
   sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

3. **Verify NVIDIA runtime:**
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.6.0-runtime-ubuntu22.04 nvidia-smi
   ```

   Should show your GPU details.

---

### **Step 3: Build Docker Image**

**From PowerShell in project directory:**

```powershell
# Navigate to project
cd D:\TrucNV\vehicle_detection_tracker

# Build image with GPU support
docker build -f Dockerfile.gpu -t vehicle-detector:gpu-latest .
```

**Build time:** ~5-10 minutes

**After build, verify:**
```powershell
docker images | findstr vehicle-detector
```

---

### **Step 4: Run Docker Container with GPU**

#### **Option A: Single Command**

```powershell
docker run --rm --gpus all `
  -it `
  -v ${PWD}:/app `
  -v ${PWD}/logs:/app/logs `
  -v ${PWD}/model:/app/model `
  vehicle-detector:gpu-latest `
  python quick_rtsp_test.py
```

#### **Option B: Using Docker Compose (RECOMMENDED)**

```powershell
# Run main GPU detector
docker-compose -f docker-compose.gpu.yml up gpu-vehicle-detector

# Run Jupyter notebook (optional)
docker-compose -f docker-compose.gpu.yml up jupyter-gpu
```

Stop with:
```powershell
docker-compose -f docker-compose.gpu.yml down
```

---

## 🧪 Testing GPU Support

### **Inside Container - Verify GPU Access**

```bash
# Check NVIDIA GPU
nvidia-smi

# Check CUDA libraries
ldd /usr/local/cuda/lib64/libcudart.so.12

# Test CuPy GPU computing
python -c "import cupy; print(cupy.cuda.Device())"

# Test with RTSP pipeline
python rtsp_gpu_pipeline.py

# Test quick RTSP test
python quick_rtsp_test.py
```

---

## 📊 Expected Results

After running the container:

```
✅ NVIDIA Container Runtime: Active
✅ GPU Access: Available
✅ CUDA Toolkit: 12.6
✅ CuPy: Working
✅ RTSP Streaming: 25+ FPS
✅ GPU Processing: Active
```

---

## 🎯 Running Vehicle Detection

### **Option 1: RTSP Real-time Detection**

```powershell
docker run --rm --gpus all `
  -it `
  -v ${PWD}:/app `
  vehicle-detector:gpu-latest `
  python rtsp_gpu_pipeline.py
```

### **Option 2: Interactive Bash in Container**

```powershell
docker run --rm --gpus all `
  -it `
  -v ${PWD}:/app `
  vehicle-detector:gpu-latest `
  /bin/bash
```

Then inside container:
```bash
# Run detection
python quick_rtsp_test.py

# Or your main detection script
cd /app
python VehicleDetectionPipeline_Integrated.py
```

### **Option 3: Jupyter Notebook**

```powershell
docker-compose -f docker-compose.gpu.yml up jupyter-gpu
```

Then access:
```
http://localhost:8888
```

---

## 🔧 Dockerfile Customization

### **To modify what runs by default:**

Edit `Dockerfile.gpu`, change last line:

```dockerfile
# Current
CMD ["python", "quick_rtsp_test.py"]

# Change to run your main script
CMD ["python", "VehicleDetectionPipeline_Integrated.py"]
```

Then rebuild:
```powershell
docker build -f Dockerfile.gpu -t vehicle-detector:gpu-latest .
```

---

## 📦 Docker Compose Usage

### **Basic Commands:**

```powershell
# Start all services
docker-compose -f docker-compose.gpu.yml up

# Start in background
docker-compose -f docker-compose.gpu.yml up -d

# View logs
docker-compose -f docker-compose.gpu.yml logs -f gpu-vehicle-detector

# Stop all services
docker-compose -f docker-compose.gpu.yml down

# Rebuild image
docker-compose -f docker-compose.gpu.yml build --no-cache

# Run specific command
docker-compose -f docker-compose.gpu.yml run gpu-vehicle-detector python quick_rtsp_test.py
```

---

## 🐛 Troubleshooting

### **Error: "docker: command not found"**
- **Solution:** Docker Desktop not installed or not in PATH
- Reinstall Docker Desktop and restart terminal

### **Error: "unknown flag: --gpus"**
- **Solution:** NVIDIA Container Toolkit not installed
- Follow Step 2 above
- Restart Docker daemon: `docker daemon restart`

### **Error: "no NVIDIA GPUs detected"**
- **Solution:** Check GPU access
- Run: `docker run --rm --gpus all nvidia/cuda:12.6.0-base nvidia-smi`
- If fails, NVIDIA driver or toolkit issue

### **Slow Performance in Container**
- Check GPU utilization: `nvidia-smi` inside container
- Check available GPU memory: `nvidia-smi`
- Reduce batch size if processing

### **RTSP Connection Failed**
- **Inside container**, test connectivity:
  ```bash
  ping aicamera.serveminecraft.net
  curl -v rtsp://admin:MOVYKV@aicamera.serveminecraft.net:554/Streaming/Channels/101
  ```

---

## 📊 Performance Comparison

### **Native vs Docker GPU**

| Metric | Native | Docker |
|--------|--------|--------|
| Startup | ~2s | ~5s |
| GPU Access | Direct | Via Runtime |
| FPS | 25-30 | 25-30 (same) |
| Memory | Shared | Isolated (4GB limit) |
| Portability | Windows only | Any system with Docker |

---

## 🚀 Scaling to Multiple GPUs

If you add more GPUs later:

```powershell
# Use all GPUs
docker run --gpus all ...

# Use specific GPU
docker run --gpus '"device=0"' ...

# Use GPUs 0 and 1
docker run --gpus '"device=0,1"' ...
```

---

## 📝 Quick Reference

```powershell
# Build
docker build -f Dockerfile.gpu -t vehicle-detector:gpu-latest .

# Test GPU
docker run --rm --gpus all vehicle-detector:gpu-latest nvidia-smi

# Run RTSP test
docker run --rm --gpus all -v ${PWD}:/app vehicle-detector:gpu-latest python quick_rtsp_test.py

# Interactive shell
docker run -it --gpus all -v ${PWD}:/app vehicle-detector:gpu-latest /bin/bash

# View all images
docker images

# Remove image
docker rmi vehicle-detector:gpu-latest

# View containers
docker ps -a
```

---

## ✅ Verification Checklist

After setup, verify:

- [ ] Docker Desktop installed and running
- [ ] `docker --version` works
- [ ] NVIDIA Container Toolkit installed
- [ ] `docker run --rm --gpus all nvidia/cuda:12.6.0-runtime-ubuntu22.04 nvidia-smi` shows GPU
- [ ] Image built: `docker images | findstr vehicle-detector`
- [ ] Container runs: `docker run --rm --gpus all vehicle-detector:gpu-latest python quick_rtsp_test.py`
- [ ] GPU detected in container
- [ ] RTSP test shows 25+ FPS
- [ ] CuPy works: `python -c "import cupy; print(cupy.cuda.Device())"`

---

## 🎯 Next Steps

1. **Install Docker Desktop** - 10 minutes
2. **Install NVIDIA Container Toolkit** - 5 minutes  
3. **Build Docker image** - 10 minutes
4. **Run container with GPU** - 2 minutes
5. **Test RTSP streaming** - 1 minute

**Total time: ~30 minutes**

---

## 📞 Additional Resources

- **Docker Docs:** https://docs.docker.com/
- **NVIDIA Container Toolkit:** https://github.com/NVIDIA/nvidia-docker
- **NVIDIA CUDA Docker:** https://hub.docker.com/r/nvidia/cuda
- **Docker Compose Reference:** https://docs.docker.com/compose/compose-file/

---

**Status:** Ready for Docker GPU setup  
**Last Updated:** 2026-03-26  
**NVIDIA Driver:** 560.94 | CUDA: 12.6 | GPU: GTX 1650
