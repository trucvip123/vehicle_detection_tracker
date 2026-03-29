# Docker GPU Quick Reference

## 🚀 Quick Start (Copy-Paste Commands)

### **1. Build Docker Image**
```powershell
cd D:\TrucNV\vehicle_detection_tracker
docker build -f Dockerfile.gpu -t vehicle-detector:gpu-latest .
```

### **2. Test GPU Access**
```powershell
docker run --rm --gpus all vehicle-detector:gpu-latest nvidia-smi
```

### **3. Run RTSP Test (Recommended First Test)**
```powershell
docker run --rm --gpus all `
  -v ${PWD}:/app `
  vehicle-detector:gpu-latest `
  python quick_rtsp_test.py
```

### **4. Interactive Shell in Container**
```powershell
docker run -it --gpus all `
  -v ${PWD}:/app `
  vehicle-detector:gpu-latest `
  /bin/bash
```

Then inside container:
```bash
# Test RTSP
python quick_rtsp_test.py

# Run GPU pipeline
python rtsp_gpu_pipeline.py

# Check GPU
nvidia-smi

# Test CuPy
python -c "import cupy; print('GPU:', cupy.cuda.Device())"

# Exit container
exit
```

### **5. Using Docker Compose**
```powershell
# Start services
docker-compose -f docker-compose.gpu.yml up

# Start in background
docker-compose -f docker-compose.gpu.yml up -d

# Stop services
docker-compose -f docker-compose.gpu.yml down

# View logs
docker-compose -f docker-compose.gpu.yml logs -f gpu-vehicle-detector
```

---

## 📊 Common Commands

### **Image Management**
```powershell
# List images
docker images

# List images with filter
docker images vehicle-detector

# Remove image
docker rmi vehicle-detector:gpu-latest

# View image history
docker history vehicle-detector:gpu-latest

# Rebuild without cache
docker build --no-cache -f Dockerfile.gpu -t vehicle-detector:gpu-latest .
```

### **Container Management**
```powershell
# List running containers
docker ps

# List all containers
docker ps -a

# Stop container
docker stop <container_id>

# Remove container
docker rm <container_id>

# View container logs
docker logs <container_id>

# Follow logs
docker logs -f <container_id>

# Container stats
docker stats

# Inspect container
docker inspect <container_id>
```

### **GPU Operations**
```powershell
# Run with specific GPU
docker run --gpus '"device=0"' ...

# Run with all GPUs
docker run --gpus all ...

# Run with 2 GPUs
docker run --gpus '"device=0,1"' ...

# View GPU inside container
docker run --rm --gpus all nvidia/cuda:12.6.0-runtime-ubuntu22.04 nvidia-smi
```

---

## 🔍 Troubleshooting Commands

### **Check Docker Status**
```powershell
# Docker info
docker info

# Test Docker
docker run hello-world

# Check Docker version
docker version
```

### **Check NVIDIA Runtime**
```powershell
# Test NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.6.0-runtime-ubuntu22.04 nvidia-smi

# If fails, check Docker config
docker info | findstr nvidia

# Restart Docker daemon
"Restart Docker Desktop from taskbar"
```

### **Debug Container**
```powershell
# Run with verbose output
docker run -v /var/lib/docker:/var/lib/docker:ro vehicle-detector:gpu-latest

# Check permissions
docker exec <container_id> id

# Test network
docker run -it vehicle-detector:gpu-latest /bin/bash
# Inside: ping google.com
```

---

## 📈 Performance Monitoring

### **Inside Container**
```bash
# GPU usage
nvidia-smi

# Continuous monitoring
watch -n 1 nvidia-smi

# Process-level GPU usage
nvidia-smi pmon

# Check GPU memory
nvidia-smi --query-gpu=memory.free,memory.used --format=csv

# Python GPU check
python -c "import cupy; 
print('GPU:', cupy.cuda.Device())
print('GPU Memory:', cupy.get_default_memory_pool().get_limit() / 1e9, 'GB')"
```

### **From Host (PowerShell)**
```powershell
# CPU/Memory usage
docker stats

# Specific container
docker stats <container_id>

# View Docker disk usage
docker system df
```

---

## 🎯 Running Detection Scripts

### **RTSP GPU Pipeline**
```powershell
docker run --rm --gpus all `
  -v ${PWD}:/app `
  -e RTSP_URL="rtsp://admin:MOVYKV@aicamera.serveminecraft.net:554/Streaming/Channels/101" `
  vehicle-detector:gpu-latest `
  python rtsp_gpu_pipeline.py
```

### **Custom Python Script**
```powershell
docker run --rm --gpus all `
  -v ${PWD}:/app `
  vehicle-detector:gpu-latest `
  python <your_script.py>
```

### **With Output Mount**
```powershell
docker run --rm --gpus all `
  -v ${PWD}:/app `
  -v ${PWD}/output:/app/output `
  -v ${PWD}/logs:/app/logs `
  vehicle-detector:gpu-latest `
  python detection_script.py
```

---

## 🔐 Production Deployment

### **Run as Background Service**
```powershell
# Start container in background
docker run -d `
  --name vehicle-detector `
  --gpus all `
  -v ${PWD}:/app `
  -v ${PWD}/logs:/app/logs `
  -restart unless-stopped `
  vehicle-detector:gpu-latest `
  python rtsp_gpu_pipeline.py
```

### **Monitor Background Container**
```powershell
# View logs
docker logs vehicle-detector

# Follow logs
docker logs -f vehicle-detector

# Container stats
docker stats vehicle-detector

# Stop container
docker stop vehicle-detector

# Restart container
docker start vehicle-detector

# Remove container
docker rm vehicle-detector
```

---

## 💾 Clean Up

### **Free Disk Space**
```powershell
# Remove unused images
docker image prune

# Remove unused containers
docker container prune

# Remove dangling volumes
docker volume prune

# Complete cleanup
docker system prune --all --volumes
```

### **Remove Specific**
```powershell
# Remove all vehicle-detector images
docker rmi $(docker images -q vehicle-detector)

# Remove all stopped containers
docker rm $(docker ps -aq)

# Remove containers by name
docker rm vehicle-detector
```

---

## 📝 Useful Aliases (PowerShell Profile)

Add to PowerShell profile (`$PROFILE`):

```powershell
# GPU Quick Commands
Set-Alias ddrun 'docker run --rm --gpus all -v ${PWD}:/app vehicle-detector:gpu-latest'
Set-Alias ddbuild 'docker build -f Dockerfile.gpu -t vehicle-detector:gpu-latest .'
Set-Alias ddtest 'docker run --rm --gpus all vehicle-detector:gpu-latest python quick_rtsp_test.py'
Set-Alias ddbash 'docker run -it --gpus all -v ${PWD}:/app vehicle-detector:gpu-latest /bin/bash'

# Docker Compose
Set-Alias dcup 'docker-compose -f docker-compose.gpu.yml up'
Set-Alias dcdown 'docker-compose -f docker-compose.gpu.yml down'
Set-Alias dclogs 'docker-compose -f docker-compose.gpu.yml logs -f'
```

Then use:
```powershell
ddbuild          # Build image
ddtest           # Run test
ddbash          # Interactive shell
dcup            # Start with compose
dcdown          # Stop with compose
```

---

## 🎬 Complete Workflow

```powershell
# 1. Navigate to project
cd D:\TrucNV\vehicle_detection_tracker

# 2. Build image (first time only)
docker build -f Dockerfile.gpu -t vehicle-detector:gpu-latest .

# 3. Test GPU access
docker run --rm --gpus all vehicle-detector:gpu-latest nvidia-smi

# 4. Run RTSP test
docker run --rm --gpus all -v ${PWD}:/app vehicle-detector:gpu-latest python quick_rtsp_test.py

# 5. If successful, run full detection
docker run --rm --gpus all -v ${PWD}:/app -v ${PWD}/logs:/app/logs vehicle-detector:gpu-latest python detection_script.py

# 6. For continuous operation
docker-compose -f docker-compose.gpu.yml up -d
docker-compose -f docker-compose.gpu.yml logs -f
```

---

## ⚡ Performance Tips

1. **Mount volumes efficiently:**
   ```powershell
   # Good - just mount needed directories
   -v ${PWD}/model:/app/model -v ${PWD}/logs:/app/logs
   
   # Avoid - mounting entire project if not needed
   -v ${PWD}:/app
   ```

2. **Set memory limits:**
   ```powershell
   docker run --memory 4g --gpus all ...
   ```

3. **Resource limits in compose:**
   - Edit `docker-compose.gpu.yml`
   - Adjust `mem_limit` and `deploy.resources`

4. **Batch processing:**
   - Run multiple containers in parallel
   - Use device selection to distribute GPUs

---

## 📞 Help & Support

```powershell
# Docker help
docker --help
docker run --help
docker exec --help

# Docker logs
docker logs <container_id>

# Inspect configuration
docker inspect <container_id>

# Debug shell
docker run -it vehicle-detector:gpu-latest /bin/bash
```

---

**Last Updated:** 2026-03-26  
**Docker Version:** 26.0+  
**NVIDIA Container Toolkit:** Latest  
**CUDA:** 12.6  
**GPU:** GTX 1650+
