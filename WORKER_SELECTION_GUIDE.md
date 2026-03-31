# 🎯 HƯỚNG DẪN CHỌN SỐ LƯỢNG WORKER

## 📊 THÔNG SỐ HỆ THỐNG CỦA BẠN

```
Physical cores:  12
Logical cores:   20
RAM:             15.82 GB
GPU:             Chưa được detect (cần activate GPU environment)
```

---

## 🧮 CÔNG THỨC TÍNH OPTIMAL WORKER COUNT

### **Trường hợp 1: GPU Available (Khuyến nghị)**
```
num_workers = (physical_cores / 2) + 1
           = (12 / 2) + 1 
           = 7 workers
```

**Lý do:**
- Model inference chạy trên GPU (không chiếm CPU)
- Worker threads chủ yếu xử lý callback và post-processing (CPU-bound)
- Chia đôi physical cores vì không phải tất cả cores cần cho inference

### **Trường hợp 2: CPU Only (Hiện tại)**
```
num_workers = (physical_cores / 4) + 1
           = (12 / 4) + 1
           = 4 workers
```

**Lý do:**
- Model inference chiếm CPU → có contention cao
- Limit workers để tránh context switching quá nhiều
- Để lại CPU resources cho model inference chính

---

## 💡 KHUYẾN NGHỊ CHO HỆ THỐNG CỦA BẠN

| Tình huống | Số workers | Tại sao |
|-----------|-----------|--------|
| **GPU enabled** | **6-7** | Inference trên GPU, workers xử lý callback |
| **CPU only** | **4-5** | Tránh CPU contention quá cao |
| **Testing/Light load** | **2-3** | Debug, không cần throughput cao |
| **High throughput** | **8** | Nếu RAM đủ (>20GB) |

---

## 🔧 CÁC THAM SỐ LIÊN QUAN

### 1. **BatchAccumulator Parameters**
```python
# plate_utils.py - line ~180
accumulator = BatchAccumulator(
    batch_size=8,                    # 👈 Adjust based on workers
    time_threshold_ms=500,           # Batch size hoặc timeout
    max_batch_wait_ms=2000
)
```

**Mối quan hệ:**
- `num_workers = 4` → `batch_size = 8-10` (workers xử lý nhanh)
- `num_workers = 6` → `batch_size = 12-16` (accumulate nhiều hơn)
- `num_workers = 2` → `batch_size = 4-6` (batch nhỏ, nhanh)

### 2. **Cấp độ Parallelism**
```
num_workers * batch_size = Total parallel tasks

Ví dụ: 4 workers × 8 batch = 32 vehicles xử lý đồng thời
```

---

## ⚙️ CÁC NƠIIJÀ THIẾT LẬP NUM_WORKERS

### 1. **plate_utils.py - InferenceQueue mặc định**
```python
# Line ~156
def get_inference_queue():
    if _inference_queue is None:
        _inference_queue = InferenceQueue(num_workers=6)  # 👈 Thay đổi ở đây
    return _inference_queue
```

**Khuyến nghị:**
- GPU: `num_workers=6` ✓ (tốt)
- CPU: Thay đổi thành `num_workers=4`

### 2. **VehicleDetectionTracker.py - Khởi tạo**
```python
# Tìm dòng: initialize_inference_queue(num_workers=...)
initialize_inference_queue(num_workers=4)  # 👈 Thay đổi ở đây
```

---

## 📈 CÁCH MICRO-TUNE (FINE-TUNING)

### **Bước 1: Chạy test với cấu hình mặc định**
```bash
# Monitor CPU/GPU usage
python test_rtsp_cpu.py
```

**Quan sát:**
- CPU usage > 90%?  → Giảm num_workers
- CPU usage < 50%?  → Tăng num_workers 
- GPU usage < 70%?  → Tăng batch_size hoặc num_workers

### **Bước 2: Công thức Điều chỉnh**
```
Nếu CPU > 90% (bottleneck):
   num_workers = current - 1

Nếu CPU < 50% (under-utilized):
   num_workers = current + 1

Nếu GPU < 70% (under-utilized):
   batch_size = current + 4
```

### **Bước 3: Kiểm tra logs**
```
[INFERENCE_QUEUE] task executed
[BATCH_ACCUM] Batch trigger: size=N or time=...
```

**Tìm:**
- `Batch trigger` thường xuyên → `batch_size` phù hợp ✓
- Không có `Batch trigger` → `batch_size` quá lớn, `time_threshold` có tác dụng

---

## 🎯 CÓ NHƯ VẬY ACTIVATE GPU ENVIRONMENT?

```bash
# Kiểm tra environment hiện tại
conda env list

# Activate GPU environment nếu có
conda activate ocr_env_py312_gpu

# Kiểm tra GPU lại
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
```

**Nếu GPU available:**
- Thay `num_workers=4` → `num_workers=6`
- Performance sẽ tốt hơn đáng kể ⚡

---

## 📌 TÓM TẮT CHO HỆ THỐNG CỦA BẠN

| Thành phố | Giá trị | Ghi chú |
|----------|--------|--------|
| **CPU Cores** | 12 physical | Đủ good |
| **RAM** | 15.82 GB | Đủ cho threading |
| **Khuyến nghị num_workers** | **4-6** | Tùy GPU |
| **Batch Size** | **8-12** | Tương ứng workers |
| **Max Throughput** | ~480-960 vehicles/batch | Workers × Batch |

---

## 🚀 SỐ WORKER NHANH QUICK REFERENCE

| Config | Workers | Batch | Thích hợp khi |
|--------|---------|-------|--------------|
| **Lightweight** | 2 | 4 | Testing, dev |
| **Balanced (You)** | 4 | 8 | CPU, production |
| **Optimized (GPU)** | 6 | 12 | GPU available |
| **Aggressive** | 8 | 16 | High-end machine |

---

**💬 HỌC TỐI:** 
- Bắt đầu với khuyến nghị
- Monitor CPU/GPU usage
- Micro-tune dần dần
- Mục tiêu: CPU 70-85%, GPU 85-95%
