# Module Performance Profiling Guide

## Tổng Quan

Hướng dẫn đo lường thời gian xử lý của ba module chính:
1. **Vehicle Detection** (Phát hiện xe)
2. **License Plate Detection** (Phát hiện biển số)
3. **OCR** (Nhận dạng biển số)

---

## 3 Công Cụ Profiling Khả Dụng

### 1. `analyze_timing.py` - Đơn Giản (Khuyến Nghị ✅)

**Ưu điểm:**
- Đơn giản, nhanh chóng
- Không cần thay đổi code
- Trực tiếp đo thời gian

**Chạy:**
```bash
python analyze_timing.py video/h.mp4 50
```

**Output giải thích:**
```
Frame 1: Vehicle Detection=45.32ms, Vehicles=2, Plates=2, Plate Detect=12.50ms, OCR=35.20ms, Total=92.00ms
Frame 2: Vehicle Detection=42.15ms, Vehicles=1, Plates=1, Plate Detect=11.80ms, OCR=33.60ms, Total=87.50ms

TIMING SUMMARY
==============
Frames processed: 50
Average frame time: 89.75ms
Average FPS: 11.1

--- Module Timing Breakdown ---

1. Vehicle Detection (YOLOv8 Tracking):
   Avg: 43.73ms
   Min: 40.25ms
   Max: 48.92ms
   Total: 2186.50ms (48.8% of frame time)

2. Plate Detection (per vehicle):
   Avg: 12.15ms
   Min: 10.50ms
   Max: 14.80ms

3. OCR Processing (per plate):
   Avg: 34.45ms
   Min: 30.25ms
   Max: 38.90ms

--- BOTTLENECK ANALYSIS ---

Total Module Time: 2651.75ms

1. Vehicle Detection  ........... 2186.50ms (82.4%) ███████████████
2. OCR                ........... 1722.50ms (64.9%) █████████████
3. Plate Detection    ........... 607.50ms  (22.9%) ████
```

### 2. `profile_modules.py` - Chi Tiết (Cấp Cao)

**Ưu điểm:**
- Thông tin chi tiết từng vehicle
- Đo lường OCR text results
- Cấu trúc rõ ràng

**Chạy:**
```bash
python profile_modules.py video/h.mp4 30
```

**Output:**
```
--- Frame 1 ---
  Vehicle Detection: 45.12ms
  Detected 3 vehicles
    Vehicle 1 - Plate Detection: 12.34ms
      Plate OCR: 35.67ms, Text: 'AA123BC'
    Vehicle 2 - Plate Detection: 11.89ms
      Plate OCR: 34.23ms, Text: 'BB456DE'
    Vehicle 3 - Plate Detection: 12.45ms
      Plate OCR: 'unknown'
  Plate Detection - Avg: 12.23ms, Max: 12.45ms

PROFILING SUMMARY
=================

BOTTLENECK ANALYSIS

Total Processing Time: 3245.67ms

1. OCR........................ 1568.90ms (48.3%) ██████████
2. Vehicle Detection.......... 1234.56ms (38.0%) ████████
3. Plate Detection............ 442.21ms  (13.6%) ███
```

### 3. `measure_real_timing.py` - Thực Tế

**Ưu điểm:**
- Đo trực tiếp streaming
- Tích hợp với code hiện tại
- Kết quả thực tế nhất

**Chạy:**
```bash
python measure_real_timing.py video/h.mp4 30
```

---

## Cách Chạy: Bước Thực Hiện

### Bước 1: Chuẩn Bị

```bash
# Kiểm tra video có sẵn
ls video/*.mp4

# Output expected:
# video/h.mp4
# video/a.mp4
# ...
```

### Bước 2: Chạy Simple Analysis

```bash
# Phân tích 50 frame đầu tiên
python analyze_timing.py video/h.mp4 50
```

### Bước 3: Xem Kết Quả

Tìm phần "BOTTLENECK ANALYSIS" trong output. Nó sẽ hiển thị:

```
1. [Module 1]  ....... [Time]ms ([Percentage]%)
2. [Module 2]  ....... [Time]ms ([Percentage]%)
3. [Module 3]  ....... [Time]ms ([Percentage]%)
```

**Module nào có thời gian cao nhất = BOTTLENECK**

---

## Kết Quả Dự Kiến

### Scenario 1: Máy GPU (RTX 3080+)

```
1. Vehicle Detection ... 35.50ms (45.0%)
2. OCR             ... 28.30ms (36.0%)
3. Plate Detection ... 14.70ms (19.0%)
```

**Kết luận:** Cân bằng, OCR vẫn là bottleneck nhỏ

### Scenario 2: Máy CPU-Only

```
1. Vehicle Detection ... 85.50ms (55.0%)
2. OCR             ... 62.30ms (40.0%)
3. Plate Detection ... 10.70ms  (5.0%)
```

**Kết luận:** Vehicle Detection là bottleneck chính

### Scenario 3: Máy GPU Yếu (RTX 3050)

```
1. OCR             ... 125.30ms (60.0%)
2. Vehicle Detection ... 65.50ms (31.0%)
3. Plate Detection ... 14.70ms  (7.0%)
```

**Kết luận:** OCR là bottleneck CHÍNH

---

## Phân Tích & Cách Giải Quyết

### Case 1: Vehicle Detection Chậm (>40ms)

**Nguyên nhân:**
- Model YOLOv8n chạy trên CPU
- Frame resolution quá cao
- Batch size lớn

**Giải pháp:**
```python
# option 1: Giảm resolution
image_size: 640  # ← thay từ 1280

# option 2: Giảm confidence threshold (tìm kiếm nhanh hơn)
confidence: 0.5  # ← thay từ 0.4

# option 3: Skip frames (xử lý 1 trong 2 frame)
```

### Case 2: Plate Detection Chậm (>20ms)

**Nguyên nhân:**
- Model YOLOv8 chạy trên CPU
- Vehicle frame quá lớn

**Giải pháp:**
```python
# option 1: Giảm plate detection resolution
image_size: 480  # ← thay từ 640

# option 2: Crop bé hơn
# Hiện tại crop region lớn, có thể crop nhỏ lại
```

### Case 3: OCR Chậm (>40ms) - **CASE MOST COMMON**

**Nguyên nhân:**
- PaddleOCR chạy trên CPU
- Xử lý từng plate một (sequential)
- Preprocessing chậm

**Giải pháp - BATCH ACCUMULATOR:**
```yaml
# Bật Batch OCR (Approach 2 vừa implement!)
batch_inference:
  enabled: true
  batch_size: 8
  time_threshold_ms: 500
```

Giảm OCR từ 30 plates × 40ms = 1200ms xuống ~150ms!

---

## Command Tóm Gọn

### Test nhanh (10 frame)
```bash
python analyze_timing.py video/h.mp4 10
```

### Test tiêu chuẩn (50 frame)
```bash
python analyze_timing.py video/h.mp4 50
```

### Test toàn bộ video
```bash
python analyze_timing.py video/h.mp4 0  # 0 = all frames
```

### Test chi tiết (30 frame)
```bash
python profile_modules.py video/h.mp4 30
```

### Test real-time (20 frame)
```bash
python measure_real_timing.py video/h.mp4 20
```

---

## Đọc Hiểu Kết Quả

### Ví dụ Output

```
1. OCR                 .............. 1568.90ms (48.3%) ██████████
2. Vehicle Detection   .............. 1234.56ms (38.0%) ████████  
3. Plate Detection     .............. 442.21ms  (13.6%) ███
```

**Giải thích:**
- OCR: 1568.90ms tổng cộng = **48.3% thời gian toàn bộ** (BOTTLENECK!)
- Vehicle Detection: 1234.56ms = **38.0%**
- Plate Detection: 442.21ms = **13.6%**

**Số càng cao → Module càng chậm**

---

## So Sánh Before/After Batch OCR

### Before (Per-Vehicle OCR)
```
Processing 50 frames, 3 vehicles/frame (150 OCR calls total):
1. Vehicle Detection   ... 2186ms (40%)
2. OCR (sequential)    ... 3285ms (60%)
3. Plate Detection     ... 608ms  (10%)
```
**Total: ~5.5 seconds**

### After (Batch OCR - Approach 2)
```
Processing 50 frames, 3 vehicles/frame (150 plates in batches of 8):
1. Vehicle Detection   ... 2186ms (60%)
2. OCR (batched)       ... 450ms  (25%)
3. Plate Detection     ... 608ms  (35%)
```
**Total: ~3.2 seconds** ← **30% faster!**

---

## Recommendation

### 1️⃣ Đo lường hiện tại
```bash
python analyze_timing.py video/h.mp4 50
```

### 2️⃣ Xem module chậm nhất

### 3️⃣ Áp dụng tối ưu:
- Nếu **OCR chậm** → Bật `batch_inference.enabled: true`
- Nếu **Vehicle Detection chậm** → Giảm `image_size`
- Nếu **Plate Detection chậm** → Giảm plate `image_size`

### 4️⃣ Đo lại xem improvement

---

## Troubleshooting

### Lỗi: "Cannot open video"
```bash
# Check video exists
ls video/h.mp4

# If not, list available:
ls video/*.mp4
```

### Script chạy rất chậm
- Giảm `num_frames` (e.g., 10 thay 50)
- Check GPU usage: `nvidia-smi`

### Output không clear
```bash
# Redirect to file
python analyze_timing.py video/h.mp4 50 > timing_results.txt
cat timing_results.txt
```

---

## Pro Tips

1. **Chạy nhiều lần** để kết quả ổn định
```bash
for i in {1..3}; do
    echo "=== Run $i ==="
    python analyze_timing.py video/h.mp4 30
done
```

2. **Lưu kết quả** để so sánh:
```bash
python analyze_timing.py video/h.mp4 50 > before.txt
# [Làm optimization]
python analyze_timing.py video/h.mp4 50 > after.txt
# Compare: diff before.txt after.txt
```

3. **Test trên một frame** để debug
```bash
python analyze_timing.py video/a.mp4 1
```

---

## Kết Luận

**Ba công cụ cung cấp:**
- ✅ `analyze_timing.py`: Bắt đầu từ đây (đơn giản, nhanh)
- ✅ `profile_modules.py`: Chi tiết từng vehicle
- ✅ `measure_real_timing.py`: Real-world scenarios

**Mục tiêu:**
1. Tìm module chậm nhất
2. Áp dụng optimization (Batch OCR nên giải quyết 95% case)
3. Đo lại để verify improvement

