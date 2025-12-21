# Hướng dẫn cài đặt GPU để tăng tốc xử lý

## Yêu cầu hệ thống

- NVIDIA GPU với CUDA support (ví dụ: GeForce GTX 1650)
- NVIDIA Driver đã được cài đặt
- CUDA Toolkit (thường được cài kèm với PyTorch)

## Kiểm tra GPU

Chạy lệnh sau để kiểm tra GPU:
```bash
nvidia-smi
```

Nếu thấy thông tin về GPU, bạn có thể sử dụng GPU.

## Cài đặt PyTorch với CUDA

### Bước 1: Kiểm tra version CUDA

Chạy lệnh để xem CUDA version:
```bash
nvidia-smi
```

Tìm dòng "CUDA Version" để biết version CUDA được hỗ trợ.

### Bước 2: Cài đặt PyTorch với CUDA

Truy cập https://pytorch.org/get-started/locally/ để chọn version phù hợp.

**Ví dụ cho Windows với CUDA 11.8:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Ví dụ cho Windows với CUDA 12.1:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Bước 3: Kiểm tra PyTorch có nhận GPU

Chạy Python và kiểm tra:
```python
import torch
print(torch.cuda.is_available())  # Phải trả về True
print(torch.cuda.get_device_name(0))  # Hiển thị tên GPU
```

## Cài đặt PaddlePaddle với GPU (cho PaddleOCR)

PaddleOCR cần PaddlePaddle với GPU support:

```bash
# Cho CUDA 11.2
python -m pip install paddlepaddle-gpu==2.5.2.post112 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html

# Hoặc cho CUDA 11.7
python -m pip install paddlepaddle-gpu==2.5.2.post117 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
```

Kiểm tra version: https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/windows-pip.html

## Sử dụng

Sau khi cài đặt, code sẽ tự động phát hiện và sử dụng GPU nếu có sẵn.

Khi chạy, bạn sẽ thấy thông báo:
- `✓ GPU detected: NVIDIA GeForce GTX 1650`
- `✓ YOLO model loaded và chuyển sang GPU`
- `✓ PaddleOCR: Sử dụng GPU`

## Lưu ý

1. Nếu không có GPU hoặc GPU không được nhận diện, hệ thống sẽ tự động chuyển sang CPU.
2. GPU sẽ giúp tăng tốc đáng kể, đặc biệt cho:
   - YOLO vehicle detection
   - YOLOv5 license plate detection
   - PaddleOCR text recognition
3. GTX 1650 có 4GB VRAM, đủ để chạy các model này cùng lúc.

## Troubleshooting

### GPU không được nhận diện

1. Kiểm tra NVIDIA Driver: `nvidia-smi`
2. Kiểm tra PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
3. Đảm bảo đã cài đúng version PyTorch với CUDA

### Lỗi "Out of Memory"

- Giảm `imgsz` trong YOLO (ví dụ: từ 1280 xuống 640)
- Giảm số workers trong ThreadPoolExecutor
- Xử lý ít frame đồng thời

