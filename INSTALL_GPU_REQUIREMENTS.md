# Hướng dẫn cài đặt dependencies cho GPU

## Lưu ý quan trọng

File `requirements-gpu.txt` này KHÔNG thể cài đặt trực tiếp bằng `pip install -r requirements-gpu.txt` vì:
1. PyTorch với CUDA cần cài từ index URL đặc biệt
2. PaddlePaddle GPU cần cài từ source khác

## Các bước cài đặt

### Bước 1: Cài đặt các dependencies cơ bản

```bash
pip install opencv-python imutils numpy pandas openpyxl requests seaborn lap filterpy Pillow ultralytics tensorflow paddleocr paddlex
```

### Bước 2: Cài đặt PyTorch với CUDA support

**Cho CUDA 12.1 (khuyến nghị, tương thích với CUDA 12.6 driver):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Hoặc cho CUDA 11.8 (nếu CUDA 12.x không hoạt động):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Kiểm tra cài đặt:
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('PyTorch version:', torch.__version__)"
```

### Bước 3: Cài đặt PaddlePaddle với GPU support

Truy cập: https://www.paddlepaddle.org.cn/install/quick

**Cho Windows với CUDA 11.2:**
```bash
python -m pip install paddlepaddle-gpu==2.5.2.post112 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
```

**Cho Windows với CUDA 11.7:**
```bash
python -m pip install paddlepaddle-gpu==2.5.2.post117 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
```

Kiểm tra cài đặt:
```bash
python -c "import paddle; print('PaddlePaddle version:', paddle.__version__); print('CUDA compiled:', paddle.device.is_compiled_with_cuda())"
```

## Script tự động cài đặt (tùy chọn)

Tạo file `install_gpu.bat` (cho Windows):

```batch
@echo off
echo Installing basic dependencies...
pip install opencv-python imutils numpy pandas openpyxl requests seaborn lap filterpy Pillow ultralytics tensorflow paddleocr paddlex

echo.
echo Installing PyTorch with CUDA 12.1...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo Checking PyTorch CUDA installation...
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('PyTorch version:', torch.__version__)"

echo.
echo ========================================
echo Please install PaddlePaddle GPU manually:
echo Visit: https://www.paddlepaddle.org.cn/install/quick
echo ========================================
pause
```

## Kiểm tra sau khi cài đặt

Chạy script kiểm tra:

```python
import torch
import paddle

print("=" * 50)
print("PyTorch:")
print(f"  Version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  GPU name: {torch.cuda.get_device_name(0)}")
    print(f"  GPU count: {torch.cuda.device_count()}")

print("\nPaddlePaddle:")
print(f"  Version: {paddle.__version__}")
print(f"  CUDA compiled: {paddle.device.is_compiled_with_cuda()}")
if paddle.device.is_compiled_with_cuda():
    try:
        paddle.device.set_device('gpu')
        print(f"  GPU available: True")
    except:
        print(f"  GPU available: False (PaddlePaddle-GPU not installed)")
print("=" * 50)
```

## Troubleshooting

### PyTorch vẫn báo "CUDA available: False"

1. Kiểm tra nvidia-smi: `nvidia-smi`
2. Đảm bảo đã cài đúng PyTorch CUDA version (không phải CPU version)
3. Kiểm tra PyTorch version có `+cu121` hoặc `+cu118` trong tên
4. Restart terminal/Python sau khi cài đặt

### PaddlePaddle không nhận GPU

1. Đảm bảo đã cài `paddlepaddle-gpu` (không phải `paddlepaddle`)
2. Kiểm tra CUDA version phù hợp
3. Xem thêm tại: https://www.paddlepaddle.org.cn/documentation/docs/zh/install/pip/windows-pip.html

### Xung đột giữa PyTorch và PaddlePaddle CUDA versions

- PyTorch và PaddlePaddle có thể sử dụng các CUDA toolkit khác nhau
- Điều này thường không gây vấn đề vì chúng sử dụng CUDA runtime khác nhau
- Nếu có lỗi, thử cài cùng CUDA version cho cả hai

