# Hướng dẫn cài đặt PyTorch với GPU support

## Tình trạng hiện tại

- ✅ GPU: NVIDIA GeForce GTX 1650 (4GB VRAM)
- ✅ CUDA Driver: 12.6
- ❌ PyTorch: `2.9.1+cpu` (chỉ hỗ trợ CPU, không hỗ trợ GPU)

## Giải pháp

Bạn cần cài đặt PyTorch với CUDA support. Với CUDA Driver 12.6, bạn có thể sử dụng PyTorch được build cho CUDA 12.1 hoặc 12.4 (tương thích ngược).

### Bước 1: Gỡ cài đặt PyTorch hiện tại (tùy chọn)

```bash
pip uninstall torch torchvision torchaudio
```

### Bước 2: Cài đặt PyTorch với CUDA 12.1 (Khuyến nghị)

**Cho Windows:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Hoặc CUDA 12.4 (nếu có sẵn):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Bước 3: Kiểm tra cài đặt

Chạy lệnh sau để kiểm tra:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('PyTorch version:', torch.__version__); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

Bạn sẽ thấy:
- `CUDA available: True`
- `PyTorch version: 2.x.x+cu121` (hoặc cu124)
- `CUDA version: 12.1` (hoặc 12.4)
- `GPU name: NVIDIA GeForce GTX 1650`

### Bước 4: Kiểm tra lại code

Sau khi cài đặt, chạy lại code:

```bash
python StreamingCameraTest.py
```

Bạn sẽ thấy thông báo:
- `✓ GPU detected: NVIDIA GeForce GTX 1650`
- `✓ YOLO model loaded và chuyển sang GPU: NVIDIA GeForce GTX 1650`
- `✓ License plate detector loaded và chuyển sang GPU`

## Lưu ý

1. **CuDNN**: PyTorch CUDA wheels đã bao gồm CuDNN, không cần cài riêng
2. **Kiểm tra version**: Đảm bảo PyTorch version có `+cu121` hoặc `+cu124` trong tên, không phải `+cpu`
3. **Tài nguyên**: Xem thêm tại https://pytorch.org/get-started/locally/

## Troubleshooting

### Nếu vẫn báo "CUDA available: False"

1. Kiểm tra lại nvidia-smi: `nvidia-smi`
2. Đảm bảo đã cài đúng PyTorch CUDA version
3. Restart Python/IDE sau khi cài đặt

### Nếu gặp lỗi khi import torch

1. Kiểm tra Python version (khuyến nghị Python 3.8-3.11)
2. Thử cài lại với pip: `pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121`


