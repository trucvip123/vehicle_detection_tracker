@echo off
chcp 65001 >nul
echo ========================================
echo Cài đặt dependencies cho GPU version
echo ========================================
echo.

echo [1/3] Cài đặt các dependencies cơ bản...
pip install opencv-python imutils numpy pandas openpyxl requests seaborn lap filterpy Pillow ultralytics tensorflow paddleocr paddlex
if errorlevel 1 (
    echo ❌ Lỗi khi cài đặt dependencies cơ bản
    pause
    exit /b 1
)
echo ✓ Đã cài đặt dependencies cơ bản
echo.

echo [2/3] Cài đặt PyTorch với CUDA 12.1 support...
echo Lưu ý: Nếu bạn có CUDA 11.8, hãy dừng script này và chạy:
echo pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo ❌ Lỗi khi cài đặt PyTorch CUDA
    pause
    exit /b 1
)
echo ✓ Đã cài đặt PyTorch với CUDA
echo.

echo [3/3] Kiểm tra PyTorch CUDA...
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
echo.

echo ========================================
echo ⚠ QUAN TRỌNG: PaddlePaddle GPU
echo ========================================
echo Bạn cần cài đặt PaddlePaddle GPU thủ công:
echo.
echo Truy cập: https://www.paddlepaddle.org.cn/install/quick
echo.
echo Hoặc chạy lệnh sau (cho CUDA 11.7):
echo python -m pip install paddlepaddle-gpu==2.5.2.post117 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
echo.
echo Hoặc cho CUDA 11.2:
echo python -m pip install paddlepaddle-gpu==2.5.2.post112 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
echo.

pause


