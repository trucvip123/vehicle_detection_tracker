"""
Script kiểm tra cài đặt GPU cho Vehicle Detection Tracker
"""

import sys


def check_pytorch():
    """Kiểm tra PyTorch và CUDA"""
    print("=" * 60)
    print("KIỂM TRA PYTORCH & CUDA")
    print("=" * 60)

    try:
        import torch

        print(f"✓ PyTorch đã được cài đặt")
        print(f"  Version: {torch.__version__}")

        if torch.cuda.is_available():
            print(f"✓ CUDA có sẵn")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  cuDNN version: {torch.backends.cudnn.version()}")
            print(f"  GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                print(
                    f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB"
                )
        else:
            print("❌ CUDA không khả dụng")
            if "+cpu" in torch.__version__:
                print("  ⚠ PyTorch đang ở version CPU")
                print("  💡 Giải pháp: Cài PyTorch với CUDA:")
                print(
                    "     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
                )
            else:
                print(
                    "  ⚠ Có thể do driver NVIDIA chưa được cài đặt hoặc không tương thích"
                )

        return torch.cuda.is_available()

    except ImportError:
        print("❌ PyTorch chưa được cài đặt")
        return False


def check_paddle():
    """Kiểm tra PaddlePaddle"""
    print("\n" + "=" * 60)
    print("KIỂM TRA PADDLEPADDLE")
    print("=" * 60)

    try:
        import paddle

        print(f"✓ PaddlePaddle đã được cài đặt")
        print(f"  Version: {paddle.__version__}")

        if paddle.device.is_compiled_with_cuda():
            print(f"✓ PaddlePaddle được compile với CUDA support")
            try:
                # Thử set GPU
                paddle.device.set_device("gpu")
                print(f"✓ GPU có thể sử dụng với PaddlePaddle")
                return True
            except Exception as e:
                print(f"⚠ GPU không khả dụng cho PaddlePaddle: {e}")
                print("  💡 Giải pháp: Cài paddlepaddle-gpu:")
                print(
                    "     python -m pip install paddlepaddle-gpu==2.5.2.post117 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html"
                )
                return False
        else:
            print("❌ PaddlePaddle được compile cho CPU (không có CUDA)")
            print("  💡 Giải pháp: Cài paddlepaddle-gpu thay vì paddlepaddle")
            return False

    except ImportError:
        print("❌ PaddlePaddle chưa được cài đặt")
        return False


def check_nvidia_driver():
    """Kiểm tra NVIDIA Driver"""
    print("\n" + "=" * 60)
    print("KIỂM TRA NVIDIA DRIVER")
    print("=" * 60)

    import subprocess

    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            print("✓ NVIDIA Driver đã được cài đặt")
            # Parse output để lấy thông tin
            lines = result.stdout.split("\n")
            for line in lines:
                if "Driver Version" in line:
                    print(f"  {line.strip()}")
                elif "CUDA Version" in line:
                    print(f"  {line.strip()}")
                elif (
                    "GeForce" in line
                    or "Quadro" in line
                    or "Tesla" in line
                    or "RTX" in line
                ):
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part in ["GeForce", "Quadro", "Tesla", "RTX"]:
                            if i + 1 < len(parts):
                                print(f"  GPU: {' '.join(parts[i:i+2])}")
                                break
            return True
        else:
            print("❌ Không thể chạy nvidia-smi")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi không tìm thấy (NVIDIA Driver chưa được cài đặt)")
        return False
    except subprocess.TimeoutExpired:
        print("⚠ nvidia-smi timeout")
        return False
    except Exception as e:
        print(f"❌ Lỗi khi kiểm tra: {e}")
        return False


def main():
    """Main function"""
    print("\n" + "=" * 60)
    print("KIỂM TRA CÀI ĐẶT GPU CHO VEHICLE DETECTION TRACKER")
    print("=" * 60 + "\n")

    nvidia_ok = check_nvidia_driver()
    pytorch_ok = check_pytorch()
    paddle_ok = check_paddle()

    print("\n" + "=" * 60)
    print("TÓM TẮT")
    print("=" * 60)
    print(f"NVIDIA Driver:  {'✓' if nvidia_ok else '❌'}")
    print(f"PyTorch CUDA:   {'✓' if pytorch_ok else '❌'}")
    print(f"PaddlePaddle:   {'✓' if paddle_ok else '❌'}")

    if nvidia_ok and pytorch_ok and paddle_ok:
        print("\n🎉 Tất cả đã sẵn sàng để sử dụng GPU!")
    else:
        print("\n⚠ Một số components chưa sẵn sàng.")
        print("  Xem file INSTALL_GPU_REQUIREMENTS.md để biết hướng dẫn chi tiết.")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
