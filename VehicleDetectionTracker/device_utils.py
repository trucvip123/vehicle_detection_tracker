"""Device utilities for GPU/CPU management."""
import torch


def check_cuda_available():
    """Kiểm tra xem CUDA có sẵn và có GPU không"""
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        return True, gpu_count, gpu_name
    return False, 0, None


def get_device(log_func):
    """Trả về device phù hợp (cuda hoặc cpu)"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        log_func(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
        log_func(f"✓ CUDA version: {torch.version.cuda}")
        return device, True
    else:
        log_func("⚠ GPU không khả dụng, sử dụng CPU")
        return torch.device('cpu'), False
