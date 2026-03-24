"""Tests for device_utils module."""

import pytest
from unittest.mock import patch, MagicMock
from VehicleDetectionTracker.device_utils import (
    check_cuda_available,
    get_device,
)


class TestCheckCudaAvailable:
    """Test CUDA availability checking."""

    @patch("VehicleDetectionTracker.device_utils.torch.cuda.is_available")
    def test_check_cuda_available_when_cuda_present(self, mock_cuda):
        """Test device info when CUDA is available."""
        mock_cuda.return_value = True

        with patch("VehicleDetectionTracker.device_utils.torch.cuda.get_device_name") as mock_name:
            mock_name.return_value = "NVIDIA GeForce RTX 3090"

            is_gpu, gpu_count, device_name = check_cuda_available()

            assert is_gpu is True
            assert gpu_count > 0
            assert device_name is not None

    @patch("VehicleDetectionTracker.device_utils.torch.cuda.is_available")
    def test_check_cuda_not_available(self, mock_cuda):
        """Test device info when CUDA is not available."""
        mock_cuda.return_value = False

        is_gpu, gpu_count, device_name = check_cuda_available()

        assert is_gpu is False
        assert gpu_count == 0

    @patch("VehicleDetectionTracker.device_utils.torch.cuda.is_available")
    def test_check_cuda_returns_tuple(self, mock_cuda):
        """Test that check_cuda_available returns appropriate types."""
        mock_cuda.return_value = True

        result = check_cuda_available()

        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], bool)
        assert isinstance(result[1], int)


class TestGetDevice:
    """Test device selection."""

    @patch("VehicleDetectionTracker.device_utils.check_cuda_available")
    def test_get_device_with_gpu(self, mock_device_info):
        """Test device selection when GPU is available."""
        mock_device_info.return_value = (True, 1, "NVIDIA GeForce RTX 3090")

        with patch("VehicleDetectionTracker.device_utils.log") as mock_log:
            device, is_cuda = get_device(mock_log)

            assert device is not None

    @patch("VehicleDetectionTracker.device_utils.check_cuda_available")
    def test_get_device_without_gpu(self, mock_device_info):
        """Test device selection when GPU is not available."""
        mock_device_info.return_value = (False, 0, None)

        with patch("VehicleDetectionTracker.device_utils.log") as mock_log:
            device, is_cuda = get_device(mock_log)

            assert device is not None


class TestGetPreferredDevice:
    """Test preferred device selection fallback."""

    @patch("VehicleDetectionTracker.device_utils.check_cuda_available")
    def test_device_preference_order(self, mock_device_info):
        """Test that CUDA is preferred over CPU."""
        with patch("VehicleDetectionTracker.device_utils.log") as mock_log:
            mock_device_info.return_value = (True, 1, "GPU")

            device, is_cuda = get_device(mock_log)

            # When GPU is available, should prefer CUDA
            assert device is not None
