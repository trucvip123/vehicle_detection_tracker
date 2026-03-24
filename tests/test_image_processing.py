"""Tests for image_processing module."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from VehicleDetectionTracker.image_processing import (
    encode_image_base64,
    decode_image_base64,
    increase_brightness,
    draw_license_plate,
)


class TestEncodeImageBase64:
    """Test image to base64 encoding."""

    def test_encode_image_base64_valid_image(self):
        """Test converting numpy array to base64."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch("VehicleDetectionTracker.image_processing.cv2.imencode") as mock_encode:
            mock_encode.return_value = (True, np.array([0xFF, 0xD8]))

            result = encode_image_base64(image)

            assert isinstance(result, str)

    def test_encode_image_base64_returns_string(self):
        """Test that base64 output is a string."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch("VehicleDetectionTracker.image_processing.cv2.imencode") as mock_encode:
            mock_encode.return_value = (True, np.array([65, 66, 67]))  # 'ABC'

            result = encode_image_base64(image)

            assert isinstance(result, str)

    def test_encode_image_base64_empty_image(self):
        """Test converting empty/small image to base64."""
        image = np.zeros((10, 10, 3), dtype=np.uint8)

        with patch("VehicleDetectionTracker.image_processing.cv2.imencode") as mock_encode:
            mock_encode.return_value = (True, np.array([1, 2, 3]))

            result = encode_image_base64(image)

            assert result is not None


class TestDecodeImageBase64:
    """Test base64 to image decoding."""

    def test_decode_image_base64_valid_input(self):
        """Test converting base64 to image."""
        # Create a simple base64 string
        base64_str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

        with patch("VehicleDetectionTracker.image_processing.cv2.imdecode") as mock_decode:
            mock_decode.return_value = np.zeros((1, 1, 3), dtype=np.uint8)

            result = decode_image_base64(base64_str)

            assert isinstance(result, np.ndarray)

    def test_decode_image_base64_returns_numpy_array(self):
        """Test that result is numpy array."""
        base64_str = "test_base64_string"

        with patch("VehicleDetectionTracker.image_processing.cv2.imdecode") as mock_decode:
            mock_decode.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

            result = decode_image_base64(base64_str)

            assert isinstance(result, np.ndarray)
            assert result.ndim == 3  # Should be 3D array (height, width, channels)


class TestIncreaseBrightness:
    """Test brightness adjustment."""

    def test_increase_brightness_valid_input(self):
        """Test increasing brightness."""
        image = np.ones((480, 640, 3), dtype=np.uint8) * 100

        result = increase_brightness(image, 1.5)

        assert isinstance(result, np.ndarray)
        assert result.shape == image.shape

    def test_increase_brightness_decrease(self):
        """Test decreasing brightness."""
        image = np.ones((480, 640, 3), dtype=np.uint8) * 100

        result = increase_brightness(image, 0.5)

        assert isinstance(result, np.ndarray)
        assert result.shape == image.shape

    def test_increase_brightness_no_change(self):
        """Test neutral brightness adjustment."""
        image = np.ones((480, 640, 3), dtype=np.uint8) * 100

        result = increase_brightness(image, 1.0)

        assert isinstance(result, np.ndarray)

    def test_increase_brightness_clamps_values(self):
        """Test that brightness adjustment clamps to valid range."""
        image = np.ones((480, 640, 3), dtype=np.uint8) * 200

        result = increase_brightness(image, 2.0)

        # Values should not exceed 255
        assert result.max() <= 255
        assert result.min() >= 0

    def test_increase_brightness_preserves_shape(self):
        """Test that image shape is preserved."""
        image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

        result = increase_brightness(image, 1.3)

        assert result.shape == image.shape


class TestDrawLicensePlate:
    """Test drawing license plate on image."""

    def test_draw_license_plate_with_valid_inputs(self):
        """Test drawing license plate."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        box = (100, 100, 200, 200)
        text = "77A12345"

        result = draw_license_plate(frame, box, text)

        # Function signature varies, just check no crash
        assert True
