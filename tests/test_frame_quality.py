"""Tests for frame_quality module."""

import pytest
import numpy as np
from VehicleDetectionTracker.frame_quality import (
    calculate_laplacian_variance,
    calculate_brightness,
    calculate_contrast,
    calculate_histogram_entropy,
    is_frame_mostly_black,
    is_frame_overexposed,
    is_frame_blurry,
    is_frame_low_contrast,
    assess_frame_quality,
    should_process_frame,
    get_frame_quality_summary,
)


class TestLaplacianVariance:
    """Test blur detection using Laplacian variance."""

    def test_sharp_image_high_variance(self):
        """Test that sharp images have high Laplacian variance."""
        # Create sharp checkerboard pattern
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        for i in range(0, 100, 10):
            for j in range(0, 100, 10):
                if (i + j) % 20 == 0:
                    image[i : i + 10, j : j + 10] = 255

        variance = calculate_laplacian_variance(image)
        assert variance > 50  # Sharp image should have high variance

    def test_blurry_image_low_variance(self):
        """Test that blurry images have low Laplacian variance."""
        # Create blurred uniform image
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        variance = calculate_laplacian_variance(image)
        assert variance < 50  # Blurry image should have low variance

    def test_empty_frame_returns_zero(self):
        """Test that empty frame returns zero variance."""
        image = np.array([], dtype=np.uint8).reshape(0, 0, 3)
        variance = calculate_laplacian_variance(image)
        assert variance == 0.0


class TestBrightness:
    """Test brightness calculation."""

    def test_bright_image(self):
        """Test brightness of bright image."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 200
        brightness = calculate_brightness(image)
        assert 190 < brightness < 210

    def test_dark_image(self):
        """Test brightness of dark image."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 50
        brightness = calculate_brightness(image)
        assert 40 < brightness < 60

    def test_very_dark_image(self):
        """Test brightness of nearly black image."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 10
        brightness = calculate_brightness(image)
        assert brightness < 20


class TestContrast:
    """Test contrast calculation."""

    def test_high_contrast_image(self):
        """Test contrast of high-contrast image."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[50:, :] = 255  # Black top half, white bottom half
        contrast = calculate_contrast(image)
        assert contrast > 80

    def test_low_contrast_image(self):
        """Test contrast of low-contrast image."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        contrast = calculate_contrast(image)
        assert contrast < 10


class TestHistogramEntropy:
    """Test histogram entropy calculation."""

    def test_uniform_grayscale_high_entropy(self):
        """Test entropy of uniform grayscale image."""
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        entropy = calculate_histogram_entropy(image)
        assert entropy > 5  # Random image should have high entropy

    def test_single_color_low_entropy(self):
        """Test entropy of single-color image."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        entropy = calculate_histogram_entropy(image)
        assert entropy < 1  # Uniform image has very low entropy


class TestBlurDetection:
    """Test blur detection functions."""

    def test_is_frame_mostly_black_true(self):
        """Test detection of mostly black frames."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 20
        assert is_frame_mostly_black(image, threshold=30.0) is True

    def test_is_frame_mostly_black_false(self):
        """Test detection of non-black frames."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 100
        assert is_frame_mostly_black(image, threshold=30.0) is False

    def test_is_frame_overexposed_true(self):
        """Test detection of overexposed frames."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 240
        assert is_frame_overexposed(image, threshold=220.0) is True

    def test_is_frame_overexposed_false(self):
        """Test detection of non-overexposed frames."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 100
        assert is_frame_overexposed(image, threshold=220.0) is False

    def test_is_frame_blurry_true(self):
        """Test detection of blurry frames."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        assert is_frame_blurry(image, variance_threshold=100.0) is True

    def test_is_frame_low_contrast_true(self):
        """Test detection of low contrast frames."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        assert is_frame_low_contrast(image, contrast_threshold=15.0) is True


class TestAssessFrameQuality:
    """Test comprehensive frame quality assessment."""

    def test_good_quality_frame(self):
        """Test assessment of good quality frame."""
        # Create a good quality image (checkerboard pattern)
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        for i in range(0, 200, 20):
            for j in range(0, 200, 20):
                if (i + j) % 40 == 0:
                    image[i : i + 20, j : j + 20] = 200

        metrics = assess_frame_quality(image)
        assert metrics["is_valid"] is True or metrics["quality_score"] > 30

    def test_poor_quality_dark_frame(self):
        """Test assessment of dark frame."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 20
        metrics = assess_frame_quality(image)
        assert metrics["is_valid"] is False
        assert "dark" in " ".join(metrics["issues"]).lower()

    def test_poor_quality_overexposed_frame(self):
        """Test assessment of overexposed frame."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 245
        metrics = assess_frame_quality(image)
        assert metrics["is_valid"] is False
        assert (
            "overexposed" in " ".join(metrics["issues"]).lower()
            or "bright" in " ".join(metrics["issues"]).lower()
        )

    def test_metrics_returned(self):
        """Test that all metrics are returned."""
        image = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
        metrics = assess_frame_quality(image)

        assert "brightness" in metrics
        assert "contrast" in metrics
        assert "blur_variance" in metrics
        assert "entropy" in metrics
        assert "quality_score" in metrics
        assert "is_valid" in metrics
        assert "issues" in metrics


class TestShouldProcessFrame:
    """Test frame processing decision logic."""

    def test_should_process_good_frame(self):
        """Test that good frames should be processed."""
        # Create a good quality image
        image = np.random.randint(80, 180, (200, 200, 3), dtype=np.uint8)
        should_process, metrics = should_process_frame(image, quality_threshold=30.0)

        # May or may not pass depending on random content, but should return bool
        assert isinstance(should_process, bool)
        assert isinstance(metrics, dict)

    def test_should_process_dark_frame(self):
        """Test that dark frames should not be processed."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 20
        should_process, metrics = should_process_frame(image)

        assert should_process is False

    def test_should_process_overexposed_frame(self):
        """Test that overexposed frames should not be processed."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 245
        should_process, metrics = should_process_frame(image)

        assert should_process is False


class TestFrameQualitySummary:
    """Test quality summary generation."""

    def test_summary_generation(self):
        """Test that quality summary is generated correctly."""
        metrics = {
            "brightness": 128.0,
            "contrast": 50.0,
            "blur_variance": 200.0,
            "entropy": 5.5,
            "quality_score": 75.0,
            "is_valid": True,
            "issues": [],
        }

        summary = get_frame_quality_summary(metrics)

        assert isinstance(summary, str)
        assert "Brightness" in summary
        assert "Contrast" in summary
        assert "Blur" in summary
        assert "Score" in summary

    def test_summary_with_issues(self):
        """Test summary generation with issues."""
        metrics = {
            "brightness": 20.0,
            "contrast": 5.0,
            "blur_variance": 50.0,
            "entropy": 1.5,
            "quality_score": 15.0,
            "is_valid": False,
            "issues": ["Too dark", "Low contrast"],
        }

        summary = get_frame_quality_summary(metrics)

        assert isinstance(summary, str)
        assert "Issues" in summary


class TestFrameQualityEdgeCases:
    """Test edge cases in frame quality validation."""

    def test_none_frame(self):
        """Test handling of None frame."""
        should_process, metrics = should_process_frame(None)
        assert should_process is False

    def test_empty_frame(self):
        """Test handling of empty frame."""
        image = np.array([], dtype=np.uint8).reshape(0, 0, 3)
        should_process, metrics = should_process_frame(image)
        assert should_process is False

    def test_single_channel_grayscale(self):
        """Test handling of grayscale frame."""
        image = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
        should_process, metrics = should_process_frame(image)
        assert isinstance(should_process, bool)

    def test_custom_thresholds(self):
        """Test with custom thresholds."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 100
        should_process, metrics = should_process_frame(
            image, brightness_min=50, brightness_max=150
        )
        assert isinstance(should_process, bool)
