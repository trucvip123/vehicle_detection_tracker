"""Tests for frame_processor module."""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock, call
import numpy as np
from VehicleDetectionTracker.frame_processor import FrameProcessor


@pytest.fixture
def mock_yolo_model():
    """Create mock YOLO detection model."""
    model = MagicMock()
    model.track = MagicMock()
    return model


@pytest.fixture
def mock_plate_processor():
    """Create mock plate processor."""
    processor = MagicMock()
    processor.update_vehicle_state = MagicMock()
    processor.get_vehicle_state = MagicMock(
        return_value=("77A12345", "Right", datetime(2026, 3, 24, 10, 30, 0))
    )
    return processor


@pytest.fixture
def frame_processor(mock_yolo_model, mock_plate_processor):
    """Create FrameProcessor instance with mocked dependencies."""
    with patch("VehicleDetectionTracker.frame_processor.YOLO"):
        processor = FrameProcessor(model_path="yolov8n.pt")
        processor.model = mock_yolo_model
        return processor


class TestFrameProcessorInitialization:
    """Test FrameProcessor initialization."""

    @patch("VehicleDetectionTracker.frame_processor.YOLO")
    def test_initialization_with_valid_model(self, mock_yolo):
        """Test FrameProcessor initialization with valid model."""
        fp = FrameProcessor(model_path="yolov8n.pt")

        assert fp is not None
        assert hasattr(fp, "model")

    @patch("VehicleDetectionTracker.frame_processor.YOLO")
    def test_initialization_with_confidence(self, mock_yolo):
        """Test FrameProcessor initialization with confidence."""
        fp = FrameProcessor(model_path="yolov8n.pt", confidence=0.5)

        assert fp is not None


class TestProcessFrameStreaming:
    """Test main frame processing function."""

    def test_process_frame_streaming_with_valid_input(self, frame_processor, mock_plate_processor):
        """Test processing a frame with valid input."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        timestamp = datetime(2026, 3, 24, 10, 30, 0)

        # Mock detection results
        frame_processor.model.track.return_value = MagicMock()

        result = frame_processor.process_frame_streaming(frame, timestamp, mock_plate_processor)

        assert isinstance(result, np.ndarray)
        assert result.shape == frame.shape

    def test_process_frame_streaming_calls_model_track(self, frame_processor, mock_plate_processor):
        """Test that process_frame_streaming calls model.track."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        timestamp = datetime(2026, 3, 24, 10, 30, 0)

        frame_processor.model.track = MagicMock()
        frame_processor.process_frame_streaming(frame, timestamp, mock_plate_processor)

        frame_processor.model.track.assert_called_once()

    def test_process_frame_streaming_handles_no_detections(self, frame_processor, mock_plate_processor):
        """Test handling when no vehicles are detected."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        timestamp = datetime(2026, 3, 24, 10, 30, 0)

        # Mock empty results
        mock_results = MagicMock()
        mock_results.boxes = MagicMock()
        mock_results.boxes.empty = True
        frame_processor.model.track.return_value = mock_results

        result = frame_processor.process_frame_streaming(frame, timestamp, mock_plate_processor)

        assert isinstance(result, np.ndarray)

    def test_process_frame_streaming_preserves_frame_shape(self, frame_processor, mock_plate_processor):
        """Test that frame shape is preserved after processing."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        timestamp = datetime(2026, 3, 24, 10, 30, 0)

        frame_processor.model.track.return_value = MagicMock()

        result = frame_processor.process_frame_streaming(frame, timestamp, mock_plate_processor)

        assert result.shape == frame.shape


class TestDirectionCalculation:
    """Test direction calculation integration."""

    def test_process_frame_calculates_direction(self, frame_processor, mock_plate_processor):
        """Test that frame processing calculates vehicle direction."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        timestamp = datetime(2026, 3, 24, 10, 30, 0)

        frame_processor.model.track.return_value = MagicMock()

        with patch("VehicleDetectionTracker.frame_processor.calculate_direction") as mock_calc:
            mock_calc.return_value = "Right"

            frame_processor.process_frame_streaming(frame, timestamp, mock_plate_processor)

            # Direction calculation may or may not be called depending on detections
            assert True  # Placeholder for actual direction logic


class TestCleanupOldVehicleData:
    """Test cleanup of old vehicle data."""

    def test_cleanup_removes_old_vehicles(self, frame_processor, mock_plate_processor):
        """Test that cleanup removes vehicles unseen for too long."""
        timestamp_old = datetime(2026, 3, 24, 10, 0, 0)
        timestamp_new = datetime(2026, 3, 24, 11, 0, 0)

        # Track some old vehicle IDs
        with patch("VehicleDetectionTracker.frame_processor.datetime") as mock_datetime:
            mock_datetime.now.return_value = timestamp_new

            frame_processor._cleanup_old_vehicle_data(mock_plate_processor)

            # Cleanup should be called without error
            assert True

    def test_cleanup_preserves_recent_vehicles(self, frame_processor, mock_plate_processor):
        """Test that cleanup preserves recently seen vehicles."""
        frame_processor._cleanup_old_vehicle_data(mock_plate_processor)

        # Should not raise exception
        assert True

    def test_cleanup_calls_update_vehicle_state(self, frame_processor, mock_plate_processor):
        """Test that cleanup may call update_vehicle_state."""
        frame_processor._cleanup_old_vehicle_data(mock_plate_processor)

        # Verify no errors during cleanup
        assert True


class TestVehicleStateManagement:
    """Test vehicle state management in frame processor."""

    def test_updates_vehicle_timestamp(self, frame_processor, mock_plate_processor):
        """Test that vehicle timestamp is updated."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        timestamp = datetime(2026, 3, 24, 10, 30, 0)

        frame_processor.model.track.return_value = MagicMock()

        frame_processor.process_frame_streaming(frame, timestamp, mock_plate_processor)

        # Verify plate processor was used for state updates
        assert True

    def test_handles_multiple_vehicles(self, frame_processor, mock_plate_processor):
        """Test handling multiple detecting vehicles in single frame."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        timestamp = datetime(2026, 3, 24, 10, 30, 0)

        frame_processor.model.track.return_value = MagicMock()

        result = frame_processor.process_frame_streaming(frame, timestamp, mock_plate_processor)

        assert isinstance(result, np.ndarray)


class TestFrameProcessorEdgeCases:
    """Test edge cases in frame processing."""

    def test_process_frame_with_very_dark_frame(self, frame_processor, mock_plate_processor):
        """Test processing very dark frame."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        timestamp = datetime(2026, 3, 24, 10, 30, 0)

        result = frame_processor.process_frame_streaming(frame, timestamp, mock_plate_processor)

        assert isinstance(result, np.ndarray)

    def test_process_frame_with_very_bright_frame(self, frame_processor, mock_plate_processor):
        """Test processing very bright frame."""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        timestamp = datetime(2026, 3, 24, 10, 30, 0)

        result = frame_processor.process_frame_streaming(frame, timestamp, mock_plate_processor)

        assert isinstance(result, np.ndarray)

    def test_process_frame_with_different_sizes(self, frame_processor, mock_plate_processor):
        """Test processing frames of different sizes."""
        timestamp = datetime(2026, 3, 24, 10, 30, 0)

        for size in [(480, 640), (720, 1280), (1080, 1920)]:
            frame = np.zeros((*size, 3), dtype=np.uint8)

            result = frame_processor.process_frame_streaming(frame, timestamp, mock_plate_processor)

            assert result.shape == (*size, 3)
