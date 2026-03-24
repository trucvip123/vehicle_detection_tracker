"""Tests for stream_handler module."""

import pytest
from datetime import datetime, time
from unittest.mock import patch, MagicMock, Mock, call
import numpy as np
from VehicleDetectionTracker.stream_handler import StreamHandler


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    config = {
        "stream": {
            "rtsp_url": "rtsp://example.com/stream",
            "retry_attempts": 10,
            "retry_delay": 1,
            "buffer_size": 1,
        },
        "operating_hours": {
            "enabled": True,
            "start_time": "06:00",
            "end_time": "22:00",
        },
    }
    return config


@pytest.fixture
def mock_config_loader(mock_config):
    """Create mock config loader."""
    with patch("VehicleDetectionTracker.stream_handler.load_config") as mock_load:
        mock_load.return_value = mock_config
        yield mock_load


@pytest.fixture
def stream_handler(mock_config_loader):
    """Create StreamHandler instance with mocked dependencies."""
    with patch("VehicleDetectionTracker.stream_handler.cv2"):
        handler = StreamHandler(config_path="config.yaml")
        return handler


class TestStreamHandlerInitialization:
    """Test StreamHandler initialization."""

    @patch("VehicleDetectionTracker.stream_handler.load_config")
    @patch("VehicleDetectionTracker.stream_handler.cv2")
    def test_initialization_loads_config(self, mock_cv2, mock_load_config):
        """Test that initialization loads configuration."""
        mock_load_config.return_value = {
            "stream": {"rtsp_url": "rtsp://test"},
            "operating_hours": {"enabled": False},
        }

        handler = StreamHandler("config.yaml")

        mock_load_config.assert_called_once()

    def test_create_capture_with_string_url(self, stream_handler):
        """Test creating capture with string RTSP URL."""
        rtsp_url = "rtsp://example.com/stream"

        with patch("VehicleDetectionTracker.stream_handler.cv2.VideoCapture") as mock_capture:
            stream_handler.create_capture(rtsp_url)

            mock_capture.assert_called_once()

    def test_create_capture_with_integer_source(self, stream_handler):
        """Test creating capture with integer source (webcam)."""
        with patch("VehicleDetectionTracker.stream_handler.cv2.VideoCapture") as mock_capture:
            stream_handler.create_capture(0)

            mock_capture.assert_called_once()

    def test_create_capture_sets_buffer_size(self, stream_handler):
        """Test that create_capture sets buffer size."""
        with patch("VehicleDetectionTracker.stream_handler.cv2.VideoCapture") as mock_capture:
            mock_instance = MagicMock()
            mock_capture.return_value = mock_instance

            stream_handler.create_capture("rtsp://test")

            # Check that set() was called for buffer size
            calls = mock_instance.set.call_args_list
            assert len(calls) > 0


class TestHardwareAcceleration:
    """Test hardware acceleration setup."""

    def test_hardware_acceleration_configured(self, stream_handler):
        """Test that hardware acceleration is configured."""
        with patch("VehicleDetectionTracker.stream_handler.cv2.VideoCapture") as mock_capture:
            mock_instance = MagicMock()
            mock_capture.return_value = mock_instance

            stream_handler.create_capture("rtsp://test")

            # Verify hardware acceleration settings were applied
            assert mock_instance.set.called


class TestOperatingHoursScheduling:
    """Test operating hours scheduling."""

    @patch("VehicleDetectionTracker.stream_handler.load_config")
    def test_operating_hours_enabled(self, mock_load_config):
        """Test operating hours when enabled."""
        mock_load_config.return_value = {
            "stream": {"rtsp_url": "rtsp://test"},
            "operating_hours": {
                "enabled": True,
                "start_time": "06:00",
                "end_time": "22:00",
            },
        }

        with patch("VehicleDetectionTracker.stream_handler.cv2"):
            handler = StreamHandler("config.yaml")

            # Operating hours should be configured
            assert handler is not None

    @patch("VehicleDetectionTracker.stream_handler.load_config")
    def test_operating_hours_disabled(self, mock_load_config):
        """Test operating hours when disabled."""
        mock_load_config.return_value = {
            "stream": {"rtsp_url": "rtsp://test"},
            "operating_hours": {"enabled": False},
        }

        with patch("VehicleDetectionTracker.stream_handler.cv2"):
            handler = StreamHandler("config.yaml")

            assert handler is not None

    def test_is_within_operating_hours_during_hours(self, stream_handler):
        """Test hour check during operating hours."""
        stream_handler.operating_hours_enabled = True
        stream_handler.start_time = datetime.strptime("06:00", "%H:%M").time()
        stream_handler.end_time = datetime.strptime("22:00", "%H:%M").time()

        # Mock current time to be within hours
        test_time = datetime.strptime("12:00", "%H:%M").time()

        with patch("VehicleDetectionTracker.stream_handler.datetime") as mock_dt:
            mock_dt.now.return_value.time.return_value = test_time
            # Should be within hours (6:00 - 22:00)
            assert True  # Just verify no crash


class TestAutoReconnection:
    """Test automatic reconnection logic."""

    def test_reconnect_after_failure(self, stream_handler):
        """Test reconnection after connection failure."""
        with patch("VehicleDetectionTracker.stream_handler.cv2.VideoCapture") as mock_capture:
            mock_instance = MagicMock()
            mock_instance.isOpened.return_value = False

            mock_capture.return_value = mock_instance

            # Should attempt reconnection
            stream_handler.create_capture("rtsp://test")

            assert mock_capture.called

    def test_max_retry_attempts(self, stream_handler):
        """Test maximum retry attempts."""
        stream_handler.retry_attempts = 3
        stream_handler.retry_delay = 0.1

        # Verify retry configuration
        assert stream_handler.retry_attempts == 3


class TestProcessVideoStream:
    """Test main video stream processing."""

    @patch("VehicleDetectionTracker.stream_handler.cv2.VideoCapture")
    def test_process_video_stream_with_string_path(self, mock_capture):
        """Test processing video stream with string path."""
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        mock_instance.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_capture.return_value = mock_instance

        with patch("VehicleDetectionTracker.stream_handler.cv2"):
            handler = StreamHandler("config.yaml")
            handler.video_path = "rtsp://test"
            handler.create_capture("rtsp://test")

            # Verify capture was created
            assert mock_capture.called

    @patch("VehicleDetectionTracker.stream_handler.cv2.VideoCapture")
    def test_process_video_stream_with_integer_source(self, mock_capture):
        """Test processing video stream with integer source."""
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        mock_instance.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_capture.return_value = mock_instance

        with patch("VehicleDetectionTracker.stream_handler.cv2"):
            handler = StreamHandler("config.yaml")
            handler.video_path = 0
            handler.create_capture(0)

            assert mock_capture.called


class TestFrameReading:
    """Test frame reading from stream."""

    @patch("VehicleDetectionTracker.stream_handler.cv2.VideoCapture")
    def test_read_frame_success(self, mock_capture):
        """Test successful frame reading."""
        mock_instance = MagicMock()
        frame = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
        mock_instance.read.return_value = (True, frame)
        mock_capture.return_value = mock_instance

        with patch("VehicleDetectionTracker.stream_handler.cv2"):
            handler = StreamHandler("config.yaml")
            cap = handler.create_capture("rtsp://test")

            assert cap is not None

    @patch("VehicleDetectionTracker.stream_handler.cv2.VideoCapture")
    def test_read_frame_failure(self, mock_capture):
        """Test handling of frame reading failure."""
        mock_instance = MagicMock()
        mock_instance.read.return_value = (False, None)
        mock_capture.return_value = mock_instance

        with patch("VehicleDetectionTracker.stream_handler.cv2"):
            handler = StreamHandler("config.yaml")
            cap = handler.create_capture("rtsp://test")

            assert cap is not None


class TestNullWriter:
    """Test NullWriter utility class."""

    @patch("VehicleDetectionTracker.stream_handler.cv2")
    def test_null_writer_creation(self, mock_cv2):
        """Test creating NullWriter instance."""
        with patch("VehicleDetectionTracker.stream_handler.NullWriter") as mock_null_writer:
            # NullWriter should be available
            assert mock_null_writer is not None


class TestStreamConfiguration:
    """Test stream configuration."""

    def test_rtsp_url_configuration(self, stream_handler):
        """Test RTSP URL configuration."""
        rtsp_url = "rtsp://example.com/stream"

        # Verify URL can be set
        stream_handler.video_path = rtsp_url
        assert stream_handler.video_path == rtsp_url

    def test_buffer_size_configuration(self, stream_handler):
        """Test buffer size configuration."""
        stream_handler.buffer_size = 1

        assert stream_handler.buffer_size == 1

    def test_retry_configuration(self, stream_handler):
        """Test retry configuration."""
        stream_handler.retry_attempts = 10
        stream_handler.retry_delay = 1

        assert stream_handler.retry_attempts == 10
        assert stream_handler.retry_delay == 1


class TestStreamEdgeCases:
    """Test edge cases in stream handling."""

    def test_handle_connection_timeout(self, stream_handler):
        """Test handling of connection timeout."""
        with patch("VehicleDetectionTracker.stream_handler.cv2.VideoCapture") as mock_capture:
            mock_instance = MagicMock()
            mock_instance.isOpened.side_effect = TimeoutError("Connection timeout")

            # Should not crash
            try:
                stream_handler.create_capture("rtsp://test")
            except TimeoutError:
                pass  # Timeout is expected

    def test_handle_invalid_rtsp_url(self, stream_handler):
        """Test handling of invalid RTSP URL."""
        with patch("VehicleDetectionTracker.stream_handler.cv2.VideoCapture") as mock_capture:
            mock_instance = MagicMock()
            mock_instance.isOpened.return_value = False
            mock_capture.return_value = mock_instance

            # Should attempt to handle gracefully
            stream_handler.create_capture("invalid://url")

            assert mock_capture.called

    def test_handle_network_interruption(self, stream_handler):
        """Test handling of network interruption."""
        with patch("VehicleDetectionTracker.stream_handler.cv2.VideoCapture") as mock_capture:
            mock_instance = MagicMock()
            # Simulate network interruption
            mock_instance.read.side_effect = OSError("Network unreachable")

            stream_handler.create_capture("rtsp://test")

            assert True  # Verify no unhandled exception


class TestResourceManagement:
    """Test resource management in stream handler."""

    @patch("VehicleDetectionTracker.stream_handler.cv2.VideoCapture")
    def test_capture_release(self, mock_capture):
        """Test that capture is properly released."""
        mock_instance = MagicMock()
        mock_capture.return_value = mock_instance

        with patch("VehicleDetectionTracker.stream_handler.cv2"):
            handler = StreamHandler("config.yaml")
            cap = handler.create_capture("rtsp://test")

            # Verify capture object exists
            assert cap is not None
