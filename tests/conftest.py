"""Pytest configuration and shared fixtures for tests."""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock
from typing import Dict, Any


@pytest.fixture
def mock_logger():
    """Mock logger function."""
    return Mock(return_value=None)


@pytest.fixture
def mock_ocr_reader():
    """Mock OCR reader."""
    reader = Mock()
    reader.ocr = Mock(return_value=[[("Test Plate", 0.95)]])
    return reader


@pytest.fixture
def mock_plate_model():
    """Mock plate detection model."""
    return Mock()


@pytest.fixture
def mock_executor():
    """Mock ThreadPoolExecutor."""
    executor = MagicMock()
    executor.submit = Mock(return_value=MagicMock())
    return executor


@pytest.fixture
def mock_detection_model():
    """Mock YOLO detection model."""
    model = Mock()
    # Mock track results
    track_result = Mock()
    track_result.boxes = Mock()
    track_result.boxes.id = Mock(return_value=None)
    track_result.boxes.xywh = Mock()
    track_result.boxes.cls = Mock()
    model.track = Mock(return_value=[track_result])
    return model


@pytest.fixture
def sample_frame():
    """Sample video frame (numpy array)."""
    import numpy as np
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_timestamp():
    """Sample timestamp."""
    return datetime(2026, 3, 24, 10, 30, 0)


@pytest.fixture
def sample_vehicle_data() -> Dict[int, Any]:
    """Sample test vehicle data."""
    return {
        "track_id": 1,
        "plate": "77A123.45",
        "direction": "Bottom",
        "timestamp": datetime(2026, 3, 24, 10, 30, 0),
    }
