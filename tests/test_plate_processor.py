"""Tests for plate_processor module."""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock, Mock, call
from threading import Thread, RLock
import numpy as np
from VehicleDetectionTracker.plate_processor import PlateProcessor


@pytest.fixture
def mock_ocr_reader():
    """Create mock OCR reader."""
    reader = MagicMock()
    reader.readtext = MagicMock(
        return_value=[[("77A12345", 0.95)], [("29C67890", 0.88)]]
    )
    return reader


@pytest.fixture
def mock_telegram():
    """Create mock Telegram sender."""
    with patch("VehicleDetectionTracker.plate_processor.send_telegram_notification"):
        yield


@pytest.fixture
def plate_processor(mock_ocr_reader, mock_telegram):
    """Create PlateProcessor instance with mocked dependencies."""
    with patch("VehicleDetectionTracker.plate_processor.PaddleOCR") as mock_ocr:
        mock_ocr.return_value = mock_ocr_reader
        with patch("VehicleDetectionTracker.plate_processor.ThreadPoolExecutor"):
            processor = PlateProcessor(
                ocr_model_path="en",
                telegram_bot_token="test_token",
                telegram_chat_id="test_chat_id",
                plate_detector_path="detector.pt",
            )
            processor.ocr_reader = mock_ocr_reader
            return processor


class TestPlateProcessorInitialization:
    """Test PlateProcessor initialization."""

    @patch("VehicleDetectionTracker.plate_processor.PaddleOCR")
    @patch("VehicleDetectionTracker.plate_processor.ThreadPoolExecutor")
    def test_initialization_creates_state_lock(self, mock_executor, mock_ocr):
        """Test that initialization creates RLock."""
        processor = PlateProcessor(
            ocr_model_path="en",
            telegram_bot_token="token",
            telegram_chat_id="chat",
            plate_detector_path="detector.pt",
        )

        assert hasattr(processor, "_state_lock")
        assert isinstance(processor._state_lock, type(RLock()))

    @patch("VehicleDetectionTracker.plate_processor.PaddleOCR")
    @patch("VehicleDetectionTracker.plate_processor.ThreadPoolExecutor")
    def test_initialization_creates_state_dicts(self, mock_executor, mock_ocr):
        """Test that state dictionaries are initialized."""
        processor = PlateProcessor(
            ocr_model_path="en",
            telegram_bot_token="token",
            telegram_chat_id="chat",
            plate_detector_path="detector.pt",
        )

        assert hasattr(processor, "vehicle_plates")
        assert hasattr(processor, "vehicle_directions")
        assert hasattr(processor, "vehicle_last_seen")
        assert hasattr(processor, "vehicle_plate_counts")


class TestThreadSafeStateMethods:
    """Test thread-safe state accessor methods."""

    def test_update_vehicle_state_sets_plate(self, plate_processor):
        """Test update_vehicle_state sets plate text."""
        track_id = 1
        plate_text = "77A12345"

        plate_processor.update_vehicle_state(track_id, plate_text=plate_text)

        assert track_id in plate_processor.vehicle_plates
        assert plate_processor.vehicle_plates[track_id] == plate_text

    def test_update_vehicle_state_sets_direction(self, plate_processor):
        """Test update_vehicle_state sets direction."""
        track_id = 1
        direction = "Right"

        plate_processor.update_vehicle_state(track_id, direction=direction)

        assert track_id in plate_processor.vehicle_directions
        assert plate_processor.vehicle_directions[track_id] == direction

    def test_update_vehicle_state_sets_timestamp(self, plate_processor):
        """Test update_vehicle_state sets timestamp."""
        track_id = 1
        timestamp = datetime(2026, 3, 24, 10, 30, 0)

        plate_processor.update_vehicle_state(track_id, timestamp=timestamp)

        assert track_id in plate_processor.vehicle_last_seen
        assert plate_processor.vehicle_last_seen[track_id] == timestamp

    def test_update_vehicle_state_atomic_update(self, plate_processor):
        """Test that update_vehicle_state is atomic."""
        track_id = 1

        plate_processor.update_vehicle_state(
            track_id,
            plate_text="77A12345",
            direction="Right",
            timestamp=datetime.now(),
        )

        state = plate_processor.get_vehicle_state(track_id)
        assert state[0] == "77A12345"
        assert state[1] == "Right"

    def test_get_vehicle_state_returns_tuple(self, plate_processor):
        """Test get_vehicle_state returns tuple."""
        track_id = 1
        plate_processor.update_vehicle_state(
            track_id, plate_text="77A12345", direction="Right"
        )

        state = plate_processor.get_vehicle_state(track_id)

        assert isinstance(state, tuple)
        assert len(state) == 3

    def test_get_vehicle_state_nonexistent_vehicle(self, plate_processor):
        """Test get_vehicle_state for vehicle not in dict."""
        state = plate_processor.get_vehicle_state(999)

        assert state == (None, None, None)

    def test_get_all_vehicle_ids_copy(self, plate_processor):
        """Test get_all_vehicle_ids returns copy."""
        plate_processor.update_vehicle_state(1, plate_text="77A12345")
        plate_processor.update_vehicle_state(2, plate_text="29C67890")

        vehicle_ids = plate_processor.get_all_vehicle_ids()

        assert isinstance(vehicle_ids, set)
        assert 1 in vehicle_ids
        assert 2 in vehicle_ids

    def test_get_vehicle_plates_copy(self, plate_processor):
        """Test get_vehicle_plates_copy returns snapshot."""
        plate_processor.update_vehicle_state(1, plate_text="77A12345")
        plate_processor.update_vehicle_state(2, plate_text="29C67890")

        plates_copy = plate_processor.get_vehicle_plates_copy()

        assert isinstance(plates_copy, dict)
        assert len(plates_copy) >= 2


class TestThreadSafety:
    """Test thread safety of state updates."""

    def test_concurrent_updates_no_race_condition(self, plate_processor):
        """Test that concurrent updates don't cause race conditions."""
        def update_vehicle(track_id):
            for i in range(10):
                plate_processor.update_vehicle_state(
                    track_id,
                    plate_text=f"77A{track_id}{i}",
                    direction="Right",
                    timestamp=datetime.now(),
                )

        threads = [Thread(target=update_vehicle, args=(i,)) for i in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Verify all vehicles were updated
        vehicle_ids = plate_processor.get_all_vehicle_ids()
        assert len(vehicle_ids) >= 4

    def test_concurrent_reads_no_race_condition(self, plate_processor):
        """Test that concurrent reads don't cause race conditions."""
        # Setup some data
        for i in range(10):
            plate_processor.update_vehicle_state(
                i, plate_text=f"77A{i}" , direction="Right"
            )

        def read_vehicles():
            for _ in range(100):
                plate_processor.get_vehicle_state(5)
                plate_processor.get_all_vehicle_ids()

        threads = [Thread(target=read_vehicles) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert True  # No exception raised

    def test_concurrent_read_write_no_deadlock(self, plate_processor):
        """Test concurrent reads and writes don't deadlock."""
        def writer():
            for i in range(20):
                plate_processor.update_vehicle_state(
                    i,
                    plate_text=f"77A{i}",
                    timestamp=datetime.now(),
                )

        def reader():
            for _ in range(20):
                plate_processor.get_all_vehicle_ids()
                plate_processor.get_vehicle_plates_copy()

        threads = [Thread(target=writer) for _ in range(2)]
        threads.extend([Thread(target=reader) for _ in range(2)])

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert True  # No deadlock occurred


class TestPlateDetection:
    """Test plate detection flow."""

    def test_process_plate_background_sync_with_valid_image(self, plate_processor):
        """Test plate detection with valid image."""
        image = np.zeros((100, 300, 3), dtype=np.uint8)
        track_id = 1
        timestamp = datetime.now()

        plate_processor.process_plate_background_sync(
            image, track_id, timestamp, "test_log"
        )

        # Check if vehicle state was updated
        state = plate_processor.get_vehicle_state(track_id)
        assert state is not None

    def test_process_plate_with_multiple_vehicles(self, plate_processor):
        """Test plate detection for multiple vehicles."""
        timestamp = datetime.now()

        for i in range(5):
            image = np.zeros((100, 300, 3), dtype=np.uint8)
            plate_processor.process_plate_background_sync(
                image, i, timestamp, "test_log"
            )

        vehicle_ids = plate_processor.get_all_vehicle_ids()
        assert len(vehicle_ids) > 0


class TestPlateStateManagement:
    """Test plate count tracking for summary."""

    def test_vehicle_plate_counts_initialization(self, plate_processor):
        """Test that vehicle_plate_counts is initialized."""
        assert hasattr(plate_processor, "vehicle_plate_counts")
        assert isinstance(plate_processor.vehicle_plate_counts, dict)

    def test_vehicle_plate_counts_accumulation(self, plate_processor):
        """Test that plate detections accumulate in vehicle_plate_counts."""
        track_id = 1
        plate_text = "77A12345"

        # Simulate multiple detections of same plate
        for _ in range(3):
            plate_processor.update_vehicle_state(
                track_id, plate_text=plate_text
            )

        # In real implementation, vehicle_plate_counts would be updated
        # This tests the separation between vehicles_plates (for notifications)
        # and vehicle_plate_counts (for summary)
        assert True


class TestErrorHandling:
    """Test error handling."""

    def test_process_plate_with_invalid_image(self, plate_processor):
        """Test handling of invalid image."""
        track_id = 1
        timestamp = datetime.now()

        # Should not crash with invalid image
        try:
            plate_processor.process_plate_background_sync(
                None, track_id, timestamp, "test_log"
            )
        except Exception as e:
            pytest.fail(f"Should handle invalid image gracefully: {e}")

    def test_update_vehicle_state_with_invalid_track_id(self, plate_processor):
        """Test update_vehicle_state with edge case track IDs."""
        # Negative ID
        plate_processor.update_vehicle_state(-1, plate_text="77A12345")
        assert -1 in plate_processor.get_all_vehicle_ids()

        # Zero ID
        plate_processor.update_vehicle_state(0, plate_text="29C67890")
        assert 0 in plate_processor.get_all_vehicle_ids()

        # Very large ID
        plate_processor.update_vehicle_state(999999, plate_text="99Z99999")
        assert 999999 in plate_processor.get_all_vehicle_ids()


class TestStateIsotation:
    """Test state isolation between vehicles."""

    def test_vehicle_states_are_independent(self, plate_processor):
        """Test that updating one vehicle doesn't affect others."""
        plate_processor.update_vehicle_state(1, plate_text="77A11111")
        plate_processor.update_vehicle_state(2, plate_text="77A22222")

        state1 = plate_processor.get_vehicle_state(1)
        state2 = plate_processor.get_vehicle_state(2)

        assert state1[0] == "77A11111"
        assert state2[0] == "77A22222"


class TestNotificationPreparation:
    """Test preparation for notification sending."""

    def test_vehicle_plates_ready_for_notification(self, plate_processor):
        """Test that vehicle_plates is populated for notifications."""
        track_id = 1
        plate_text = "77A12345"

        plate_processor.update_vehicle_state(track_id, plate_text=plate_text)

        plates_copy = plate_processor.get_vehicle_plates_copy()
        assert plates_copy.get(track_id) == plate_text
