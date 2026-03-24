"""Tests for logging_utils module."""

import pytest
import os
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, mock_open, MagicMock
from VehicleDetectionTracker.logging_utils import (
    _ensure_log_dir,
    _write_to_log_file,
    log,
    log_plate,
)


class TestEnsureLogDir:
    """Test log directory creation."""

    def test_ensure_log_dir_creates_logs_folder(self, tmp_path):
        """Test that logs directory is created."""
        with patch("VehicleDetectionTracker.logging_utils.Path") as mock_path:
            mock_dir = MagicMock()
            mock_path.return_value = mock_dir
            mock_dir.mkdir = MagicMock()

            result = _ensure_log_dir()

            mock_dir.mkdir.assert_called_once_with(exist_ok=True)

    def test_ensure_log_dir_returns_path(self):
        """Test that _ensure_log_dir returns a Path object."""
        result = _ensure_log_dir()
        assert isinstance(result, Path)


class TestWriteToLogFile:
    """Test log file writing."""

    def test_write_to_log_file_success(self, tmp_path):
        """Test writing to log file."""
        log_file = tmp_path / "test.log"

        with patch("VehicleDetectionTracker.logging_utils._ensure_log_dir") as mock_log_dir:
            mock_log_dir.return_value = tmp_path
            _write_to_log_file("Test message", "app")

        assert log_file.exists()

    def test_write_to_log_file_appends_message(self, tmp_path):
        """Test that messages are appended to log file."""
        with patch("VehicleDetectionTracker.logging_utils._ensure_log_dir") as mock_log_dir:
            mock_log_dir.return_value = tmp_path
            _write_to_log_file("Message 1", "app")
            _write_to_log_file("Message 2", "app")

        # Verify both messages are in file
        log_files = list(tmp_path.glob("*.log"))
        assert len(log_files) > 0

    def test_write_to_log_file_handles_errors(self):
        """Test error handling when writing to log."""
        with patch("VehicleDetectionTracker.logging_utils._ensure_log_dir") as mock_log_dir:
            mock_log_dir.side_effect = Exception("Disk error")

            # Should not raise exception
            with patch("builtins.print") as mock_print:
                _write_to_log_file("Test message", "app")
                mock_print.assert_called()


class TestLog:
    """Test log function."""

    def test_log_prints_message(self):
        """Test that log prints to console."""
        with patch("builtins.print") as mock_print:
            with patch("VehicleDetectionTracker.logging_utils._write_to_log_file"):
                log("Test message", "app")
                mock_print.assert_called_once()

    def test_log_includes_timestamp(self):
        """Test that log includes timestamp."""
        with patch("builtins.print") as mock_print:
            with patch("VehicleDetectionTracker.logging_utils._write_to_log_file"):
                log("Test message", "app")
                call_args = mock_print.call_args[0][0]
                assert "[" in call_args and "]" in call_args

    def test_log_with_different_categories(self):
        """Test log with different categories."""
        with patch("builtins.print"):
            with patch("VehicleDetectionTracker.logging_utils._write_to_log_file"):
                log("Message 1", "app")
                log("Message 2", "plate")
                log("Message 3", "custom")


class TestLogPlate:
    """Test log_plate function."""

    def test_log_plate_includes_track_id(self):
        """Test that log_plate includes track_id."""
        with patch("builtins.print") as mock_print:
            with patch("VehicleDetectionTracker.logging_utils._write_to_log_file"):
                log_plate(42, "Plate detected", "plate")
                call_args = mock_print.call_args[0][0]
                assert "42" in call_args

    def test_log_plate_includes_message(self):
        """Test that log_plate includes the message."""
        with patch("builtins.print") as mock_print:
            with patch("VehicleDetectionTracker.logging_utils._write_to_log_file"):
                log_plate(1, "Test plate message", "plate")
                call_args = mock_print.call_args[0][0]
                assert "Test plate message" in call_args

    def test_log_plate_with_different_track_ids(self):
        """Test log_plate with various track IDs."""
        with patch("builtins.print"):
            with patch("VehicleDetectionTracker.logging_utils._write_to_log_file"):
                for track_id in [1, 42, 999]:
                    log_plate(track_id, f"Vehicle {track_id}", "plate")
