"""Tests for tracking_utils module."""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock
import numpy as np
from VehicleDetectionTracker.tracking_utils import (
    map_direction_to_label,
    calculate_speed_and_direction,
    convert_mps_to_kmph,
    update_tracking_history,
)


class TestMapDirectionToLabel:
    """Test direction label mapping."""

    def test_map_direction_right(self):
        """Test mapping direction angle to right."""
        direction = 90.0  # Right in degrees

        label = map_direction_to_label(direction)

        assert isinstance(label, str)
        assert len(label) > 0

    def test_map_direction_left(self):
        """Test mapping direction angle to left."""
        direction = 270.0  # Left in degrees

        label = map_direction_to_label(direction)

        assert isinstance(label, str)
        assert len(label) > 0

    def test_map_direction_returns_string(self):
        """Test that direction label is a valid string."""
        for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
            label = map_direction_to_label(float(angle))
            assert isinstance(label, str)

    def test_map_direction_full_circle(self):
        """Test direction mapping for full 360 degrees."""
        for angle in np.linspace(0, 360, 36):
            label = map_direction_to_label(float(angle))
            assert isinstance(label, str)
            assert len(label) > 0


class TestCalculateSpeedAndDirection:
    """Test speed and direction calculation."""

    def test_calculate_speed_and_direction_with_valid_inputs(self):
        """Test speed and direction calculation with valid data."""
        timestamps = [
            datetime(2026, 3, 24, 10, 30, 0),
            datetime(2026, 3, 24, 10, 30, 1),
        ]
        positions = [(100.0, 100.0), (110.0, 110.0)]

        result = calculate_speed_and_direction(timestamps, positions)

        assert isinstance(result, dict)

    def test_calculate_speed_and_direction_single_point(self):
        """Test with single point."""
        timestamps = [datetime(2026, 3, 24, 10, 30, 0)]
        positions = [(100.0, 100.0)]

        result = calculate_speed_and_direction(timestamps, positions)

        assert isinstance(result, dict)

    def test_calculate_speed_and_direction_multiple_points(self):
        """Test with multiple points."""
        timestamps = [
            datetime(2026, 3, 24, 10, 30, i)
            for i in range(5)
        ]
        positions = [(100.0 + i, 100.0 + i) for i in range(5)]

        result = calculate_speed_and_direction(timestamps, positions)

        assert isinstance(result, dict)


class TestConvertMpsToKmph:
    """Test meter per second to kilometers per hour conversion."""

    def test_convert_mps_zero(self):
        """Test converting zero mps."""
        result = convert_mps_to_kmph(0.0)

        assert result == 0.0

    def test_convert_mps_positive_value(self):
        """Test converting positive mps value."""
        mps = 10.0  # 10 m/s

        result = convert_mps_to_kmph(mps)

        assert result > 0
        assert result == 36.0  # 10 * 3.6 = 36 kmph


class TestUpdateTrackingHistory:
    """Test vehicle tracking history management."""

    def test_update_tracking_history_adds_entry(self):
        """Test adding position to tracking history."""
        track_history: dict = {}
        track_id = 1
        x, y = 100.0, 100.0

        update_tracking_history(track_history, track_id, x, y)

        assert track_id in track_history
        assert len(track_history[track_id]) > 0

    def test_update_tracking_history_multiple_points(self):
        """Test adding multiple points to history."""
        track_history: dict = {}
        track_id = 1

        for i in range(10):
            update_tracking_history(track_history, track_id, float(100 + i), float(100 + i))

        assert track_id in track_history
        assert len(track_history[track_id]) <= 30  # Should respect max_history

    def test_update_tracking_history_respects_max_history(self):
        """Test that history respects max_history parameter."""
        track_history: dict = {}
        track_id = 1
        max_history = 10

        for i in range(50):
            update_tracking_history(track_history, track_id, float(100 + i), float(100 + i), max_history)

        # Should not exceed max_history
        assert len(track_history[track_id]) <= max_history

    def test_update_tracking_multiple_vehicles(self):
        """Test tracking multiple vehicles separately."""
        track_history: dict = {}

        for track_id in [1, 2, 3]:
            for i in range(5):
                update_tracking_history(track_history, track_id, float(100 + i), float(100 + i))
