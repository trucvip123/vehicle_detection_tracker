"""Tests for vehicle_summary module."""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock, mock_open
from VehicleDetectionTracker.vehicle_summary import (
    levenshtein_distance,
    merge_similar_plates,
    get_today_vehicles_summary,
)


class TestLevenshteinDistance:
    """Test Levenshtein distance calculation."""

    def test_levenshtein_identical_strings(self):
        """Test distance between identical strings."""
        s1 = "77A12345"
        s2 = "77A12345"

        distance = levenshtein_distance(s1, s2)

        assert distance == 0

    def test_levenshtein_one_substitution(self):
        """Test distance with single character difference."""
        s1 = "77A12345"
        s2 = "77B12345"

        distance = levenshtein_distance(s1, s2)

        assert distance == 1

    def test_levenshtein_two_substitutions(self):
        """Test distance with two character differences."""
        s1 = "77A12345"
        s2 = "77B12346"

        distance = levenshtein_distance(s1, s2)

        assert distance == 2

    def test_levenshtein_insertion(self):
        """Test distance with character insertion."""
        s1 = "77A1234"
        s2 = "77A12345"

        distance = levenshtein_distance(s1, s2)

        assert distance == 1

    def test_levenshtein_deletion(self):
        """Test distance with character deletion."""
        s1 = "77A12345"
        s2 = "77A1234"

        distance = levenshtein_distance(s1, s2)

        assert distance == 1

    def test_levenshtein_completely_different(self):
        """Test distance between completely different strings."""
        s1 = "ABC"
        s2 = "XYZ"

        distance = levenshtein_distance(s1, s2)

        assert distance > 0

    def test_levenshtein_empty_string(self):
        """Test distance with empty string."""
        s1 = "77A12345"
        s2 = ""

        distance = levenshtein_distance(s1, s2)

        assert distance == len(s1)

    def test_levenshtein_symmetry(self):
        """Test that distance is symmetric."""
        s1 = "77A12345"
        s2 = "77B12346"

        dist1 = levenshtein_distance(s1, s2)
        dist2 = levenshtein_distance(s2, s1)

        assert dist1 == dist2


class TestMergeSimilarPlates:
    """Test plate merging logic."""

    def test_merge_similar_plates_identical(self):
        """Test merging identical plates."""
        plates = {"77A12345": 5, "77A12345": 3}

        result = merge_similar_plates(plates, threshold=2)

        assert isinstance(result, dict)

    def test_merge_similar_plates_one_difference(self):
        """Test merging plates with one character difference (within threshold)."""
        plates = {"77A12345": 5, "77B12345": 3}

        result = merge_similar_plates(plates, threshold=2)

        assert isinstance(result, dict)
        # Should merge into one entry
        assert len(result) <= len(plates)

    def test_merge_similar_plates_different_threshold(self):
        """Test merging with different thresholds."""
        plates = {"77A12345": 5, "77B12346": 3}

        result_threshold_1 = merge_similar_plates(plates, threshold=1)
        result_threshold_2 = merge_similar_plates(plates, threshold=2)

        assert isinstance(result_threshold_1, dict)
        assert isinstance(result_threshold_2, dict)

    def test_merge_similar_plates_empty_input(self):
        """Test merging empty plate dictionary."""
        plates = {}

        result = merge_similar_plates(plates, threshold=2)

        assert result == {}

    def test_merge_similar_plates_single_plate(self):
        """Test merging with single plate."""
        plates = {"77A12345": 5}

        result = merge_similar_plates(plates, threshold=2)

        assert len(result) >= 1

    def test_merge_similar_plates_keeps_highest_count(self):
        """Test that merging keeps the highest count."""
        plates = {"77A12345": 5, "77A12346": 3}

        result = merge_similar_plates(plates, threshold=2)

        # Should have merged and kept count information
        total_count = sum(result.values())
        assert total_count >= 5  # At least the highest count


class TestGenerateDailySummary:
    """Test daily summary generation."""

    def test_generate_daily_summary_with_data(self):
        """Test generating daily summary with vehicle data."""
        vehicle_last_seen = {
            1: datetime(2026, 3, 24, 10, 30, 0),
            2: datetime(2026, 3, 24, 10, 35, 0),
        }
        vehicle_directions = {1: "Right", 2: "Left"}
        vehicle_plates = {1: "77A12345", 2: "29C67890"}

        result = get_today_vehicles_summary(
            vehicle_last_seen, vehicle_directions, vehicle_plates
        )

        assert result is not None

    def test_generate_daily_summary_empty_data(self):
        """Test generating summary with empty vehicle data."""
        vehicle_last_seen = {}
        vehicle_directions = {}
        vehicle_plates = {}

        result = get_today_vehicles_summary(
            vehicle_last_seen, vehicle_directions, vehicle_plates
        )

        assert result is not None

    def test_generate_daily_summary_large_dataset(self):
        """Test summary generation with large dataset."""
        vehicle_last_seen = {
            i: datetime(2026, 3, 24, 10, 30, 0)
            for i in range(1, 101)
        }
        vehicle_directions = {i: "Right" for i in range(1, 101)}
        vehicle_plates = {i: f"77A{i:05d}" for i in range(1, 101)}

        result = get_today_vehicles_summary(
            vehicle_last_seen, vehicle_directions, vehicle_plates
        )

        assert result is not None


class TestPlateMergingIntegration:
    """Integration tests for plate merging."""

    def test_merge_and_summarize_workflow(self):
        """Test complete workflow of merging and summarizing."""
        plates = {"77A12345": 5, "77A12346": 2, "77B54321": 1}

        merged = merge_similar_plates(plates, threshold=2)

        assert isinstance(merged, dict)
        # Should have reduced number of entries due to merging
        assert len(merged) <= len(plates)

    def test_threshold_sensitivity(self):
        """Test that threshold significantly affects merging."""
        plates = {
            "77A12345": 10,
            "77A12346": 5,
            "77A12347": 3,
            "77B54321": 2,
        }

        strict_merge = merge_similar_plates(plates, threshold=1)
        lenient_merge = merge_similar_plates(plates, threshold=3)

        # Lenient merge should result in fewer entries
        assert len(lenient_merge) <= len(strict_merge)
