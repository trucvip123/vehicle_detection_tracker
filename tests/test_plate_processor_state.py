import threading
import unittest
from datetime import datetime

from VehicleDetectionTracker.plate_processor import PlateProcessor


class PlateProcessorStatePersistenceTest(unittest.TestCase):
    def test_entry_exit_times_are_serialized(self):
        processor = PlateProcessor.__new__(PlateProcessor)
        processor._state_lock = threading.RLock()
        processor.vehicle_entry_times = {}
        processor.vehicle_exit_times = {}
        processor.vehicle_plates = {}
        processor.vehicle_plate_counts = {}
        processor.vehicle_plate_counts_each_frame = {}
        processor.vehicle_directions = {}
        processor.vehicle_last_seen = {}
        processor.vehicle_detected_plate_images = {}
        processor.vehicle_pending_futures = {}
        processor.vehicle_pending_task_count = {}
        processor.vehicle_pending_queue_tasks = {}
        processor._vehicles_without_plate_logged = set()
        processor._task_count_lock = threading.Lock()
        processor.log = lambda *_args, **_kwargs: None

        entry_time = datetime(2026, 8, 5, 15, 57, 17)
        exit_time = datetime(2026, 8, 5, 15, 58, 10)

        processor.mark_vehicle_entry(7, entry_time)
        processor.mark_vehicle_exit(7, exit_time)

        state = processor._build_state_payload()

        self.assertIn("vehicle_entry_exit_times", state)
        self.assertEqual(state["vehicle_entry_exit_times"]["7"]["entry_time"], entry_time.isoformat())
        self.assertEqual(state["vehicle_entry_exit_times"]["7"]["exit_time"], exit_time.isoformat())


if __name__ == "__main__":
    unittest.main()
