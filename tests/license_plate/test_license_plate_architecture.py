import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from tempfile import TemporaryDirectory

import numpy as np

from VehicleDetectionTracker.application.license_plate.batch_detect_license_plates import (
    BatchDetectLicensePlatesUseCase,
)
from VehicleDetectionTracker.application.license_plate.dto import PlateDetectionConfig
from VehicleDetectionTracker.domain.license_plate.entities import PlateBoundingBox
from VehicleDetectionTracker.infrastructure.license_plate.batch_accumulator import (
    BatchAccumulator,
)
from VehicleDetectionTracker.infrastructure.license_plate.inference_queue import (
    InferenceQueue,
)
from VehicleDetectionTracker.infrastructure.license_plate.ocr_service import (
    LicensePlateOCRService,
)
from VehicleDetectionTracker.infrastructure.license_plate.plate_image_processor import (
    PlateImageProcessor,
)
from VehicleDetectionTracker.infrastructure.license_plate.yolo_detector import (
    YOLOPlateDetector,
)
from VehicleDetectionTracker.plate_processor import PlateProcessor
from VehicleDetectionTracker.plate_utils import (
    batch_detect_license_plates,
    detect_license_plate_sync,
    submit_plate_detection_async,
)
import VehicleDetectionTracker.plate_utils as plate_utils


class DeferredFuture:
    def add_done_callback(self, callback):
        self.callback = callback

    def complete(self):
        self.callback(self)


class DeferredExecutor:
    def __init__(self):
        self.future = DeferredFuture()
        self.submissions = []

    def submit(self, function, argument):
        self.submissions.append((function, argument))
        return self.future


class AcceptingBatchAccumulator:
    def add_detection(self, *_args):
        return False


class BatchNotificationTests(unittest.TestCase):
    def create_processor(self):
        processor = PlateProcessor.__new__(PlateProcessor)
        processor.log = lambda *_args, **_kwargs: None
        processor._task_count_lock = threading.Lock()
        processor._state_lock = threading.RLock()
        processor.vehicle_pending_task_count = {}
        processor._batch_vehicle_track_ids = set()
        processor._batch_notification_finalized = set()
        processor._batch_notification_inflight = set()
        processor._batch_notification_last_attempt = {}
        processor.vehicle_plate_counts_each_frame = {}
        processor.vehicle_plates = {}
        processor.vehicle_directions = {}
        processor.vehicle_last_seen = {}
        processor.executor = DeferredExecutor()
        return processor

    def test_pending_count_tracks_dispatched_batches_not_accumulated_frames(self):
        processor = self.create_processor()
        processor.batch_enabled = True
        processor.batch_accumulator = AcceptingBatchAccumulator()
        vehicle_data = {
            "frame": np.zeros((8, 8, 3), dtype=np.uint8),
            "timestamp_str": "time",
            "vehicle_dir": "vehicle_dir",
            "direction": "bottom",
            "timestamp": None,
        }

        processor.submit_plate_processing_batch({163: vehicle_data})
        processor.submit_plate_processing_batch({163: vehicle_data})
        self.assertEqual(processor.vehicle_pending_task_count, {})

        processor._process_batch({163: vehicle_data})
        self.assertEqual(processor.vehicle_pending_task_count[163], 1)

        notified = []
        processor.send_final_vehicle_notification = lambda track_id, vehicle_dir=None: (
            notified.append(track_id) or True
        )
        processor.executor.future.complete()

        self.assertEqual(processor.vehicle_pending_task_count[163], 0)
        self.assertEqual(notified, [])

        processor.vehicle_plate_counts_each_frame[163] = {"77H-163.00": 1}
        processor.vehicle_last_seen[163] = datetime.now()
        import VehicleDetectionTracker.plate_processor as plate_processor_module

        vehicle_uuid = processor.get_or_create_uuid(163)
        with plate_processor_module._last_frame_time_lock:
            plate_processor_module._last_frame_received_time[163] = time.time() - 3
        with plate_processor_module._vehicle_telegram_sent_lock:
            plate_processor_module._vehicle_telegram_sent_with_plate.discard(vehicle_uuid)

        processor._check_and_notify_timed_out_vehicles()
        self.assertEqual(notified, [])
        notification_task, track_id = processor.executor.submissions[-1]
        notification_task(track_id)

        self.assertEqual(notified, [163])

    def test_batch_ocr_plate_is_available_to_final_plate_selection(self):
        processor = self.create_processor()

        processor._handle_batch_plate_result(
            163,
            {"text": "77H-046.12", "count": 1, "confidence": 0.721},
            "bottom",
            datetime(2026, 10, 8, 15, 36, 10),
            "screenshots/20261008/1536_163",
        )

        self.assertEqual(processor.get_most_detected_plate(163), ("77H-046.12", 1))

    def test_execute_batch_ocr_aggregates_every_frame_for_vehicle(self):
        processor = self.create_processor()
        processor.plate_model = FakePlateModel()
        processor.ocr_reader = FakeOCRReader()
        processor._model_lock = threading.Lock()
        processor.detection_config = {}
        frame = np.zeros((80, 160, 3), dtype=np.uint8)
        timestamp = datetime(2026, 10, 8, 16, 3, 11)
        frame_data = {
            "frame": frame,
            "direction": "Bottom",
            "timestamp": timestamp,
            "timestamp_str": "20261008_160311_000",
            "vehicle_dir": "screenshots/20261008/1603_163",
        }
        vehicle_data = {
            **frame_data,
            "frames": [dict(frame_data), dict(frame_data)],
        }

        results = processor._execute_batch_ocr({163: vehicle_data})

        self.assertEqual(len(results), 2)
        self.assertEqual(processor.get_most_detected_plate(163), ("77A1234", 2))


class FakeBox:
    def __init__(self, coordinates=(10, 10, 80, 40), confidence=0.9):
        self.xyxy = np.array([coordinates])
        self.conf = np.array([confidence])


class FakeResult:
    def __init__(self, boxes=None):
        self.boxes = [FakeBox()] if boxes is None else boxes


class FakePlateModel:
    def __init__(
        self,
        coordinates=(10, 10, 80, 40),
        confidence=0.9,
        no_detections=False,
        fail_batch=False,
    ):
        self.coordinates = coordinates
        self.confidence = confidence
        self.no_detections = no_detections
        self.fail_batch = fail_batch
        self.batch_calls = 0
        self.single_calls = 0

    def predict(self, image=None, **kwargs):
        if image is None:
            image = kwargs.get("source")
        if isinstance(image, list):
            self.batch_calls += 1
            if self.fail_batch:
                raise RuntimeError("intentional batch inference failure")
            return [self._predict_one(frame) for frame in image]
        self.single_calls += 1
        return [self._predict_one(image)]

    def _predict_one(self, image):
        if image[0, 0, 0] == 1:
            raise RuntimeError("intentional vehicle-specific inference failure")
        boxes = [] if self.no_detections else [FakeBox(self.coordinates, self.confidence)]
        return FakeResult(boxes)

    def to(self, _device):
        return self


class FakeOCRReader:
    def read_license_plate(self, _image):
        return "77A1234"


class TrueBatchUseCaseTests(unittest.TestCase):
    @staticmethod
    def create_use_case(model, lock):
        return BatchDetectLicensePlatesUseCase(
            YOLOPlateDetector(model, model_lock=lock, image_size=416, log_func=lambda _msg: None),
            LicensePlateOCRService(FakeOCRReader(), model_lock=lock, log_func=lambda _msg: None),
            PlateImageProcessor(),
            PlateDetectionConfig(image_size=416, use_true_batch=True),
            lambda _message: None,
        )

    def test_opt_in_uses_one_list_input_prediction(self):
        model = FakePlateModel()
        lock = threading.Lock()
        use_case = self.create_use_case(model, lock)
        frames = {
            track_id: np.zeros((80, 160, 3), dtype=np.uint8)
            for track_id in (201, 202)
        }

        results = use_case.execute(frames)

        self.assertEqual(model.batch_calls, 1)
        self.assertEqual(model.single_calls, 0)
        self.assertEqual(results[201]["text"], "77A1234")
        self.assertEqual(results[202]["text"], "77A1234")

    def test_batch_error_falls_back_per_vehicle(self):
        model = FakePlateModel(fail_batch=True)
        lock = threading.Lock()
        use_case = self.create_use_case(model, lock)
        failed_frame = np.zeros((80, 160, 3), dtype=np.uint8)
        failed_frame[0, 0, 0] = 1
        good_frame = np.zeros((80, 160, 3), dtype=np.uint8)

        results = use_case.execute({201: failed_frame, 202: good_frame})

        self.assertEqual(model.batch_calls, 1)
        self.assertEqual(results[201]["text"], None)
        self.assertEqual(results[201]["count"], 0)
        self.assertEqual(results[202]["text"], "77A1234")


class PlateImageProcessorTests(unittest.TestCase):
    def test_crop_detection_region_uses_centered_square_of_bottom_half(self):
        image = np.zeros((100, 160, 3), dtype=np.uint8)

        crop = PlateImageProcessor().crop_detection_region(image)

        self.assertEqual(crop.shape, (50, 50, 3))

    def test_crop_bounding_box_clips_coordinates(self):
        image = np.zeros((40, 60, 3), dtype=np.uint8)

        crop, clipped = PlateImageProcessor.crop_bounding_box(
            image,
            PlateBoundingBox(-5, -3, 80, 50),
        )

        self.assertEqual(clipped, PlateBoundingBox(0, 0, 60, 40))
        self.assertEqual(crop.shape, (40, 60, 3))


class PlateDetectionFacadeTests(unittest.TestCase):
    @staticmethod
    def run_sync(model, reader, vehicle_dir):
        return detect_license_plate_sync(
            model,
            np.zeros((120, 160, 3), dtype=np.uint8),
            reader,
            threading.Lock(),
            "20261008_143237_000",
            vehicle_dir=vehicle_dir,
            track_id=164,
        )

    def test_sync_detection_preserves_legacy_result_shape(self):
        with TemporaryDirectory() as vehicle_dir:
            result = self.run_sync(FakePlateModel(), FakeOCRReader(), vehicle_dir)

        self.assertEqual(result, {"text": "77A1234", "count": 1})

    def test_model_none_preserves_empty_result(self):
        with TemporaryDirectory() as vehicle_dir:
            result = self.run_sync(None, FakeOCRReader(), vehicle_dir)

        self.assertEqual(result, {"text": None, "count": None})

    def test_empty_frame_preserves_none_count(self):
        result = detect_license_plate_sync(
            FakePlateModel(),
            np.empty((0, 0, 3), dtype=np.uint8),
            FakeOCRReader(),
            threading.Lock(),
            "20261008_143237_000",
        )

        self.assertEqual(result, {"text": None, "count": None})

    def test_none_frame_preserves_none_count(self):
        with TemporaryDirectory() as vehicle_dir:
            result = detect_license_plate_sync(
                FakePlateModel(),
                None,
                FakeOCRReader(),
                threading.Lock(),
                "20261008_143237_000",
                vehicle_dir=vehicle_dir,
            )

        self.assertEqual(result, {"text": None, "count": None})

    def test_batch_empty_and_single_vehicle(self):
        lock = threading.Lock()
        model = FakePlateModel()
        reader = FakeOCRReader()

        self.assertEqual(batch_detect_license_plates(model, {}, reader, lock, {}), {})
        result = batch_detect_license_plates(
            model,
            {103: np.zeros((80, 160, 3), dtype=np.uint8)},
            reader,
            lock,
            {},
        )

        self.assertEqual(result[103]["text"], "77A1234")

    def test_async_facade_preserves_callback_contract(self):
        previous_queue = plate_utils._inference_queue
        inference_queue = InferenceQueue(num_workers=1, log_func=lambda _message: None)
        plate_utils._inference_queue = inference_queue
        callback_event = threading.Event()
        callback_results = []

        try:
            with TemporaryDirectory() as vehicle_dir:
                submit_plate_detection_async(
                    FakePlateModel(),
                    np.zeros((120, 160, 3), dtype=np.uint8),
                    FakeOCRReader(),
                    threading.Lock(),
                    "20261008_143237_000",
                    lambda result: (callback_results.append(result), callback_event.set()),
                    vehicle_dir=vehicle_dir,
                    track_id=164,
                )
                self.assertTrue(callback_event.wait(timeout=3))
        finally:
            inference_queue.shutdown()
            plate_utils._inference_queue = previous_queue

        self.assertEqual(callback_results, [{"text": "77A1234", "count": 1}])

    def test_no_detection_returns_zero_count(self):
        with TemporaryDirectory() as vehicle_dir:
            result = self.run_sync(
                FakePlateModel(no_detections=True),
                FakeOCRReader(),
                vehicle_dir,
            )

        self.assertEqual(result, {"text": None, "count": 0})

    def test_low_confidence_detection_is_rejected(self):
        with TemporaryDirectory() as vehicle_dir:
            result = self.run_sync(
                FakePlateModel(confidence=0.1),
                FakeOCRReader(),
                vehicle_dir,
            )

        self.assertEqual(result, {"text": None, "count": 1})

    def test_small_plate_detection_is_rejected(self):
        with TemporaryDirectory() as vehicle_dir:
            result = self.run_sync(
                FakePlateModel(coordinates=(10, 10, 30, 20)),
                FakeOCRReader(),
                vehicle_dir,
            )

        self.assertEqual(result, {"text": None, "count": 1})

    def test_missing_ocr_reader_returns_detection_count(self):
        with TemporaryDirectory() as vehicle_dir:
            result = self.run_sync(FakePlateModel(), None, vehicle_dir)

        self.assertEqual(result, {"text": None, "count": 1})

    def test_batch_with_regular_lock_is_not_deadlocked_and_isolates_errors(self):
        failed_frame = np.zeros((80, 160, 3), dtype=np.uint8)
        failed_frame[0, 0, 0] = 1
        good_frame = np.zeros((80, 160, 3), dtype=np.uint8)

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                batch_detect_license_plates,
                FakePlateModel(),
                {101: failed_frame, 102: good_frame},
                FakeOCRReader(),
                threading.Lock(),
                {},
            )
            results = future.result(timeout=3)

        self.assertEqual(results[101]["text"], None)
        self.assertEqual(results[101]["count"], 0)
        self.assertEqual(results[102]["text"], "77A1234")
        self.assertEqual(results[102]["count"], 1)


class BatchAccumulatorTests(unittest.TestCase):
    def test_repeated_track_id_retains_frames_and_counts_each_frame(self):
        accumulator = BatchAccumulator(
            batch_size=2,
            time_threshold_ms=10_000,
            log_func=lambda _message: None,
        )
        frame = np.zeros((8, 8, 3), dtype=np.uint8)

        first_triggered = accumulator.add_detection(
            7, frame, "first", "vehicle_dir", "bottom", None
        )
        second_triggered = accumulator.add_detection(
            7, frame, "second", "vehicle_dir", "bottom", None
        )
        stats = accumulator.get_batch_stats()
        batch = accumulator.flush()

        self.assertFalse(first_triggered)
        self.assertTrue(second_triggered)
        self.assertEqual(stats["pending_items"], 2)
        self.assertEqual(stats["pending_vehicles"], 1)
        self.assertEqual(len(batch[7]["frames"]), 2)
        self.assertEqual(batch[7]["timestamp_str"], "second")

    def test_get_batch_waits_for_timeout_and_returns_partial_items(self):
        accumulator = BatchAccumulator(
            batch_size=4,
            time_threshold_ms=10_000,
            log_func=lambda _message: None,
        )
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(accumulator.get_batch, 500)
            time.sleep(0.02)
            accumulator.add_detection(7, frame, "time", "vehicle_dir", "bottom", None)
            result = future.result(timeout=2)

        self.assertEqual(set(result), {7})

    def test_batch_size_trigger_and_flush(self):
        accumulator = BatchAccumulator(
            batch_size=1,
            time_threshold_ms=10_000,
            log_func=lambda _message: None,
        )
        frame = np.zeros((8, 8, 3), dtype=np.uint8)

        triggered = accumulator.add_detection(9, frame, "time", "vehicle_dir", "bottom", None)

        self.assertTrue(triggered)
        self.assertEqual(set(accumulator.flush()), {9})


class InferenceQueueTests(unittest.TestCase):
    def test_callable_task_and_callback_complete_before_shutdown(self):
        callback_event = threading.Event()
        callback_result = []
        inference_queue = InferenceQueue(
            num_workers=1,
            log_func=lambda _message: None,
        )

        inference_queue.submit_task(
            lambda: 42,
            (),
            lambda result: (callback_result.append(result), callback_event.set()),
        )
        self.assertTrue(callback_event.wait(timeout=2))
        self.assertTrue(inference_queue.wait_for_all_tasks())
        inference_queue.shutdown()

        self.assertEqual(callback_result, [42])


class PlateDetectionConfigTests(unittest.TestCase):
    def test_config_is_loaded_from_existing_mappings(self):
        config = PlateDetectionConfig.from_mapping(
            {"image_size": 416, "min_width": 44},
            {"batch_size": 12, "max_wait_time_ms": 2500},
        )

        self.assertEqual(config.image_size, 416)
        self.assertEqual(config.min_width, 44)
        self.assertEqual(config.batch_size, 12)
        self.assertEqual(config.max_batch_wait_ms, 2500)
        self.assertFalse(config.use_true_batch)

        enabled_config = PlateDetectionConfig.from_mapping(
            batch_values={"use_true_batch": True}
        )
        self.assertTrue(enabled_config.use_true_batch)


if __name__ == "__main__":
    unittest.main()