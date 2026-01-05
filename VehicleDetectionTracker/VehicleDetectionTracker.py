import logging
from concurrent.futures import ThreadPoolExecutor
import os

from ultralytics import YOLO

from VehicleDetectionTracker.function.paddleocr_wrapper import create_paddleocr_reader
from VehicleDetectionTracker.plate_utils import initialize_plate_detector
from VehicleDetectionTracker.config_loader import (
    get_config,
    get_threading_config,
    get_paths_config,
    get_display_config,
    get_advanced_config,
)

from VehicleDetectionTracker.device_utils import get_device
from VehicleDetectionTracker.logging_utils import log
from VehicleDetectionTracker.excel_utils import ExcelManager
from VehicleDetectionTracker.plate_processor import PlateProcessor, reset_telegram_sent
from VehicleDetectionTracker.frame_processor import FrameProcessor
from VehicleDetectionTracker.stream_handler import StreamHandler
from VehicleDetectionTracker.image_utils_helper import draw_plate_text_corner

logging.getLogger("ultralytics").setLevel(logging.WARNING)

# Suppress FFmpeg/HEVC codec warnings
os.environ["FFREPORT"] = "file=/dev/null"
import cv2

cv2.setLogLevel(0)  # Disable OpenCV logging


class VehicleDetectionTracker:
    """Main vehicle detection and tracking class."""

    def __init__(
        self,
        model_path="yolov8n.pt",
        excel_output_path="vehicle_data.xlsx",
        initialize_all_models=True,
        use_gpu=None,
    ):
        """
        Initialize the VehicleDetection class.

        Args:
            model_path (str): Path to the YOLO model file.
            excel_output_path (str): Path to Excel file for saving vehicle data.
            initialize_all_models (bool): If True, initialize all models immediately.
        """
        log("Initializing Vehicle Detection Tracker...")

        # Load config
        self.config = get_config()
        paths_config = get_paths_config()
        display_config = get_display_config()
        advanced_config = get_advanced_config()

        # Override with parameters
        model_path = model_path or paths_config.get("yolo_model", "yolov8n.pt")
        excel_output_path = excel_output_path or paths_config.get(
            "excel_output", "vehicle_data.xlsx"
        )
        initialize_all_models = (
            initialize_all_models
            if initialize_all_models is not None
            else advanced_config.get("initialize_all_models", True)
        )

        # Get device (GPU/CPU), allow override by user
        if use_gpu is not None:
            self.device, self.use_gpu = ("cuda:0", True) if use_gpu else ("cpu", False)
        else:
            self.device, self.use_gpu = get_device(log)

        # Load YOLO model
        log("Loading YOLO vehicle detection model...")
        self.model = YOLO(model_path)
        if self.use_gpu:
            self.model.to(self.device)
            log(f"✓ YOLO model loaded on GPU")
        else:
            log("✓ YOLO model loaded (using CPU)")

        # Stream frame size
        self.stream_frame_size = display_config.get("stream_frame_size", None)
        log(f"stream_frame_size: {self.stream_frame_size}")

        # Thread pool for async operations
        threading_config = get_threading_config()
        max_workers = threading_config.get("max_workers", 4)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

        # Initialize Excel manager
        self.excel_manager = ExcelManager(excel_output_path, log)

        # Initialize OCR reader first (before creating plate_processor)
        self.ocr_reader = None

        # Initialize processors (with None models first)
        self.plate_model = None
        self.plate_processor = PlateProcessor(
            self.plate_model, self.ocr_reader, self._executor, log
        )
        self.frame_processor = FrameProcessor(self.model, log)
        self.stream_handler = StreamHandler(log)

        # Now initialize OCR reader if needed
        if initialize_all_models:
            log("Initializing OCR reader...")
            self._initialize_ocr_reader()
            log("✓ OCR reader initialized")
        else:
            log("⚠ OCR reader will be initialized on first use (lazy loading)")

        # Now initialize plate detector
        log("Loading license plate detector model...")
        self._initialize_plate_detector()
        log("✓ License plate detector loaded")

        log("✓ All initialization complete!")

    def _initialize_plate_detector(self):
        """Initialize the license plate detector model."""
        try:
            device_str = "cuda:0" if self.use_gpu else "cpu"
            self.plate_model = initialize_plate_detector(
                "model/LP_detector.pt", device=device_str
            )
            self.plate_processor.plate_model = self.plate_model
        except Exception as e:
            log(f"Error loading license plate model: {e}")
            self.plate_model = None

    def _initialize_ocr_reader(self):
        """Initialize OCR reader (PaddleOCR)."""
        if self.ocr_reader is None:
            self.ocr_reader = create_paddleocr_reader(
                lang="en", use_angle_cls=True, show_log=False, use_gpu=self.use_gpu
            )
            # Update plate_processor if it exists
            if hasattr(self, "plate_processor") and self.plate_processor:
                self.plate_processor.ocr_reader = self.ocr_reader

    def ensure_all_models_initialized(self):
        """Ensure all models are initialized."""
        if self.model is None:
            raise RuntimeError("YOLO model not initialized!")

        if self.plate_model is None:
            log("Warning: Plate detector not initialized, reinitializing...")
            self._initialize_plate_detector()

        if self.ocr_reader is None:
            log("Warning: OCR reader not initialized, initializing now...")
            self._initialize_ocr_reader()
            log("✓ OCR reader initialized")

    def get_initialization_status(self):
        """Get status of model initialization."""
        return {
            "yolo_model": self.model is not None,
            "plate_detector": self.plate_model is not None,
            "ocr_reader": self.ocr_reader is not None,
        }

    def process_frame_streaming(self, frame, frame_timestamp):
        """
        Optimized frame processing for streaming.

        Args:
            frame (numpy.ndarray): Input frame for processing.
            frame_timestamp (datetime): Timestamp of the frame.

        Returns:
            numpy.ndarray: Frame with license plates displayed.
        """
        display_frame = self.frame_processor.process_frame_streaming(
            frame, frame_timestamp, self.plate_processor
        )
        return draw_plate_text_corner(
            display_frame, self.plate_processor.vehicle_plates
        )

    def process_video_streaming(
        self,
        video_path,
        display_window=True,
        max_reconnect_attempts=None,
        reconnect_delay=None,
    ):
        """
        Process video/camera stream with auto-reconnect support.

        Args:
            video_path (str or int): Path to video file or camera index
            display_window (bool): Whether to display the video window
            max_reconnect_attempts (int): Maximum reconnect attempts
            reconnect_delay (int): Delay in seconds between reconnects
        """
        self.stream_handler.process_video_stream(
            video_path,
            self.frame_processor,
            self.plate_processor,
            display_window=display_window,
            stream_frame_size=self.stream_frame_size,
            max_reconnect_attempts=max_reconnect_attempts,
            reconnect_delay=reconnect_delay,
        )

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, "_executor") and self._executor:
            self._executor.shutdown(wait=True)
        self.plate_processor.vehicle_plates.clear()
        reset_telegram_sent()
