import logging
from concurrent.futures import ThreadPoolExecutor
import os
import warnings

# Set environment variables BEFORE importing any libraries to suppress warnings
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["PADDLE_EXTENSION_COMPILE_FLAG"] = "0"  # Disable ccache warning
os.environ["HF_HUB_OFFLINE"] = "1"  # Disable Hugging Face online checks
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logging
os.environ["GLOG_minloglevel"] = "2"  # Suppress glog (OneDNN)
os.environ["PADDLE_GLOG_LEVEL"] = "2"  # Suppress PaddlePaddle glog
os.environ["MKL_THREADING_LAYER"] = "GNU"  # Suppress OneDNN threading

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

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
from VehicleDetectionTracker.plate_processor import PlateProcessor, reset_telegram_sent
from VehicleDetectionTracker.frame_processor import FrameProcessor
from VehicleDetectionTracker.stream_handler import StreamHandler
from VehicleDetectionTracker.gpu_optimizer import GPUOptimizer, PerformanceMonitor

logging.getLogger("ultralytics").setLevel(logging.WARNING)
logging.getLogger("paddleocr").setLevel(logging.WARNING)
logging.getLogger("paddle").setLevel(logging.WARNING)

# Suppress FFmpeg/HEVC codec warnings
os.environ["FFREPORT"] = "file=/dev/null"
import cv2

cv2.setLogLevel(0)  # Disable OpenCV logging


class VehicleDetectionTracker:
    """Main vehicle detection and tracking class."""

    def __init__(
        self,
        initialize_all_models=True,
        use_gpu=None,
    ):
        """
        Initialize the VehicleDetection class.

        Args:
            initialize_all_models (bool): If True, initialize all models immediately.
        """
        log("Initializing Vehicle Detection Tracker...")

        # Load config
        self.config = get_config()
        paths_config = get_paths_config()
        display_config = get_display_config()
        advanced_config = get_advanced_config()

        # Override with parameters
        vehicle_model_path = paths_config.get("yolo_model", "yolov8n.pt")
        plate_model_path = paths_config.get("plate_model", "model/LP_detector.pt")
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

        # Initialize GPU optimizer for performance improvements
        self.gpu_optimizer = GPUOptimizer(device=self.device, use_gpu=self.use_gpu)
        
        # Set inference resolution for faster processing (optional)
        # Uncomment to enable resolution downscaling (huge FPS improvement)
        # self.gpu_optimizer.set_inference_resolution(1280, 720)  # Downscale to 1280x720 for inference
        # Supported resolutions:
        #   (1280, 720): ~18 FPS on RTSP (balanced)
        #   (960, 540):  ~25 FPS on RTSP (faster)
        #   (640, 360):  ~35 FPS on RTSP (very fast)
        
        self.performance_monitor = PerformanceMonitor()

        # Store paths for later use
        self.plate_model_path = plate_model_path

        # Load YOLO model
        log("Loading YOLO vehicle detection model...")
        self.model = YOLO(vehicle_model_path)
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
        max_workers = threading_config.get("max_workers", 5)
        log(f"Initializing thread pool with max_workers={max_workers}...")
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

        # Initialize OCR reader first (before creating plate_processor)
        self.ocr_reader = None

        # Initialize processors (with None models first)
        self.plate_model = None
        self.plate_processor = PlateProcessor(
            self.plate_model, self.ocr_reader, self._executor, log
        )
        self.frame_processor = FrameProcessor(self.model, log, gpu_optimizer=self.gpu_optimizer)
        
        # Link frame_processor to plate_processor for tracker reset on daily reset
        self.plate_processor.frame_processor = self.frame_processor
        
        # Resume track ID counter from today's state so a restart within the same day
        # does not reuse IDs that already belong to recorded vehicles.
        self.plate_processor.sync_track_id_counter()
        
        self.stream_handler = StreamHandler(log, self.plate_processor)

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
        
        # Log today's total vehicles
        self._log_today_vehicles_summary()

    def _initialize_plate_detector(self):
        """Initialize the license plate detector model."""
        try:
            device_str = "cuda:0" if self.use_gpu else "cpu"
            self.plate_model = initialize_plate_detector(
                self.plate_model_path, device=device_str
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

    def _log_today_vehicles_summary(self):
        """Log the total number of vehicles that entered today during initialization."""
        try:
            from VehicleDetectionTracker.vehicle_summary import get_today_vehicles_summary
            
            # Get vehicle state from plate_processor (thread-safe)
            vehicle_plates = self.plate_processor.get_vehicle_plates_copy()
            vehicle_directions = self.plate_processor.get_vehicle_directions_copy()
            vehicle_last_seen = self.plate_processor.get_vehicle_last_seen_copy()
            
            # Count all vehicles entering (direction contains "bottom")
            all_entering_vehicles = [
                tid for tid, direction in vehicle_directions.items()
                if "bottom" in direction.lower()
            ]
            total_all_entering = len(all_entering_vehicles)
            
            # Get today's vehicles summary (with plates detected)
            today_summary = get_today_vehicles_summary(
                vehicle_last_seen,
                vehicle_directions,
                vehicle_plates,
                log_func=log
            )
            
            # Filter out UNKNOWN plates (None values)
            today_summary_filtered = [
                (plate_text, count) for plate_text, count in today_summary
                if plate_text is not None
            ]
            
            total_with_plates = len(today_summary_filtered)
            total_without_plates = total_all_entering - total_with_plates
            
            # Display comprehensive summary
            log("\n" + "="*80)
            log(f"📊 TODAY'S VEHICLE SUMMARY")
            log("="*80)
            log(f"🚗 Total vehicles entered: {total_all_entering}")
            log(f"   ✓ With plates identified: {total_with_plates}")
            log(f"   ✗ Without plates: {total_without_plates}")
            log("-"*80)
            
            if total_with_plates > 0:
                log("📋 PLATE BREAKDOWN:")
                # Log each vehicle with its plate count (excluding UNKNOWN)
                for plate_text, count in today_summary_filtered:
                    log(f"   🚗 {plate_text:15s} : {count:2d} vehicle(s)")
            
            log("="*80 + "\n")
                
        except Exception as e:
            log(f"⚠ Error logging today's vehicles summary: {e}")
            import traceback
            log(traceback.format_exc())

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
        # === TIME BLOCK: OVERALL STREAMING ===
        # with time_block("[STREAM_TOTAL]", log):
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
        """Clean up resources and send all pending notifications."""
        import time
        import sys
        from VehicleDetectionTracker.plate_utils import get_inference_queue
        
        print("[CLEANUP] ⏳ Starting cleanup sequence...", flush=True)
        sys.stdout.flush()
        
        # First, wait for all pending background tasks to complete with their callbacks
        print("[CLEANUP] ⏳ Step 1: Waiting for all pending background tasks...", flush=True)
        sys.stdout.flush()
        if hasattr(self, "plate_processor") and self.plate_processor:
            result = self.plate_processor.wait_all_background_tasks(timeout=60)
            print(f"[CLEANUP] ✓ Background tasks wait returned: {result}", flush=True)
        
        # Send notifications for all vehicles with completed tasks
        # print("[CLEANUP] ⏳ Step 2: Sending notifications for all completed vehicles...", flush=True)
        # sys.stdout.flush()
        # if hasattr(self, "plate_processor") and self.plate_processor:
        #     self.plate_processor.send_notifications_for_completed_vehicles()
        
        # Shutdown the inference queue gracefully
        # This waits for all queued tasks + signals workers to exit (they're daemon threads)
        print("[CLEANUP] ⏳ Step 3: Shutting down inference queue gracefully...", flush=True)
        sys.stdout.flush()
        try:
            queue_instance = get_inference_queue()
            if queue_instance:
                queue_instance.shutdown()
                print("[CLEANUP] ✓ Inference queue shutdown complete (workers will exit as daemon threads)", flush=True)
        except Exception as e:
            print(f"[CLEANUP] ⚠ Error shutting down inference queue: {e}", flush=True)
        
        # Give threads a moment to complete
        print("[CLEANUP] ⏳ Step 4: Giving threads time to complete...", flush=True)
        sys.stdout.flush()
        time.sleep(0.5)
        
        # Now shut down the executor
        print("[CLEANUP] ⏳ Step 5: Shutting down thread executor...", flush=True)
        sys.stdout.flush()
        if hasattr(self, "_executor") and self._executor:
            self._executor.shutdown(wait=True)  # Wait for remaining tasks
            print("[CLEANUP] ✓ Executor shutdown complete", flush=True)
        
        sys.stdout.flush()
        
        # Clean up state
        if hasattr(self, "plate_processor") and self.plate_processor:
            self.plate_processor.vehicle_plates.clear()
        
        reset_telegram_sent()
        print("[CLEANUP] ✓✓ CLEANUP COMPLETE - Now exiting...", flush=True)
        sys.stdout.flush()


