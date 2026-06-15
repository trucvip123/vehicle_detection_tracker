import cv2
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
import threading
import queue
from VehicleDetectionTracker.logging_utils import log as shared_log


# ============================================================================
# Queue-based Inference System to avoid model_lock bottleneck
# ============================================================================

class InferenceQueue:
    """Queue-based inference system to prevent lock contention."""
    
    def __init__(self, num_workers=6):
        """Initialize inference queue with worker threads.
        
        Args:
            num_workers (int): Number of worker threads for inference
        """
        self.task_queue = queue.Queue()
        self.num_workers = num_workers
        self.workers = []
        self.running = True
        self._start_workers()
        _log(f"[INFERENCE_QUEUE] Initialized with {num_workers} worker threads")
    
    def _start_workers(self):
        """Start worker threads."""
        for i in range(self.num_workers):
            # Use daemon=True so they don't block program exit
            # But we'll explicitly wait for queue to drain in shutdown()
            worker = threading.Thread(target=self._worker_loop, daemon=True, name=f"InferenceWorker-{i}")
            worker.start()
            self.workers.append(worker)
            _log(f"[INFERENCE_QUEUE] Started worker thread {i}: {worker.name} (daemon=True)")
    
    def _worker_loop(self):
        """Main loop for worker thread - processes tasks from queue."""
        worker_name = threading.current_thread().name
        _log(f"[INFERENCE_QUEUE] {worker_name} started")
        
        while self.running:
            try:
                # Get task from queue (block with timeout to allow graceful shutdown)
                task = self.task_queue.get(timeout=1.0)
                if task is None:  # Shutdown signal
                    _log(f"[INFERENCE_QUEUE] {worker_name} received shutdown signal")
                    break
                
                # Unpack task
                task_func, task_args, callback = task
                
                try:
                    _log(f"[INFERENCE_QUEUE] {worker_name} executing task...")
                    # Execute the task
                    result = task_func(*task_args)
                    _log(f"[INFERENCE_QUEUE] {worker_name} task executed, result={result if result is None else 'plate_detected'}")
                    # Call callback with result
                    if callback:
                        _log(f"[INFERENCE_QUEUE] {worker_name} calling callback...")
                        callback(result)
                        _log(f"[INFERENCE_QUEUE] {worker_name} callback completed")
                except Exception as e:
                    _log(f"[INFERENCE_QUEUE] ⚠ {worker_name} Error in worker task: {e}")
                    import traceback
                    _log(f"[INFERENCE_QUEUE] {worker_name} Traceback: {traceback.format_exc()}")
                    if callback:
                        try:
                            callback(None)
                        except Exception as cb_err:
                            _log(f"[INFERENCE_QUEUE] ⚠ {worker_name} Callback error: {cb_err}")
                finally:
                    self.task_queue.task_done()
            
            except queue.Empty:
                continue
            except Exception as e:
                _log(f"[INFERENCE_QUEUE] ⚠ {worker_name} Worker error: {e}")
        
        _log(f"[INFERENCE_QUEUE] {worker_name} stopped")
    
    def submit_task(self, task_func, task_args, callback=None):
        """Submit a task to the inference queue.
        
        Args:
            task_func: Function to execute
            task_args: Tuple of arguments for task_func
            callback: Optional callback function called with result when task completes
        """
        self.task_queue.put((task_func, task_args, callback))
    
    def wait_for_all_tasks(self, timeout=None):
        """Wait for all queued tasks to complete.
        
        Args:
            timeout: Maximum time to wait (None = wait forever, as per queue.join())
            
        Returns:
            bool: True if all tasks completed, False if we need to give up
        """
        try:
            import sys
            queue_size = self.task_queue.qsize()
            _log(f"[INFERENCE_QUEUE] Waiting for {queue_size} tasks to complete (this blocks until all tasks are processed)...")
            sys.stdout.flush()
            
            self.task_queue.join()  # Blocks until all tasks are done (task_done called for each)
            
            _log(f"[INFERENCE_QUEUE] ✓ Queue.join() completed - all tasks are done")
            return True
        except Exception as e:
            _log(f"[INFERENCE_QUEUE] ⚠ Error waiting for tasks: {e}")
            return False
    
    def shutdown(self):
        """Shutdown the inference queue gracefully.
        
        This will:
        1. Wait for all pending tasks in queue to complete
        2. Send shutdown signals to workers
        3. Workers are daemon threads, so they'll exit when main thread exits
        """
        _log(f"[INFERENCE_QUEUE] Starting graceful shutdown...")
        
        # Step 1: Wait for all remaining tasks to complete
        _log(f"[INFERENCE_QUEUE] Step 1: Waiting for all queued tasks to complete...")
        self.wait_for_all_tasks()
        
        # Step 2: Signal workers to stop
        _log(f"[INFERENCE_QUEUE] Step 2: Signaling workers to stop...")
        self.running = False
        
        # Step 3: Send None signals for clean shutdown
        for _ in range(self.num_workers):
            self.task_queue.put(None)
        
        _log(f"[INFERENCE_QUEUE] Shutdown initiated - daemon workers will exit with main thread")
        _log(f"[INFERENCE_QUEUE] ✓ Shutdown complete")


# Global inference queue instance
_inference_queue = None


def initialize_inference_queue(num_workers=6):
    """Initialize the global inference queue."""
    global _inference_queue
    if _inference_queue is None:
        _inference_queue = InferenceQueue(num_workers=num_workers)
    return _inference_queue


def get_inference_queue():
    """Get the global inference queue instance."""
    global _inference_queue
    if _inference_queue is None:
        _inference_queue = InferenceQueue(num_workers=6)
    return _inference_queue


# ============================================================================
# Batch Accumulator System for Async Batch Processing
# ============================================================================

class BatchAccumulator:
    """
    Accumulates plate detection requests from multiple frames and processes them in batches.
    Provides both size-triggered and time-triggered batch processing for optimal throughput.
    """
    
    def __init__(self, 
                 batch_size=8,
                 time_threshold_ms=500,
                 max_batch_wait_ms=2000,
                 log_func=None):
        """Initialize batch accumulator.
        
        Args:
            batch_size (int): Target batch size for triggering OCR (default: 8)
            time_threshold_ms (int): Max time to wait before processing partial batch (ms)
            max_batch_wait_ms (int): Absolute max wait time before forced flush (ms)
            log_func: Logging function (uses _log if None)
        """
        self.batch_size = batch_size
        self.time_threshold_ms = time_threshold_ms
        self.max_batch_wait_ms = max_batch_wait_ms
        self.log_func = log_func or _log
        
        # Batch storage: {accumulator_id: {track_id: plate_data}}
        self.pending_batch = {}  # Current batch being accumulated
        self.batch_lock = threading.Lock()
        self.batch_event = threading.Event()  # Signaled when batch ready
        
        # Timing
        self.batch_start_time = None
        self.last_flush_time = datetime.now()
        
        # Processing state
        self.is_processing = False
        self.total_processed = 0
        self.total_batches = 0
        
        self.log_func(f"[BATCH_ACCUM] Initialized: batch_size={batch_size}, "
                     f"time_threshold={time_threshold_ms}ms, max_wait={max_batch_wait_ms}ms")
    
    def add_detection(self, track_id: int, vehicle_frame: np.ndarray, 
                     timestamp_str: str, vehicle_dir: str, 
                     direction: str, timestamp) -> bool:
        """Add a detected vehicle to the batch accumulator.
        
        Args:
            track_id: Vehicle tracking ID
            vehicle_frame: Extracted vehicle frame
            timestamp_str: Timestamp string for logging
            vehicle_dir: Output directory for screenshots
            direction: Direction label
            timestamp: Frame timestamp
            
        Returns:
            bool: True if batch was triggered, False otherwise
        """
        with self.batch_lock:
            # Initialize batch start time on first item
            if not self.pending_batch and self.batch_start_time is None:
                self.batch_start_time = datetime.now()
            
            # Add to batch
            self.pending_batch[track_id] = {
                'frame': vehicle_frame.copy(),
                'timestamp_str': timestamp_str,
                'vehicle_dir': vehicle_dir,
                'direction': direction,
                'timestamp': timestamp
            }
            
            batch_size_now = len(self.pending_batch)
            elapsed_ms = (datetime.now() - self.batch_start_time).total_seconds() * 1000
            
            self.log_func(f"[BATCH_ACCUM] Added track_id={track_id}, batch_size={batch_size_now}/{self.batch_size}, "
                         f"elapsed={elapsed_ms:.0f}ms/{self.time_threshold_ms}ms")
            
            # Check if batch should be processed
            should_process = (
                batch_size_now >= self.batch_size or  # Size threshold reached
                elapsed_ms >= self.time_threshold_ms   # Time threshold reached
            )
            
            if should_process:
                self.batch_event.set()
                self.log_func(f"[BATCH_ACCUM] ✓ Batch trigger: size={batch_size_now} or time={elapsed_ms:.0f}ms")
                return True
            
            return False
    
    def get_batch(self, wait_timeout_ms=None) -> dict:
        """Get and clear pending batch.
        
        Args:
            wait_timeout_ms: Wait up to this many milliseconds for batch to fill
            
        Returns:
            dict: Current batch or empty dict if not ready
        """
        with self.batch_lock:
            if wait_timeout_ms and not self.pending_batch:
                self.batch_event.clear()
                # Wait outside lock to avoid deadlock
            
            result = self.pending_batch.copy()
            self.pending_batch.clear()
            self.batch_start_time = None
            self.batch_event.clear()
            
            if result:
                self.log_func(f"[BATCH_ACCUM] Batch retrieved: size={len(result)}")
            
            return result
    
    def flush(self) -> dict:
        """Force flush current batch regardless of size.
        
        Returns:
            dict: Current batch (even if partial)
        """
        with self.batch_lock:
            result = self.pending_batch.copy()
            
            if result:
                elapsed = (datetime.now() - self.batch_start_time).total_seconds() * 1000 if self.batch_start_time else 0
                self.log_func(f"[BATCH_ACCUM] ✓✓ Batch flushed: size={len(result)}, elapsed={elapsed:.0f}ms")
            
            self.pending_batch.clear()
            self.batch_start_time = None
            self.batch_event.clear()
            
            return result
    
    def get_batch_stats(self) -> dict:
        """Get current batch statistics.
        
        Returns:
            dict with batch info
        """
        with self.batch_lock:
            elapsed_ms = (datetime.now() - self.batch_start_time).total_seconds() * 1000 if self.batch_start_time else 0
            return {
                'pending_items': len(self.pending_batch),
                'elapsed_ms': elapsed_ms,
                'total_processed': self.total_processed,
                'total_batches': self.total_batches,
                'is_processing': self.is_processing
            }


# Global batch accumulator instance
_batch_accumulator = None


def initialize_batch_accumulator(batch_size=8, time_threshold_ms=500):
    """Initialize the global batch accumulator."""
    global _batch_accumulator
    if _batch_accumulator is None:
        _batch_accumulator = BatchAccumulator(
            batch_size=batch_size,
            time_threshold_ms=time_threshold_ms
        )
    return _batch_accumulator


def get_batch_accumulator():
    """Get the global batch accumulator instance."""
    global _batch_accumulator
    if _batch_accumulator is None:
        _batch_accumulator = BatchAccumulator()
    return _batch_accumulator


def _ensure_log_dir():
    """Ensure logs directory exists."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    return log_dir


def _log(message):
    """Print log message with datetime timestamp and save to file."""
    shared_log(message)


def initialize_plate_detector(model_path="model/LP_detector.pt", device=None):
    """Load and return the license plate detector (YOLOv8/YOLOv5) or None on error.

    Args:
        model_path (str): Path to the model file (YOLOv8 or YOLOv5 format)
        device (str): Device to use ('cuda' or 'cpu'). If None, auto-detect.
    """
    try:
        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda:0":
            # Convert 'cuda:0' to 'cuda' for compatibility
            device = "cuda"
        
        # Verify model file exists
        model_file = Path(model_path)
        if not model_file.exists():
            _log(f"[PLATE_DETECTOR] ⚠ Model file not found: {model_path}, trying alternative path...")
            # Try alternative path with default name
            model_file = Path("model/license_plate_detector.pt")
            if model_file.exists():
                model_path = str(model_file)
            else:
                raise FileNotFoundError(f"Model file not found at {model_path} or model/license_plate_detector.pt")

        # Try loading as YOLOv8 model first
        _log(f"[PLATE_DETECTOR] Loading model from {model_path} on device={device}...")
        try:
            plate_model = YOLO(model_path)
            plate_model.to(device)
            
            if device == "cuda" and torch.cuda.is_available():
                _log(
                    f"[PLATE_DETECTOR] ✓ YOLOv8 model loaded and moved to GPU: {torch.cuda.get_device_name(0)}"
                )
            else:
                _log("[PLATE_DETECTOR] ✓ YOLOv8 model loaded (using CPU)")
            
            return plate_model
            
        except Exception as yolov8_error:
            # Fallback: Try loading as YOLOv5 model using torch.hub
            _log(f"[PLATE_DETECTOR] ⚠ YOLOv8 loading failed: {str(yolov8_error)[:100]}...")
            _log(f"[PLATE_DETECTOR] Attempting YOLOv5 fallback loading...")
            
            try:
                # Add yolov5 directory to sys.path for local loading
                import sys
                yolov5_path = Path(__file__).parent.parent / "yolov5"
                if str(yolov5_path) not in sys.path:
                    sys.path.insert(0, str(yolov5_path))
                
                # Try torch.hub.load for YOLOv5
                plate_model = torch.hub.load(
                    "ultralytics/yolov5",
                    "custom",
                    path=model_path,
                    force_reload=False,
                    device=device,
                )
                
                _log(
                    f"[PLATE_DETECTOR] ✓ YOLOv5 model loaded (fallback method)"
                )
                return plate_model
                
            except Exception as yolov5_error:
                _log(f"[PLATE_DETECTOR] ⚠ YOLOv5 fallback also failed: {yolov5_error}")
                raise Exception(f"Failed to load model with both YOLOv8 and YOLOv5 methods. YOLOv8 error: {yolov8_error}, YOLOv5 error: {yolov5_error}")
    
    except Exception as e:
        _log(f"Error loading license plate model: {e}")
        return None


def preprocess_plate_image(plate_image):
    try:
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        denoised = cv2.fastNlMeansDenoising(thresh)
        kernel = np.ones((1, 1), np.uint8)
        dilated = cv2.dilate(denoised, kernel, iterations=1)
        return dilated
    except Exception as e:
        _log(f"Error in plate image preprocessing: {e}")
        return plate_image


def _sync_plate_inference(plate_model, square_frame, model_lock, size=None):
    """Run plate model synchronously in a thread-safe way and return results or None (YOLOv8)."""
    if plate_model is None:
        return None

    # Check if square_frame is valid
    if square_frame is None or square_frame.size == 0:
        _log("[PLATE_INFERENCE] ❌ square_frame is None or empty")
        return None

    # Load config if size not specified
    if size is None:
        try:
            from VehicleDetectionTracker.config_loader import get_plate_detection_config

            plate_config = get_plate_detection_config()
            size = plate_config.get(
                "image_size", 640
            )  # Default: 640 for YOLOv8
        except:
            size = 640

    try:
        with model_lock:
            # YOLOv8: inference with imgsz parameter
            results = plate_model.predict(square_frame, imgsz=size, verbose=False)
            # Returns a list of Results objects, take the first one
            return results[0] if results else None
    except Exception as e:
        _log(f"[PLATE_INFERENCE] ❌ Error during inference with imgsz={size}: {e}")
        # Fallback: try without size parameter (use default)
        try:
            with model_lock:
                results = plate_model.predict(square_frame, verbose=False)
                return results[0] if results else None
        except Exception as e2:
            _log(f"[PLATE_INFERENCE] ❌ Fallback inference also failed: {e2}")
            return None


def detect_license_plate_sync(
    plate_model,
    vehicle_frame,
    ocr_reader,
    model_lock,
    timestamp_str,
    vehicle_dir="screenshots",
    track_id=None,
):
    """Detect license plate synchronously with detailed logging for debugging (YOLOv8)."""
    try:
        _log(f"[PLATE_DETECT] vehicle_id={track_id} Bắt đầu detect license plate")

        # Check plate model
        if plate_model is None:
            _log(f"[PLATE_DETECT] vehicle_id={track_id} ❌ plate_model is None, return None")
            return {"text": None, "count": None}

        _log(
            f"[PLATE_DETECT] vehicle_id={track_id} Vehicle frame shape: {vehicle_frame.shape if vehicle_frame is not None else 'None'}"
        )

        # Run plate detection inference
        _log(f"[PLATE_DETECT] vehicle_id={track_id} Đang chạy plate model inference...")

        height, width = vehicle_frame.shape[:2]
        mid_height = height // 2

        # Crop nửa dưới
        bottom_half = vehicle_frame[mid_height:, :, :]

        # Lấy kích thước của bottom_half
        bottom_height, bottom_width = bottom_half.shape[:2]

        # Để thành hình vuông, dùng kích thước nhỏ hơn
        square_size = min(bottom_height, bottom_width)

        # Crop 2 bên để center
        left = (bottom_width - square_size) // 2
        right = left + square_size

        square_frame = bottom_half[:, left:right, :]

        results = _sync_plate_inference(plate_model, square_frame, model_lock)

        if results is None:
            _log(f"[PLATE_DETECT] vehicle_id={track_id} ❌ Inference results is None")
            return {"text": None, "count": None}

        # YOLOv8: Use boxes attribute instead of pred
        boxes = results.boxes
        num_detections = len(boxes) if boxes is not None else 0
        
        if num_detections == 0:
            _log(f"[PLATE_DETECT] vehicle_id={track_id} ❌ Không có detection nào")
            return {"text": None, "count": 0}

        _log(f"[PLATE_DETECT] vehicle_id={track_id} Raw detections: {num_detections}")
        
        # Log all detections
        for i, box in enumerate(boxes):
            bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            conf = float(box.conf[0])
            _log(f"[PLATE_DETECT] vehicle_id={track_id} Detection {i}: bbox={bbox}, confidence={conf:.3f}")

        # Get best detection (highest confidence)
        best_box = boxes[0]  # boxes are sorted by confidence by default
        confidence = float(best_box.conf[0])
        x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())

        _log(
            f"[PLATE_DETECT] vehicle_id={track_id} Best detection: bbox=({x1},{y1},{x2},{y2}), confidence={confidence:.3f}"
        )

        # Check plate length
        length_plate = x2 - x1
        height_plate = y2 - y1
        _log(
            f"[PLATE_DETECT] vehicle_id={track_id} Plate dimensions: width={length_plate}, height={height_plate}"
        )

        # Load config for plate detection
        try:
            from VehicleDetectionTracker.config_loader import (
                get_plate_detection_config,
            )

            plate_config = get_plate_detection_config()
            min_width = plate_config.get("min_width", 40)
            min_height = plate_config.get("min_height", 20)
            min_confidence = plate_config.get("min_confidence", 0.25)
        except:
            min_width = 40
            min_height = 20
            min_confidence = 0.25

        # Check confidence threshold
        if confidence < min_confidence:
            _log(
                f"[PLATE_DETECT] vehicle_id={track_id} ❌ Confidence quá thấp ({confidence:.3f} < {min_confidence}), return None"
            )
            return {"text": None, "count": num_detections}

        if length_plate < min_width:
            _log(
                f"[PLATE_DETECT] vehicle_id={track_id} ❌ Plate quá nhỏ (width={length_plate} < {min_width}), return None"
            )
            return {"text": None, "count": num_detections}

        if height_plate < min_height:
            _log(
                f"[PLATE_DETECT] vehicle_id={track_id} ❌ Plate quá nhỏ (height={height_plate} < {min_height}), return None"
            )
            return {"text": None, "count": num_detections}

        # Validate bbox coordinates against frame dimensions to avoid index out of bounds
        frame_height, frame_width = square_frame.shape[:2]
        x1 = max(0, min(x1, frame_width - 1))
        y1 = max(0, min(y1, frame_height - 1))
        x2 = max(x1 + 1, min(x2, frame_width))
        y2 = max(y1 + 1, min(y2, frame_height))

        # Extract plate image
        plate_image = square_frame[y1:y2, x1:x2]
        if plate_image.size == 0:
            _log(f"[PLATE_DETECT] vehicle_id={track_id} ❌ Plate image size = 0, return None")
            return {"text": None, "count": num_detections}

        _log(f"[PLATE_DETECT] vehicle_id={track_id} ✓ Extracted plate image shape: {plate_image.shape}")

        # Save plate image
        filename = f"{vehicle_dir}/license_frame_{timestamp_str}.png"
        cv2.imwrite(filename, plate_image)

        # Check OCR reader
        if ocr_reader is None:
            _log(f"[PLATE_DETECT] vehicle_id={track_id} ⚠ OCR reader is None, return bbox only")
            return {"text": None, "count": num_detections}

        # Try OCR with different deskew directions
        lp = "unknown"

        with model_lock:
            lp = ocr_reader.read_license_plate(plate_image)

        _log(
            f"[PLATE_DETECT] vehicle_id={track_id} ✓ Tìm thấy biển số: '{lp}'"
        )
        return {"text": lp, "count": num_detections}
    except Exception as e:
        _log(f"[PLATE_DETECT] vehicle_id={track_id} ❌ ERROR in license plate detection: {e}")
        import traceback

        _log(f"[PLATE_DETECT] vehicle_id={track_id} Traceback: {traceback.format_exc()}")
        return {"text": None, "count": None}


def submit_plate_detection_async(
    plate_model,
    vehicle_frame,
    ocr_reader,
    model_lock,
    timestamp_str,
    callback,
    vehicle_dir="screenshots",
    track_id=None,
):
    """
    Submit plate detection task to inference queue asynchronously.
    
    The callback will be called with the result dict: {"text": plate_text, "count": num_detections}
    This avoids model_lock bottleneck by queuing up tasks instead of blocking.
    
    Args:
        plate_model: YOLOv8 model
        vehicle_frame: Cropped vehicle frame
        ocr_reader: OCR reader instance
        model_lock: Threading lock for model access
        timestamp_str: Timestamp string for logging
        callback: Function called with result when detection completes: callback(result_dict)
        vehicle_dir: Directory to save plate images
        track_id: Vehicle track ID for logging
    """
    try:
        # Create a task function that wraps the sync detection
        def _detection_task():
            return detect_license_plate_sync(
                plate_model,
                vehicle_frame,
                ocr_reader,
                model_lock,
                timestamp_str,
                vehicle_dir=vehicle_dir,
                track_id=track_id,
            )
        
        # Get the inference queue and submit the task
        queue_instance = get_inference_queue()
        queue_instance.submit_task(_detection_task, (), callback)
        
        _log(f"[PLATE_DETECT] vehicle_id={track_id} ✓ Submitted to inference queue (queue_size={queue_instance.task_queue.qsize()})")
    except Exception as e:
        _log(f"[PLATE_DETECT] vehicle_id={track_id} ❌ Error submitting to inference queue: {e}")
        # Fallback: call callback with None
        if callback:
            callback({"text": None, "count": None})


# ============================================================================
# Batch Detection and OCR Functions for Approach 2
# ============================================================================

def batch_detect_license_plates(
    plate_model: YOLO,
    vehicle_frames_dict: dict,  # {track_id: frame}
    ocr_reader,
    model_lock,
    detection_config: dict
) -> dict:  # {track_id: {text, count}}
    """
    Batch detect license plates in multiple vehicles and run OCR in batch.
    
    This is the core function for Approach 2 batch processing.
    Optimized for multi-GPU/multi-core processing.
    
    Args:
        plate_model: YOLOv8 model for plate detection
        vehicle_frames_dict: Dict of {track_id: vehicle_frame}
        ocr_reader: OCR reader instance
        model_lock: Thread lock for model inference
        detection_config: Config dict with detection parameters
    
    Returns:
        Dict of {track_id: {text, count, boxes}} with detection and OCR results
    """
    start_time = datetime.now()
    batch_results = {}
    
    _log(f"[BATCH_DETECT] Starting batch processing: {len(vehicle_frames_dict)} vehicles")
    
    try:
        # Step 1: Detect plates in all vehicles
        _log(f"[BATCH_DETECT] Step 1: Running plate detection on {len(vehicle_frames_dict)} vehicles...")
        
        plate_detections = {}  # {track_id: {boxes, num_detections, frame}}
        detection_time_start = datetime.now()
        
        with model_lock:
            for track_id, frame in vehicle_frames_dict.items():
                try:
                    _log(f"[BATCH_DETECT]   track_id={track_id} Running detection...")
                    
                    results = _sync_plate_inference(plate_model, frame, model_lock)
                    
                    if results is None:
                        plate_detections[track_id] = {
                            'boxes': None,
                            'num_detections': 0,
                            'frame': frame
                        }
                        continue
                    
                    boxes = results.boxes
                    num_detections = len(boxes) if boxes is not None else 0
                    
                    plate_detections[track_id] = {
                        'boxes': boxes,
                        'num_detections': num_detections,
                        'frame': frame
                    }
                    
                    _log(f"[BATCH_DETECT]   track_id={track_id} ✓ Detected {num_detections} plates")
                except Exception as det_err:
                    _log(f"[BATCH_DETECT]   track_id={track_id} ❌ Detection error: {det_err}")
                    plate_detections[track_id] = {
                        'boxes': None,
                        'num_detections': 0,
                        'frame': frame
                    }
        
        detection_time = (datetime.now() - detection_time_start).total_seconds() * 1000
        _log(f"[BATCH_DETECT] Step 1 complete: Detection took {detection_time:.1f}ms")
        
        # Step 2: Extract best plate crops from all detections
        _log(f"[BATCH_DETECT] Step 2: Extracting plate crops...")
        
        plate_crops_dict = {}  # {track_id: plate_image}
        metadata_dict = {}     # {track_id: {num_detections, confidence}}
        
        for track_id, detection in plate_detections.items():
            boxes = detection['boxes']
            num_detections = detection['num_detections']
            frame = detection['frame']
            
            if num_detections > 0 and boxes is not None:
                try:
                    # Get best detection (highest confidence)
                    best_box = boxes[0]
                    confidence = float(best_box.conf[0])
                    x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
                    
                    # Validate and clip coordinates
                    frame_height, frame_width = frame.shape[:2]
                    x1 = max(0, min(x1, frame_width - 1))
                    y1 = max(0, min(y1, frame_height - 1))
                    x2 = max(x1 + 1, min(x2, frame_width))
                    y2 = max(y1 + 1, min(y2, frame_height))
                    
                    # Extract plate
                    plate_image = frame[y1:y2, x1:x2]
                    
                    if plate_image.size > 0:
                        plate_crops_dict[track_id] = plate_image
                        metadata_dict[track_id] = {
                            'num_detections': num_detections,
                            'confidence': confidence,
                            'bbox': (x1, y1, x2, y2)
                        }
                        _log(f"[BATCH_DETECT]   track_id={track_id} ✓ Extracted plate crop (conf={confidence:.3f})")
                    else:
                        metadata_dict[track_id] = {
                            'num_detections': num_detections,
                            'confidence': confidence,
                            'bbox': None
                        }
                        _log(f"[BATCH_DETECT]   track_id={track_id} ⚠ Plate crop empty")
                except Exception as extract_err:
                    _log(f"[BATCH_DETECT]   track_id={track_id} ❌ Crop extraction error: {extract_err}")
                    metadata_dict[track_id] = {
                        'num_detections': num_detections,
                        'confidence': 0,
                        'bbox': None
                    }
            else:
                metadata_dict[track_id] = {
                    'num_detections': num_detections,
                    'confidence': 0,
                    'bbox': None
                }
        
        _log(f"[BATCH_DETECT] Step 2 complete: Extracted {len(plate_crops_dict)} plate crops")
        
        # Step 3: Batch OCR all detected plates
        _log(f"[BATCH_DETECT] Step 3: Running batch OCR on {len(plate_crops_dict)} plates...")
        
        ocr_time_start = datetime.now()
        batch_ocr_results = _batch_ocr_plates(ocr_reader, plate_crops_dict, model_lock)
        ocr_time = (datetime.now() - ocr_time_start).total_seconds() * 1000
        
        _log(f"[BATCH_DETECT] Step 3 complete: OCR took {ocr_time:.1f}ms")
        
        # Step 4: Combine results
        _log(f"[BATCH_DETECT] Step 4: Combining results...")
        
        for track_id in vehicle_frames_dict.keys():
            if track_id in plate_crops_dict:
                ocr_text = batch_ocr_results.get(track_id, "unknown")
                num_detections = metadata_dict[track_id]['num_detections']
                batch_results[track_id] = {
                    'text': ocr_text,
                    'count': num_detections,
                    'confidence': metadata_dict[track_id]['confidence']
                }
                _log(f"[BATCH_DETECT]   track_id={track_id} Result: text='{ocr_text}', count={num_detections}")
            else:
                # No valid plate crops
                batch_results[track_id] = {
                    'text': None,
                    'count': metadata_dict.get(track_id, {}).get('num_detections', 0),
                    'confidence': 0
                }
        
        total_time = (datetime.now() - start_time).total_seconds() * 1000
        _log(f"[BATCH_DETECT] ✓ Batch processing complete: {len(batch_results)} vehicles, "
             f"detection={detection_time:.1f}ms, ocr={ocr_time:.1f}ms, total={total_time:.1f}ms")
        
        return batch_results
        
    except Exception as e:
        _log(f"[BATCH_DETECT] ❌ Error in batch processing: {e}")
        import traceback
        _log(f"[BATCH_DETECT] Traceback: {traceback.format_exc()}")
        
        # Return partial results for vehicles we did process
        return batch_results


def _batch_ocr_plates(ocr_reader, plate_crops_dict: dict, model_lock) -> dict:
    """
    Run OCR on multiple plate crops in batch.
    
    Args:
        ocr_reader: OCR reader instance
        plate_crops_dict: Dict of {track_id: plate_image}
        model_lock: Thread lock for model access
    
    Returns:
        Dict of {track_id: ocr_text}
    """
    results = {}
    
    if not ocr_reader:
        _log(f"[BATCH_OCR] ⚠ OCR reader is None, skipping batch OCR")
        for track_id in plate_crops_dict.keys():
            results[track_id] = "unknown"
        return results
    
    try:
        # Check if OCR reader supports batch processing
        if hasattr(ocr_reader, 'read_license_plate_batch'):
            _log(f"[BATCH_OCR] Using native batch OCR processing...")
            
            with model_lock:
                plate_images = list(plate_crops_dict.values())
                track_ids = list(plate_crops_dict.keys())
                
                batch_results = ocr_reader.read_license_plate_batch(plate_images)
                
                for track_id, ocr_text in zip(track_ids, batch_results):
                    results[track_id] = ocr_text if ocr_text else "unknown"
                    _log(f"[BATCH_OCR]   track_id={track_id} Result: '{results[track_id]}'")
        else:
            # Fallback: Sequential OCR but keep lock contention minimal
            _log(f"[BATCH_OCR] OCR reader doesn't support batch, running sequential OCR under single lock...")
            
            with model_lock:
                for track_id, plate_image in plate_crops_dict.items():
                    try:
                        ocr_text = ocr_reader.read_license_plate(plate_image)
                        results[track_id] = ocr_text if ocr_text else "unknown"
                        _log(f"[BATCH_OCR]   track_id={track_id} Result: '{results[track_id]}'")
                    except Exception as ocr_err:
                        _log(f"[BATCH_OCR]   track_id={track_id} ❌ OCR error: {ocr_err}")
                        results[track_id] = "unknown"
        
        _log(f"[BATCH_OCR] ✓ Batch OCR complete: {len(results)} results")
        return results
        
    except Exception as e:
        _log(f"[BATCH_OCR] ❌ Error in batch OCR: {e}")
        import traceback
        _log(f"[BATCH_OCR] Traceback: {traceback.format_exc()}")
        
        # Return defaults for all
        for track_id in plate_crops_dict.keys():
            results[track_id] = "unknown"
        
        return results
