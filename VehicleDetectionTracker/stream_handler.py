"""Video streaming and processing handler."""
import cv2
import time
from datetime import datetime
from VehicleDetectionTracker.config_loader import get_rtsp_config
from VehicleDetectionTracker.time_scheduler import is_outside_operating_hours, get_time_info
from VehicleDetectionTracker.utils.send_bot import send_warning_to_telegram


class StreamHandler:
    """Handles video/camera stream processing."""
    
    def __init__(self, log_func):
        self.log = log_func
        self._stream_notify_sent = False
    
    def create_capture(self, video_path):
        """Create VideoCapture with optimized settings for RTSP streams."""
        cap = cv2.VideoCapture(video_path)

        # Optimize for RTSP streams
        if isinstance(video_path, str) and video_path.startswith("rtsp://"):
            rtsp_config = get_rtsp_config()
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, rtsp_config.get('open_timeout_ms', 5000))
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, rtsp_config.get('read_timeout_ms', 5000))

        return cap
    
    def process_video_stream(
        self,
        video_path,
        frame_processor,
        plate_processor,
        display_window=True,
        stream_frame_size=None,
        max_reconnect_attempts=None,
        reconnect_delay=None,
    ):
        """
        Process video/camera stream with optimized performance and auto-reconnect.

        Args:
            video_path: Path to video file or camera index
            frame_processor: FrameProcessor instance
            plate_processor: PlateProcessor instance
            display_window: Whether to display the video window
            stream_frame_size: Tuple (width, height) for frame resizing
            max_reconnect_attempts: Maximum reconnect attempts
            reconnect_delay: Delay between reconnects
        """
        rtsp_config = get_rtsp_config()
        if max_reconnect_attempts is None:
            max_reconnect_attempts = rtsp_config.get('max_reconnect_attempts', 10)
        if reconnect_delay is None:
            reconnect_delay = rtsp_config.get('reconnect_delay', 1)
        
        is_mp4_file = isinstance(video_path, str) and video_path.lower().endswith(".mp4")

        cap = None
        consecutive_failures = 0
        max_consecutive_failures = rtsp_config.get('max_consecutive_failures', 10)
        last_outside_hours_log = None

        while True:
            # Check operating hours
            if is_outside_operating_hours():
                time_info = get_time_info()
                current_time = time_info.get('current_time', 'unknown')
                
                now = datetime.now()
                if last_outside_hours_log is None or (now - last_outside_hours_log).total_seconds() > 3600:
                    self.log(f"⏱ Ngoài giờ vận hành ({current_time}). Chờ giờ vận hành...")
                    if cap is not None:
                        cap.release()
                        cap = None
                    last_outside_hours_log = now
                
                time.sleep(30)
                continue
            
            if last_outside_hours_log is not None:
                time_info = get_time_info()
                current_time = time_info.get('current_time', 'unknown')
                self.log(f"✓ Bắt đầu giờ vận hành ({current_time}). Khởi động xử lý stream...")
                last_outside_hours_log = None
            
            # Create or recreate capture
            if cap is None or not cap.isOpened():
                time.sleep(reconnect_delay)

                if cap is not None:
                    cap.release()

                cap = self.create_capture(video_path)

                if not cap.isOpened():
                    self.log(f"Không thể mở camera/video stream: {video_path}")
                    consecutive_failures += 1

                    try:
                        if not self._stream_notify_sent:
                            warn_msg = f"Không thể mở camera/video stream: {video_path} (attempt {consecutive_failures})"
                            self.log(warn_msg)
                            send_warning_to_telegram(warn_msg)
                            self._stream_notify_sent = True
                    except Exception as e:
                        self.log(f"Failed to send Telegram warning: {e}")

                    if consecutive_failures >= 20:
                        warn_msg = f"Không thể kết nối sau nhiều lần thử. Kiểm tra đường dẫn RTSP hoặc kết nối mạng."
                        self.log(warn_msg)
                        send_warning_to_telegram(warn_msg)
                        self._stream_notify_sent = True
                        break

            try:
                success, frame = cap.read()

                if not success or frame is None:
                    consecutive_failures += 1

                    if is_mp4_file:
                        self.log("Video playback completed.")
                        break

                    if consecutive_failures >= max_consecutive_failures:
                        cap.release()
                        cap = None
                        consecutive_failures = 0
                        continue

                    continue

                consecutive_failures = 0

                # Optionally resize the frame
                if stream_frame_size and frame is not None:
                    try:
                        frame = cv2.resize(
                            frame, stream_frame_size, interpolation=cv2.INTER_AREA
                        )
                    except Exception:
                        pass

                timestamp = datetime.now()

                # Process frame
                try:
                    display_frame = frame_processor.process_frame_streaming(frame, timestamp, plate_processor)
                except Exception as e:
                    self.log(f"Lỗi xử lý frame: {e}")
                    continue

                if display_window:
                    display_size = (1280, 720)
                    if display_frame is not None:
                        try:
                            display_frame_resized = cv2.resize(
                                display_frame, display_size, interpolation=cv2.INTER_AREA
                            )
                        except Exception:
                            display_frame_resized = display_frame
                    else:
                        display_frame_resized = display_frame
                    cv2.imshow("Vehicle Detection - Streaming Mode", display_frame_resized)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            except Exception as e:
                error_msg = str(e).lower()
                if "hevc" in error_msg or "codec" in error_msg or "ref with poc" in error_msg:
                    self.log(f"Lỗi codec (bỏ qua frame): {e}")
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        self.log("Quá nhiều lỗi codec liên tiếp. Đang thử kết nối lại...")
                        cap.release()
                        cap = None
                        consecutive_failures = 0
                    continue
                else:
                    self.log(f"Lỗi không mong muốn: {e}")
                    consecutive_failures += 1
                    if consecutive_failures >= 10:
                        self.log("Đang thử kết nối lại sau lỗi...")
                        cap.release()
                        cap = None
                        consecutive_failures = 0

        # Cleanup
        try:
            if cap is not None:
                cap.release()
            if display_window:
                cv2.destroyAllWindows()
        except Exception as e:
            self.log(f"Lỗi khi cleanup: {e}")
