"""Video streaming and processing handler."""

import cv2
import time
import os
import sys
import logging
import torch
from datetime import datetime
from VehicleDetectionTracker.config_loader import get_rtsp_config
from VehicleDetectionTracker.time_scheduler import (
    is_outside_operating_hours,
    get_time_info,
)
from VehicleDetectionTracker.utils.send_bot import send_warning_to_telegram

# Suppress FFmpeg warnings completely
os.environ["FFREPORT"] = "file=/dev/null"
os.environ["FFMPEG_SUPPRESS_LOG_LEVEL"] = "quiet"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "quiet"
# Enable GPU acceleration for FFmpeg
os.environ["OPENCV_FFMPEG_CUVID_DECODER"] = "1"
cv2.setLogLevel(0)
logging.getLogger("cv2").setLevel(logging.WARNING)


# Redirect stderr to suppress FFmpeg codec messagesân
class NullWriter:
    def write(self, s):
        pass

    def flush(self):
        pass


_original_stderr = sys.stderr


class StreamHandler:
    """Handles video/camera stream processing."""

    def __init__(self, log_func, plate_processor=None):
        self.log = log_func
        self.plate_processor = plate_processor
        self._stream_notify_sent = False
        self._last_daily_summary_time = None  # Track last time daily summary was sent

    def create_capture(self, video_path):
        """Create VideoCapture with GPU hardware decoding for RTSP streams."""
        # Suppress FFmpeg warnings during VideoCapture creation
        sys.stderr = NullWriter()
        try:
            # Enable GPU decoding with CUVID (NVIDIA)
            cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)

            # Enable hardware acceleration if available
            try:
                # Try to enable hardware acceleration (D3D11 on Windows)
                cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_D3D11)
                # self.log("✓ Hardware acceleration enabled for stream decoding")
            except AttributeError:
                # VIDEO_ACCELERATION_D3D11 not available, try generic method
                try:
                    cap.set(cv2.CAP_PROP_HW_ACCELERATION, 1)
                    self.log("✓ Hardware acceleration enabled for stream decoding")
                except Exception as e:
                    self.log(f"Note: Hardware acceleration not available: {e}")

            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        finally:
            # Restore stderr
            sys.stderr = _original_stderr

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
            max_reconnect_attempts = rtsp_config.get("max_reconnect_attempts", 10)
        if reconnect_delay is None:
            reconnect_delay = rtsp_config.get("reconnect_delay", 1)

        is_mp4_file = isinstance(video_path, str) and video_path.lower().endswith(
            ".mp4"
        )

        cap = None
        consecutive_failures = 0
        max_consecutive_failures = rtsp_config.get("max_consecutive_failures", 10)
        last_outside_hours_log = None
        was_in_operating_hours = (
            False  # Track operating hours state for end-of-day notification
        )
        frame_id = 0

        while True:

            # Check operating hours and get next end_time
            is_outside, end_time = is_outside_operating_hours()
            if is_outside:
                # Transition from operating hours to outside hours - send daily summary
                if was_in_operating_hours and self.plate_processor is not None:
                    now = datetime.now()
                    # Only send once per day at the end time
                    if (
                        self._last_daily_summary_time is None
                        or (now - self._last_daily_summary_time).total_seconds() > 86400
                    ):  # 24 hours
                        try:
                            self.log("[Thông báo] Gửi tổng hợp xe hàng ngày...")
                            self.plate_processor.save_daily_vehicle_summary()
                            self._last_daily_summary_time = now
                        except Exception as e:
                            self.log(f"Lỗi gửi thông báo hàng ngày: {e}")

                was_in_operating_hours = False
                time_info = get_time_info()
                current_time = time_info.get("current_time", "unknown")

                now = datetime.now()
                if (
                    last_outside_hours_log is None
                    or (now - last_outside_hours_log).total_seconds() > 3600
                ):
                    self.log(
                        f"⏱ Ngoài giờ vận hành ({current_time}). Chờ giờ vận hành..."
                    )
                    if cap is not None:
                        cap.release()
                        cap = None
                    last_outside_hours_log = now

                # Calculate time to next end_time for notification
                if end_time is not None:
                    wait_seconds = (end_time - now).total_seconds()
                    if wait_seconds > 0:
                        self.log(
                            f"[Thông báo] Sẽ kiểm tra lại vào {end_time.strftime('%H:%M:%S')}"
                        )
                        time.sleep(
                            min(wait_seconds, 1800)
                        )  # sleep up to 30 min or until end_time
                    else:
                        time.sleep(30)
                else:
                    time.sleep(30)
                continue

            was_in_operating_hours = True
            if last_outside_hours_log is not None:
                time_info = get_time_info()
                current_time = time_info.get("current_time", "unknown")
                self.log(
                    f"✓ Bắt đầu giờ vận hành ({current_time}). Khởi động xử lý stream..."
                )
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
                # Suppress FFmpeg warnings during frame reading
                sys.stderr = NullWriter()
                try:
                    success, frame = cap.read()
                    frame_id += 1
                    if frame_id % 4 != 0:
                        continue
                    if frame_id > 1000:
                        frame_id = 0
                        continue
                finally:
                    sys.stderr = _original_stderr

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
                    display_frame = frame_processor.process_frame_streaming(
                        frame, timestamp, plate_processor
                    )
                except Exception as e:
                    self.log(f"Lỗi xử lý frame: {e}")
                    continue

                if display_window:
                    display_size = (1280, 720)
                    if display_frame is not None:
                        try:
                            display_frame_resized = cv2.resize(
                                display_frame,
                                display_size,
                                interpolation=cv2.INTER_AREA,
                            )
                        except Exception:
                            display_frame_resized = display_frame
                    else:
                        display_frame_resized = display_frame

                    # Display vehicle count on the frame
                    if (
                        display_frame_resized is not None
                        and plate_processor is not None
                    ):
                        try:
                            today_str = datetime.now().strftime("%Y%m%d")
                            vehicles_today = [
                                tid
                                for tid, ts in plate_processor.vehicle_last_seen.items()
                                if ts == today_str
                                and "top"
                                in plate_processor.vehicle_directions.get(
                                    tid, ""
                                ).lower()
                            ]
                            vehicle_count = len(vehicles_today)

                            # Draw text on frame
                            text = f"Xe vao: {vehicle_count}"
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 1.2
                            font_color = (0, 255, 0)  # Green
                            font_thickness = 2
                            text_size = cv2.getTextSize(
                                text, font, font_scale, font_thickness
                            )[0]

                            # Position: top-right corner with padding
                            x = display_frame_resized.shape[1] - text_size[0] - 20
                            y = 40

                            # Draw background rectangle for better visibility
                            cv2.rectangle(
                                display_frame_resized,
                                (x - 10, y - text_size[1] - 10),
                                (x + text_size[0] + 10, y + 10),
                                (0, 0, 0),
                                -1,
                            )

                            # Draw text
                            cv2.putText(
                                display_frame_resized,
                                text,
                                (x, y),
                                font,
                                font_scale,
                                font_color,
                                font_thickness,
                            )
                        except Exception as e:
                            self.log(f"Lỗi hiển thị số lượng xe: {e}")

                    cv2.imshow(
                        "Vehicle Detection - Streaming Mode", display_frame_resized
                    )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            except Exception as e:
                error_msg = str(e).lower()
                if (
                    "hevc" in error_msg
                    or "codec" in error_msg
                    or "ref with poc" in error_msg
                ):
                    self.log(f"Lỗi codec (bỏ qua frame): {e}")
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        self.log(
                            "Quá nhiều lỗi codec liên tiếp. Đang thử kết nối lại..."
                        )
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
