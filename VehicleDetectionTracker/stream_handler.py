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
        self._last_daily_summary_date = None  # Track last date daily summary was sent (YYYYMMDD format)

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
        was_in_operating_hours = False  # Track operating hours state for end-of-day notification
        frame_id = 0

        while True:
            # Check operating hours and get next end_time
            is_outside, end_time = is_outside_operating_hours()
            # self.log(f"[DEBUG] is_outside_operating_hours: {is_outside}, end_time: {end_time}")
            if is_outside:
                # Transition from operating hours to outside hours - send daily summary
                if was_in_operating_hours and self.plate_processor is not None:
                    today_date = datetime.now().strftime("%Y%m%d")
                    # Only send once per day - check if today's summary has been sent
                    if self._last_daily_summary_date != today_date:
                        try:
                            self.log(f"[Thông báo] Gửi tổng hợp xe hàng ngày ({today_date})...")
                            self.plate_processor.save_daily_vehicle_summary()
                            self._last_daily_summary_date = today_date
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
                    if frame_id > 100:
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

                    # Display vehicle count and license plates on the frame
                    if (
                        display_frame_resized is not None
                        and plate_processor is not None
                    ):
                        try:
                            today_str = datetime.now().strftime("%Y%m%d")
                            vehicles_today = [
                                tid
                                for tid, ts in plate_processor.vehicle_last_seen.items()
                                if hasattr(ts, "strftime")
                                and ts.strftime("%Y%m%d") == today_str
                                and "bottom"
                                in plate_processor.vehicle_directions.get(
                                    tid, ""
                                ).lower()
                            ]
                            vehicle_count = len(vehicles_today)
                            
                            # Get today's vehicles summary from plate_processor
                            vehicle_list = plate_processor.get_today_vehicles_summary()

                            # Draw total count on frame (top-right corner)
                            text_total = f"Total Vehicles: {vehicle_count}"
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 1.0
                            font_color = (0, 255, 0)  # Green
                            font_thickness = 2
                            text_size = cv2.getTextSize(
                                text_total, font, font_scale, font_thickness
                            )[0]

                            # Position: top-right corner with padding
                            x_total = display_frame_resized.shape[1] - text_size[0] - 20
                            y_total = 40

                            # Draw background rectangle for better visibility
                            cv2.rectangle(
                                display_frame_resized,
                                (x_total - 10, y_total - text_size[1] - 10),
                                (x_total + text_size[0] + 10, y_total + 10),
                                (0, 0, 0),
                                -1,
                            )

                            # Draw total count text
                            cv2.putText(
                                display_frame_resized,
                                text_total,
                                (x_total, y_total),
                                font,
                                font_scale,
                                font_color,
                                font_thickness,
                            )

                            # Draw vehicle list with plates and counts (left side) - detailed view
                            if vehicle_list:
                                y_list = 80
                                line_height = 38
                                max_vehicles_display = 8  # Show max 8 vehicles (more space for detail)
                                
                                # Title with background
                                title_text = "Detail Vehicle List"
                                title_size = cv2.getTextSize(title_text, font, 0.9, 2)[0]
                                cv2.rectangle(
                                    display_frame_resized,
                                    (10, y_list - title_size[1]),
                                    (340, y_list + 18),
                                    (0, 100, 255),  # Orange background
                                    -1,
                                )
                                cv2.putText(
                                    display_frame_resized,
                                    title_text,
                                    (20, y_list + 10),
                                    font,
                                    0.9,
                                    (255, 255, 255),  # White text
                                    2,
                                )
                                y_list += line_height + 5
                                
                                # Draw separator line
                                cv2.line(
                                    display_frame_resized,
                                    (10, y_list - 10),
                                    (340, y_list - 10),
                                    (0, 255, 255),
                                    2,
                                )
                                y_list += 5
                                
                                # Calculate total for percentage
                                total_count = sum(count for _, count in vehicle_list)
                                
                                # Draw each vehicle info with details
                                for idx, (plate, count) in enumerate(vehicle_list[:max_vehicles_display]):
                                    # Format: "1. 29A12345        3 xe"
                                    vehicle_text = f"{idx+1}. {plate}"
                                    detail_text = f"{count} xe"
                                    
                                    # Background for vehicle entry
                                    bg_color = (20, 80, 150)  # Dark blue background
                                    cv2.rectangle(
                                        display_frame_resized,
                                        (10, y_list - 5),
                                        (310, y_list + line_height - 10),
                                        bg_color,
                                        -1,
                                    )
                                    
                                    # Draw border
                                    cv2.rectangle(
                                        display_frame_resized,
                                        (10, y_list - 5),
                                        (310, y_list + line_height - 10),
                                        (0, 255, 255),  # Cyan border
                                        2,
                                    )
                                    
                                    # Draw plate number (large)
                                    cv2.putText(
                                        display_frame_resized,
                                        vehicle_text,
                                        (20, y_list + 20),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.85,
                                        (0, 255, 0),  # Green for plate
                                        3,  # Thicker for bold effect
                                    )
                                    
                                    # Draw count and percentage (smaller, right aligned)
                                    detail_size = cv2.getTextSize(detail_text, font, 0.7, 1)[0]
                                    cv2.putText(
                                        display_frame_resized,
                                        detail_text,
                                        (290 - detail_size[0], y_list + 15),
                                        font,
                                        0.7,
                                        (255, 255, 0),  # Yellow for count
                                        1,
                                    )
                                    
                                    y_list += line_height
                        except Exception as e:
                            self.log(f"Loi hien thi bien so xe: {e}")

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
