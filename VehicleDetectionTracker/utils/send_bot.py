import os
from dotenv import load_dotenv
import requests
import time
import tempfile
import cv2
from datetime import datetime
from typing import Optional, Dict

load_dotenv()

# Try to load config, fallback to defaults if not available
try:
    from VehicleDetectionTracker.config_loader import get_telegram_config, get_advanced_config
    from VehicleDetectionTracker.logging_utils import log

    _use_config = True
except ImportError:
    _use_config = False

    # Fallback logging if import fails
    def log(message, category="telegram"):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")


def _prepare_telegram_photo(image_path: str, max_upload_kb: int = 900) -> tuple:
    """
    Prepare an image for Telegram upload.

    Returns:
        (prepared_path, mime_type, temp_file_to_cleanup)
    """
    try:
        original_size = os.path.getsize(image_path)
    except Exception:
        return image_path, "image/png", None

    # Keep original file if already small enough
    if original_size <= max_upload_kb * 1024:
        mime_type = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
        return image_path, mime_type, None

    img = cv2.imread(image_path)
    if img is None:
        return image_path, "image/png", None

    h, w = img.shape[:2]
    # Downscale very large images before JPEG encode for faster upload
    max_dim = 1600
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    temp_fd, temp_path = tempfile.mkstemp(prefix="tg_upload_", suffix=".jpg")
    os.close(temp_fd)

    # Try progressively lower quality until target size is met
    for quality in [85, 75, 65, 55, 45, 35]:
        ok = cv2.imwrite(temp_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            continue
        try:
            if os.path.getsize(temp_path) <= max_upload_kb * 1024:
                return temp_path, "image/jpeg", temp_path
        except Exception:
            continue

    # Return last successful JPEG even if still larger than target
    if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
        return temp_path, "image/jpeg", temp_path

    try:
        os.remove(temp_path)
    except OSError:
        pass
    return image_path, "image/png", None


def send_notify_to_telegram(license_plate, direction, timestamp=None, image_path=None):
    """
    Send a notification message to a configured Telegram chat/group with optional image attachment.

    This function is synchronous (uses requests) and is safe to call from
    synchronous code. If you need to call it from async code, run it in a
    thread using `asyncio.to_thread(send_notify_to_telegram, ...)`.

    Args:
        license_plate (str): Detected license plate text
        direction (str): Vehicle direction
        timestamp (datetime): Optional timestamp of detection
        image_path (str): Optional path to image file to attach

    Environment variables (optional):
      TELEGRAM_BOT_TOKEN - bot token (fallback to hardcoded token)
      TELEGRAM_CHAT_ID  - chat id (can be user ID or group ID, groups use negative numbers like -1001234567890)

    Note: To get your group chat ID:
      1. Add @userinfobot to your Telegram group
      2. It will show the group chat ID (usually starts with -100)
      3. Or use @RawDataBot to see the chat ID in the message object
      4. Make sure your bot is added to the group and has permission to send messages
    """
    # Check if notifications are enabled in config
    if _use_config:
        try:
            if not get_advanced_config().get("send_telegram_notification", True):
                log("[Telegram] Thông báo Telegram bị vô hiệu hóa trong cấu hình", "telegram")
                return {
                    "ok": False,
                    "error": "Notifications disabled in config (send_telegram_notification: false)",
                }
        except Exception as e:
            log(f"[Telegram] Lỗi khi kiểm tra cấu hình: {e}", "telegram")
            # Continue anyway if config loading fails
    
    TELEGRAM_BOT_TOKEN = os.getenv(
        "TELEGRAM_BOT_TOKEN",
        "",
    )
    # Get chat ID - can be positive (user) or negative (group/channel)
    # CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "6717527117") # for user
    CHAT_ID = os.getenv("TELEGRAM_GROUP_ID", "-4668166355")  # for group
    # Ensure chat_id is string (Telegram API accepts both string and int)
    CHAT_ID = str(CHAT_ID)

    # Map some direction labels to Vietnamese phrases
    if "top" in direction.lower():
        direction = "ra khỏi"
    elif "bottom" in direction.lower():
        direction = "vào"
    # Format timestamp for easy reading
    if isinstance(timestamp, str):
        dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S_%f")
    elif isinstance(timestamp, datetime):
        # If it's already a datetime object, use it directly
        dt = timestamp
    else:
        # If timestamp is None or invalid, use current time
        dt = datetime.now()
    formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")

    # Build message and strip leading/trailing whitespace
    message = f"Phát hiện biển số xe {license_plate} đi {direction} khu vực mỏ lúc {formatted_time} !".strip()
    log(f"[Telegram] Chuẩn bị gửi thông báo: {message}", "telegram")

    # Retry configuration - load from config if available
    if _use_config:
        telegram_config = get_telegram_config()
        max_retries = telegram_config.get("max_retries", 3)
        base_timeout = telegram_config.get("request_timeout", 15)
        connect_timeout = telegram_config.get("connect_timeout", 10)
        read_timeout = telegram_config.get("read_timeout", 60)
        retry_delays = telegram_config.get("retry_delays", [2, 4, 8])
        dns_retry_delay = telegram_config.get("dns_retry_delay", 2)
        max_upload_kb = telegram_config.get("max_upload_kb", 900)
    else:
        max_retries = 3
        base_timeout = 15
        connect_timeout = 10
        read_timeout = 60
        retry_delays = [2, 4, 8]
        dns_retry_delay = 2
        max_upload_kb = 900

    last_error = None

    prepared_image_path = image_path
    prepared_mime_type = "image/png"
    temp_upload_file = None
    if image_path and os.path.exists(image_path):
        prepared_image_path, prepared_mime_type, temp_upload_file = _prepare_telegram_photo(
            image_path,
            max_upload_kb=max_upload_kb,
        )

        if prepared_image_path != image_path and os.path.exists(prepared_image_path):
            original_size = os.path.getsize(image_path)
            prepared_size = os.path.getsize(prepared_image_path)
            reduction = (1 - (prepared_size / max(1, original_size))) * 100
            log(
                f"[Telegram] Compressed image for upload: {original_size / 1024:.1f}KB -> {prepared_size / 1024:.1f}KB ({reduction:.1f}% smaller)",
                "telegram",
            )

    try:
        for attempt in range(max_retries):
            try:
                file_timeout = base_timeout  # Default timeout
            
                # If image_path is provided, send photo with caption
                if prepared_image_path and os.path.exists(prepared_image_path):
                    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
                
                    # Validate image file before sending
                    try:
                        file_size = os.path.getsize(prepared_image_path)
                        if file_size == 0:
                            log(f"[Telegram] ❌ Image file is empty: {prepared_image_path}", "telegram")
                            return {"ok": False, "error": f"Image file is empty: {prepared_image_path}"}
                    
                        if file_size > 50 * 1024 * 1024:  # 50MB limit for Telegram
                            log(f"[Telegram] ❌ Image file too large ({file_size / 1024 / 1024:.1f}MB > 50MB): {prepared_image_path}", "telegram")
                            return {"ok": False, "error": f"Image file too large: {file_size / 1024 / 1024:.1f}MB"}
                    
                        log(f"[Telegram] Image file: {os.path.basename(prepared_image_path)} ({file_size} bytes)", "telegram")
                    except Exception as e:
                        log(f"[Telegram] ❌ Error validating image file: {e}", "telegram")
                        return {"ok": False, "error": f"Image validation error: {str(e)}"}

                    with open(prepared_image_path, "rb") as photo:
                        # File must be in tuple format with (filename, file_obj, content_type)
                        upload_name = "plate_screenshot.jpg" if prepared_mime_type == "image/jpeg" else "plate_screenshot.png"
                        files = {"photo": (upload_name, photo, prepared_mime_type)}
                        payload = {
                            "chat_id": CHAT_ID,
                            "caption": message,
                            "parse_mode": "HTML",  # Changed to HTML as Markdown can cause issues with some characters
                        }

                        # Size-aware read timeout: đợi Telegram xử lý upload đủ lâu
                        # để tránh timeout giả (nguyên nhân gửi trùng khi retry).
                        # ~1s cho mỗi 100KB, và tối thiểu là read_timeout đã cấu hình.
                        file_timeout = max(read_timeout, base_timeout + (file_size / (100 * 1024)))
                        log(f"[Telegram] Upload timeout: connect={connect_timeout:.0f}s, read={file_timeout:.1f}s (file size: {file_size / 1024:.1f}KB)", "telegram")

                        # Use (connect, read) tuple timeout for better reliability
                        response = requests.post(
                            url, files=files, data=payload, timeout=(connect_timeout, file_timeout)
                        )
                        response.raise_for_status()
                        try:
                            result = response.json()
                            if attempt > 0:
                                log(
                                    f"[Telegram] Gửi thành công sau {attempt + 1} lần thử",
                                    "telegram",
                                )
                            return result
                        except ValueError:
                            return {
                                "ok": False,
                                "error": "invalid_json_response",
                                "status_code": response.status_code, 
                            }
                else:
                    # Send text message only
                    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
                    payload = {
                        "chat_id": CHAT_ID,
                        "text": message,
                        "parse_mode": "HTML",
                    }

                    response = requests.post(url, json=payload, timeout=(connect_timeout, read_timeout))
                    response.raise_for_status()
                    try:
                        result = response.json()
                        if attempt > 0:
                            log(
                                f"[Telegram] Gửi thành công sau {attempt + 1} lần thử",
                                "telegram",
                            )
                        return result
                    except ValueError:
                        return {
                            "ok": False,
                            "error": "invalid_json_response",
                            "status_code": response.status_code,
                        }

            except requests.exceptions.Timeout as e:
                last_error = e
                if attempt < max_retries - 1:
                    delay = retry_delays[min(attempt, len(retry_delays) - 1)]
                    timeout_used = file_timeout if prepared_image_path and os.path.exists(prepared_image_path) else base_timeout
                    log(
                        f"[Telegram] Timeout kết nối (lần thử {attempt + 1}/{max_retries}, timeout={timeout_used:.1f}s), thử lại sau {delay} giây...",
                        "telegram",
                    )
                    time.sleep(delay)
                else:
                    timeout_used = file_timeout if prepared_image_path and os.path.exists(prepared_image_path) else base_timeout
                    log(f"[Telegram] ❌ Timeout sau {max_retries} lần thử (timeout={timeout_used:.1f}s): {e}", "telegram")
                    log(f"[Telegram] 💡 Kiểm tra kết nối internet hoặc tăng timeout trong cấu hình", "telegram")

            except requests.exceptions.ConnectionError as e:
                last_error = e
                error_str = str(e).lower()

                # Kiểm tra nếu là lỗi DNS (không thể resolve hostname)
                # Các lỗi DNS thường chứa: "failed to resolve", "getaddrinfo failed", "name resolution"
                is_dns_error = any(
                    keyword in error_str
                    for keyword in [
                        "failed to resolve",
                        "getaddrinfo failed",
                        "name resolution",
                        "nodename nor servname provided",
                        "name or service not known",
                    ]
                )

                if is_dns_error:
                    # Lỗi DNS không nên retry nhiều vì sẽ không giải quyết được
                    # Chỉ thử lại 1 lần nữa (tổng 2 lần) để đảm bảo không phải tạm thời
                    if attempt < 1:  # Chỉ retry 1 lần cho DNS error
                        delay = dns_retry_delay
                        log(
                            f"[Telegram] Lỗi DNS (không thể phân giải tên miền) (lần thử {attempt + 1}/2), thử lại sau {delay} giây...",
                            "telegram",
                        )
                        time.sleep(delay)
                    else:
                        log(
                            f"[Telegram] ❌ Lỗi DNS: Không thể kết nối đến Telegram API (kiểm tra kết nối internet/DNS)",
                            "telegram",
                        )
                        log(f"[Telegram] Chi tiết: {e}", "telegram")
                        # Không retry nữa, return ngay
                        return {"ok": False, "error": f"DNS resolution failed: {str(e)}"}
                else:
                    # Các lỗi kết nối khác (có thể retry)
                    if attempt < max_retries - 1:
                        delay = retry_delays[min(attempt, len(retry_delays) - 1)]
                        log(
                            f"[Telegram] Lỗi kết nối (lần thử {attempt + 1}/{max_retries}), thử lại sau {delay} giây...",
                            "telegram",
                        )
                        time.sleep(delay)
                    else:
                        log(
                            f"[Telegram] ❌ Không thể kết nối sau {max_retries} lần thử: {e}",
                            "telegram",
                        )

            except requests.exceptions.RequestException as e:
                # For other HTTP errors (4xx, 5xx), don't retry as they're likely permanent
                log(f"[Telegram] ❌ Lỗi HTTP: {e}", "telegram")
                return {
                    "ok": False,
                    "error": str(e),
                    "status_code": getattr(e.response, "status_code", None),
                }

            except Exception as e:
                # For unexpected errors, don't retry
                log(f"[Telegram] ❌ Lỗi không mong đợi: {e}", "telegram")
                return {"ok": False, "error": str(e)}

        # If we've exhausted all retries
        return {
            "ok": False,
            "error": f"Failed after {max_retries} attempts: {str(last_error)}",
        }
    finally:
        if temp_upload_file and os.path.exists(temp_upload_file):
            try:
                os.remove(temp_upload_file)
            except OSError:
                pass


def send_warning_to_telegram(warning_message: str):
    """
    Send a warning message to a configured Telegram chat/group.

    Environment variables (optional):
      TELEGRAM_BOT_TOKEN - bot token (fallback to hardcoded token)
      TELEGRAM_CHAT_ID  - chat id (can be user ID or group ID, groups use negative numbers like -1001234567890)

    Note: To get your group chat ID:
      1. Add @userinfobot to your Telegram group
      2. It will show the group chat ID (usually starts with -100)
      3. Or use @RawDataBot to see the chat ID in the message object
      4. Make sure your bot is added to the group and has permission to send messages
    """
    # Check if notifications are enabled in config
    if _use_config:
        try:
            if not get_advanced_config().get("send_telegram_notification", True):
                log("[Telegram] Thông báo Telegram bị vô hiệu hóa trong cấu hình", "telegram")
                return {
                    "ok": False,
                    "error": "Notifications disabled in config (send_telegram_notification: false)",
                }
        except Exception as e:
            log(f"[Telegram] Lỗi khi kiểm tra cấu hình: {e}", "telegram")
            # Continue anyway if config loading fails

    TELEGRAM_BOT_TOKEN = os.getenv(
        "TELEGRAM_BOT_TOKEN",
        "",
    )
    # Get chat ID - can be positive (user) or negative (group/channel)
    # CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "6717527117") # for user
    CHAT_ID = os.getenv("TELEGRAM_GROUP_ID", "-4668166355")  # for group
    # Ensure chat_id is string (Telegram API accepts both string and int)
    CHAT_ID = str(CHAT_ID)

    # Build message and strip leading/trailing whitespace
    message = f"{warning_message} !".strip()

    # Retry configuration - load from config if available
    if _use_config:
        telegram_config = get_telegram_config()
        max_retries = telegram_config.get("max_retries", 3)
        base_timeout = telegram_config.get("request_timeout", 15)
        retry_delays = telegram_config.get("retry_delays", [2, 4, 8])
        dns_retry_delay = telegram_config.get("dns_retry_delay", 2)
    else:
        max_retries = 3
        base_timeout = 15
        retry_delays = [2, 4, 8]
        dns_retry_delay = 2

    last_error = None

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}

        response = requests.post(url, json=payload, timeout=base_timeout)
    except requests.exceptions.Timeout as e:
        last_error = e

    except requests.exceptions.ConnectionError as e:
        last_error = e
        error_str = str(e).lower()

        # Kiểm tra nếu là lỗi DNS (không thể resolve hostname)
        is_dns_error = any(
            keyword in error_str
            for keyword in [
                "failed to resolve",
                "getaddrinfo failed",
                "name resolution",
                "nodename nor servname provided",
                "name or service not known",
            ]
        )
    except requests.exceptions.RequestException as e:
        # For other HTTP errors (4xx, 5xx), don't retry as they're likely permanent
        log(f"[Telegram] ❌ Lỗi HTTP: {e}", "telegram")
        return {
            "ok": False,
            "error": str(e),
            "status_code": getattr(e.response, "status_code", None),
        }

    except Exception as e:
        # For unexpected errors, don't retry
        log(f"[Telegram] ❌ Lỗi không mong đợi: {e}", "telegram")
        return {"ok": False, "error": str(e)}

    # If we've exhausted all retries
    return {
        "ok": False,
        "error": f"Failed after {max_retries} attempts: {str(last_error)}",
    }


# send_warning_to_telegram(warning_message="Test cảnh báo từ hệ thống Vehicle Detection Tracker")
