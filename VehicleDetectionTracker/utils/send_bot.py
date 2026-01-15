import os
from dotenv import load_dotenv
import requests
import time
from datetime import datetime
from typing import Optional, Dict

load_dotenv()

# Try to load config, fallback to defaults if not available
try:
    from VehicleDetectionTracker.config_loader import get_telegram_config
    from VehicleDetectionTracker.logging_utils import log

    _use_config = True
except ImportError:
    _use_config = False

    # Fallback logging if import fails
    def log(message, category="telegram"):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")


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
    dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S_%f")
    formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")

    # Build message and strip leading/trailing whitespace
    message = f"Phát hiện biển số xe {license_plate} đi {direction} khu vực mỏ lúc {formatted_time} !".strip()
    log(f"[Telegram] Chuẩn bị gửi thông báo: {message}", "telegram")

    # Retry configuration
    max_retries = 3
    base_timeout = 15  # Increased timeout to 15 seconds
    retry_delays = [2, 4, 8]  # Exponential backoff delays in seconds

    last_error = None

    for attempt in range(max_retries):
        try:
            # If image_path is provided, send photo with caption
            if image_path and os.path.exists(image_path):
                url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"

                with open(image_path, "rb") as photo:
                    files = {"photo": photo}
                    payload = {
                        "chat_id": CHAT_ID,
                        "caption": message,
                        "parse_mode": "Markdown",
                    }

                    # Use increased timeout for better reliability
                    response = requests.post(
                        url, files=files, data=payload, timeout=base_timeout
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
                    "parse_mode": "Markdown",
                }

                response = requests.post(url, json=payload, timeout=base_timeout)
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
                delay = retry_delays[attempt]
                log(
                    f"[Telegram] Timeout kết nối (lần thử {attempt + 1}/{max_retries}), thử lại sau {delay} giây...",
                    "telegram",
                )
                time.sleep(delay)
            else:
                log(f"[Telegram] ❌ Timeout sau {max_retries} lần thử: {e}", "telegram")

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
                dns_retry_delay = 2
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
                    delay = retry_delays[attempt]
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
    TELEGRAM_BOT_TOKEN = os.getenv(
        "TELEGRAM_BOT_TOKEN",
        "",
    )
    print("TELEGRAM_BOT_TOKEN:", TELEGRAM_BOT_TOKEN)

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
        payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}

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
