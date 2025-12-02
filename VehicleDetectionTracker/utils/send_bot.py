import os
import requests


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
        "7606643120:AAG9CFp6WQhFewyePoykExUTm8eNHJDp_yM",
    )
    # Get chat ID - can be positive (user) or negative (group/channel)
    # CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "6717527117") # for user
    CHAT_ID = os.getenv("TELEGRAM_GROUP_ID", "-4668166355")  # for group
    # Ensure chat_id is string (Telegram API accepts both string and int)
    CHAT_ID = str(CHAT_ID)

    # Map some direction labels to Vietnamese phrases
    if direction == "Top":
        direction = "ra khỏi"
    elif direction == "Bottom":
        direction = "vào"

    # Build message and strip leading/trailing whitespace
    message = f"Phát hiện biển số xe {license_plate} đi {direction} khu vực mỏ lúc {timestamp} !".strip()

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

                # Use a short timeout so background threads don't hang indefinitely
                response = requests.post(url, files=files, data=payload, timeout=10)
                response.raise_for_status()
                try:
                    return response.json()
                except ValueError:
                    return {
                        "ok": False,
                        "error": "invalid_json_response",
                        "status_code": response.status_code,
                    }
        else:
            # Send text message only
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}

            response = requests.post(url, json=payload, timeout=5)
            response.raise_for_status()
            try:
                return response.json()
            except ValueError:
                return {
                    "ok": False,
                    "error": "invalid_json_response",
                    "status_code": response.status_code,
                }

    except Exception as e:
        # Don't raise in background tasks; log and return an error dict
        print(f"send_notify_to_telegram error: {e}")
        return {"ok": False, "error": str(e)}
