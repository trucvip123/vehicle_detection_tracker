import os
import requests


def send_notify_to_telegram(license_plate, direction, timestamp=None):
    """
    Send a notification message to a configured Telegram chat.

    This function is synchronous (uses requests) and is safe to call from
    synchronous code. If you need to call it from async code, run it in a
    thread using `asyncio.to_thread(send_notify_to_telegram, ...)`.

    Environment variables (optional):
      TELEGRAM_BOT_TOKEN - bot token (fallback to hardcoded token)
      TELEGRAM_CHAT_ID  - chat id (fallback to hardcoded id)
    """
    TELEGRAM_BOT_TOKEN = os.getenv(
        "TELEGRAM_BOT_TOKEN",
        "7606643120:AAG9CFp6WQhFewyePoykExUTm8eNHJDp_yM",
    )
    CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "6717527117")

    # Map some direction labels to Vietnamese phrases
    if direction == "Top":
        direction = "vào"
    elif direction == "Bottom":
        direction = "ra khỏi"

    # Build message and strip leading/trailing whitespace
    message = f"Phát hiện biển số xe {license_plate} đi {direction} khu vực mỏ vào lúc {timestamp} !".strip()

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}

    try:
        # Use a short timeout so background threads don't hang indefinitely
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()
        try:
            return response.json()
        except ValueError:
            return {"ok": False, "error": "invalid_json_response", "status_code": response.status_code}
    except Exception as e:
        # Don't raise in background tasks; log and return an error dict
        print(f"send_notify_to_telegram error: {e}")
        return {"ok": False, "error": str(e)}