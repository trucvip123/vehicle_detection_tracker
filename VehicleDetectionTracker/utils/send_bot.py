import requests


def send_order_to_telegram(license_plate, direction, timestamp=None):
    TELEGRAM_BOT_TOKEN = "7606643120:AAG9CFp6WQhFewyePoykExUTm8eNHJDp_yM"
    CHAT_ID = "6717527117"
    
    if direction == "Top":
        direction = "vào"
    elif direction == "Bottom":
        direction = "ra khỏi"
        
    message = f"""
    Phát hiện biển số xe {license_plate} đi {direction} khu vực mỏ vào lúc {timestamp} !
    """
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    
    response = requests.post(url, json=payload)
    return response.json()