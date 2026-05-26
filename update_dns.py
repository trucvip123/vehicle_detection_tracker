import requests

ZONE_ID = ""
RECORD_ID = ""
TOKEN = ""

DOMAIN = "aicamera.dienthanhliem.com"

ip = requests.get("https://api.ipify.org").text

url = (
    f"https://api.cloudflare.com/client/v4/"
    f"zones/{ZONE_ID}/dns_records/{RECORD_ID}"
)

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

payload = {
    "type": "A",
    "name": DOMAIN,
    "content": ip,
    "ttl": 60,
    "proxied": False
}

response = requests.put(
    url,
    headers=headers,
    json=payload
)

print(response.json())