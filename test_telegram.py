"""
Test Telegram notifications directly
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from VehicleDetectionTracker.utils.send_bot import send_notify_to_telegram

print("\n" + "="*80)
print("SENDING TEST TELEGRAM NOTIFICATION")
print("="*80 + "\n")

# Test parameters
test_plate = "ABC123XY"
test_direction = "entering"
test_image = None  # No image for this test

print(f"[TEST] Sending notification:")
print(f"  Plate: {test_plate}")
print(f"  Direction: {test_direction}")
print(f"  With Image: {test_image}\n")

try:
    result = send_notify_to_telegram(
        license_plate=test_plate,
        direction=test_direction,
        image_path=test_image
    )
    
    print("[RESULT]:")
    print(f"  Status: {result.get('ok', False)}")
    
    if result.get('ok'):
        print(f"  Message ID: {result.get('result', {}).get('message_id', 'N/A')}")
        print("\n✅ SUCCESS! Telegram notification sent!")
    else:
        print(f"  Error: {result.get('error', 'Unknown error')}")
        print(f"  Details: {result}")
        print("\n❌ Failed to send - check error above")

except Exception as e:
    print(f"❌ Exception occurred: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80 + "\n")
