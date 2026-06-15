"""
Resend daily vehicle summary Telegram notification.

Reads today's state file and sends the daily summary notification.
Usage: python resend_daily_summary.py [YYYYMMDD]
"""

import json
import os
import sys
from datetime import datetime

# Add workspace root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from VehicleDetectionTracker.vehicle_summary import save_daily_vehicle_summary
from VehicleDetectionTracker.utils.send_bot import send_warning_to_telegram


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def main():
    date_str = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y%m%d")
    state_file = os.path.join("vehicle_state", f"vehicle_state_{date_str}.json")

    if not os.path.exists(state_file):
        log(f"[ERROR] State file not found: {state_file}")
        sys.exit(1)

    with open(state_file, "r", encoding="utf-8") as f:
        state = json.load(f)

    log(f"[INFO] Loaded state from: {state_file}")

    vehicle_directions: dict = {
        int(k): v for k, v in state.get("vehicle_directions", {}).items()
    }
    vehicle_plates: dict = {
        int(k): v for k, v in state.get("vehicle_plates", {}).items()
    }
    vehicle_plate_counts: dict = {
        int(k): v for k, v in state.get("vehicle_plate_counts", {}).items()
    }

    # When loading from saved state, all vehicles in the state file are assumed valid.
    # No need to reconstruct vehicle_last_seen per track.
    vehicle_last_seen: dict = {}

    all_track_ids = set(vehicle_directions.keys()) | set(vehicle_plates.keys())
    log(f"[INFO] Vehicles in state: {sorted(all_track_ids)}")
    log(f"[INFO] vehicle_directions: {vehicle_directions}")
    log(f"[INFO] vehicle_plates: {vehicle_plates}")

    save_daily_vehicle_summary(
        vehicle_last_seen,
        vehicle_directions,
        vehicle_plates,
        log,
        send_warning_to_telegram,
        date_str,
        vehicle_plate_counts,
    )


if __name__ == "__main__":
    main()
