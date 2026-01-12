"""
Scheduler script to send daily vehicle count summary at 18:00.
"""

from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime
from VehicleDetectionTracker.plate_processor import PlateProcessor
from VehicleDetectionTracker.config_loader import get_plate_detection_config


# Dummy log function (replace with your actual logger if needed)
def log(msg):
    print(f"[{datetime.now()}] {msg}")


def run_daily_summary():
    # Load config and models as needed
    config = get_plate_detection_config()
    plate_model = None  # Load your plate model here
    ocr_reader = None  # Load your OCR reader here
    executor = None  # If you use ThreadPoolExecutor, set it up here
    processor = PlateProcessor(plate_model, ocr_reader, executor, log)
    processor.save_daily_vehicle_summary()
    log("Sent daily vehicle summary notification.")


if __name__ == "__main__":
    scheduler = BlockingScheduler()
    scheduler.add_job(run_daily_summary, "cron", hour=18, minute=0)
    log("Scheduler started. Will send daily vehicle summary at 18:00.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log("Scheduler stopped.")
