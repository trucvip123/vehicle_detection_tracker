"""
Config Loader cho Vehicle Detection Tracker
Đọc và quản lý các cấu hình từ file config.yaml
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
from VehicleDetectionTracker.logging_utils import log


class ConfigLoader:
    """Class để load và quản lý config"""

    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._config is None:
            self._load_config()

    def _load_config(self):
        """Load config từ file YAML"""
        # Tìm file config.yaml trong thư mục VehicleDetectionTracker
        config_path = Path(__file__).parent / "config.yaml"

        if not config_path.exists():
            # Fallback: tìm trong thư mục gốc
            config_path = Path(__file__).parent.parent / "config.yaml"

        if not config_path.exists():
            log(f"⚠ Warning: Config file not found at {config_path}", "config")
            log("⚠ Using default configuration", "config")
            return

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f) or {}

        except Exception as e:
            log(f"⚠ Error loading config: {e}", "config")
            log("⚠ Using default configuration", "config")

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Lấy giá trị từ config bằng đường dẫn
        Ví dụ: get('detection.confidence') -> 0.3
        """
        keys = key_path.split(".")
        value = self._config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def get_detection_config(self) -> Dict:
        """Lấy config cho detection"""
        return self._config.get("detection", {})

    def get_tracking_config(self) -> Dict:
        """Lấy config cho tracking"""
        return self._config.get("tracking", {})

    def get_plate_detection_config(self) -> Dict:
        """Lấy config cho plate detection"""
        return self._config.get("plate_detection", {})

    def get_ocr_config(self) -> Dict:
        """Lấy config cho OCR"""
        return self._config.get("ocr", {})

    def get_rtsp_config(self) -> Dict:
        """Lấy config cho RTSP"""
        return self._config.get("rtsp", {})

    def get_telegram_config(self) -> Dict:
        """Lấy config cho Telegram"""
        return self._config.get("telegram", {})

    def get_threading_config(self) -> Dict:
        """Lấy config cho threading"""
        return self._config.get("threading", {})

    def get_paths_config(self) -> Dict:
        """Lấy config cho paths"""
        return self._config.get("paths", {})

    def get_display_config(self) -> Dict:
        """Lấy config cho display"""
        return self._config.get("display", {})

    def get_advanced_config(self) -> Dict:
        """Lấy config cho advanced settings"""
        return self._config.get("advanced", {})

    def get_operating_hours_config(self) -> Dict:
        """Lấy config cho operating_hours"""
        return self._config.get("operating_hours", {})

    # Convenience function
    def get_operating_hours_config() -> Dict:
        """Lấy operating_hours config"""
        return get_config().get_operating_hours_config()


# Singleton instance
def get_config() -> ConfigLoader:
    """Lấy instance của ConfigLoader (singleton)"""
    return ConfigLoader()


# Convenience functions
def get_detection_config() -> Dict:
    """Lấy detection config"""
    return get_config().get_detection_config()


def get_tracking_config() -> Dict:
    """Lấy tracking config"""
    return get_config().get_tracking_config()


def get_plate_detection_config() -> Dict:
    """Lấy plate detection config"""
    return get_config().get_plate_detection_config()


def get_ocr_config() -> Dict:
    """Lấy OCR config"""
    return get_config().get_ocr_config()


def get_rtsp_config() -> Dict:
    """Lấy RTSP config"""
    return get_config().get_rtsp_config()


def get_telegram_config() -> Dict:
    """Lấy Telegram config"""
    return get_config().get_telegram_config()


def get_threading_config() -> Dict:
    """Lấy threading config"""
    return get_config().get_threading_config()


def get_paths_config() -> Dict:
    """Lấy paths config"""
    return get_config().get_paths_config()


def get_display_config() -> Dict:
    """Lấy display config"""
    return get_config().get_display_config()


def get_advanced_config() -> Dict:
    """Lấy advanced config"""
    return get_config().get_advanced_config()
