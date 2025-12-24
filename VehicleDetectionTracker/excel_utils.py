"""Excel file handling utilities."""

import os
import pandas as pd
import threading


class ExcelManager:
    """Thread-safe Excel file manager."""

    def __init__(self, excel_output_path, log_func):
        self.excel_output_path = excel_output_path
        self._excel_lock = threading.Lock()
        self.log = log_func
        self._initialize_excel_file()

    def _initialize_excel_file(self):
        """Initialize Excel file with headers if it doesn't exist."""
        if not os.path.exists(self.excel_output_path):
            df = pd.DataFrame(
                columns=["Vehicle_ID", "License_Plate", "Direction_Label", "Timestamp"]
            )
            df.to_excel(self.excel_output_path, index=False, engine="openpyxl")
            self.log(f"Created Excel file: {self.excel_output_path}")

    def save_to_excel(self, vehicle_id, license_plate, direction_label, timestamp):
        """
        Save vehicle data to Excel file (thread-safe).
        Only saves once per vehicle_id.

        Args:
            vehicle_id: Vehicle track ID
            license_plate: Detected license plate text
            direction_label: Vehicle direction label
            timestamp: Detection timestamp
        """
        try:
            with self._excel_lock:
                # Read existing data
                if os.path.exists(self.excel_output_path):
                    df = pd.read_excel(self.excel_output_path, engine="openpyxl")
                else:
                    df = pd.DataFrame(
                        columns=[
                            "Vehicle_ID",
                            "License_Plate",
                            "Direction_Label",
                            "Timestamp",
                        ]
                    )

                # Append new row
                new_row = {
                    "Vehicle_ID": vehicle_id,
                    "License_Plate": license_plate,
                    "Direction_Label": direction_label,
                    "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

                # Save to Excel
                df.to_excel(self.excel_output_path, index=False, engine="openpyxl")
        except Exception as e:
            self.log(f"Error saving to Excel: {e}")
