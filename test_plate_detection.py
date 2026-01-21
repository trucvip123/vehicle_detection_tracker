"""
Test script for license plate detection and OCR from vehicle_frame images.

Usage:
    python test_plate_detection.py [image_path]

Example:
    python test_plate_detection.py screenshots/vehicle_frame_20260104_132626_159.png
"""

import cv2
import sys
import threading
from datetime import datetime
from pathlib import Path

from numpy import True_

from VehicleDetectionTracker.plate_utils import (
    initialize_plate_detector,
    detect_license_plate_sync,
)
from VehicleDetectionTracker.function.paddleocr_wrapper import create_paddleocr_reader
from VehicleDetectionTracker.logging_utils import log


def test_plate_detection(image_path, use_gpu=None):
    """
    Test license plate detection and OCR on a single image.

    Args:
        image_path (str): Path to the vehicle_frame image
        use_gpu (bool, optional): Force GPU usage. If None, auto-detect.
    """
    print("=" * 60)
    print("License Plate Detection Test")
    print("=" * 60)

    # Check if image exists
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"❌ Error: Image not found at {image_path}")
        return

    # Load image
    print(f"\n📷 Loading image: {image_path}")
    vehicle_frame = cv2.imread(str(image_path))

    if vehicle_frame is None:
        print(f"❌ Error: Could not load image from {image_path}")
        return

    print(f"✓ Image loaded successfully")
    print(f"  - Shape: {vehicle_frame.shape}")
    print(f"  - Size: {vehicle_frame.shape[1]}x{vehicle_frame.shape[0]}")

    # Initialize plate detector
    print(f"\n🔍 Initializing license plate detector...")
    plate_model = initialize_plate_detector(
        model_path="model/LP_detector.pt", device=use_gpu
    )

    if plate_model is None:
        print("❌ Error: Failed to initialize plate detector")
        return

    print("✓ Plate detector initialized")

    # Initialize OCR reader
    print(f"\n📝 Initializing OCR reader...")
    try:
        ocr_reader = create_paddleocr_reader(
            lang="en", use_angle_cls=True, use_gpu=use_gpu
        )
        print("✓ OCR reader initialized")
    except Exception as e:
        print(f"⚠ Warning: Failed to initialize OCR reader: {e}")
        print("  Continuing without OCR (will only detect plate bounding box)")
        ocr_reader = None

    # Create threading lock (required by detect_license_plate_sync)
    model_lock = threading.Lock()

    # Generate timestamp for output files
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

    # Run detection
    print(f"\n🚀 Running license plate detection and OCR...")
    print("-" * 60)

    result = detect_license_plate_sync(
        plate_model=plate_model,
        vehicle_frame=vehicle_frame,
        ocr_reader=ocr_reader,
        model_lock=model_lock,
        timestamp_str=timestamp_str,
    )

    print("-" * 60)

    # Display results
    print(f"\n📊 Results:")
    print("=" * 60)

    if result["bbox"] is not None:
        x1, y1, x2, y2 = result["bbox"]
        print(f"✓ License plate detected!")
        print(f"  - Bounding box: ({x1}, {y1}) to ({x2}, {y2})")
        print(f"  - Width: {x2 - x1}px")
        print(f"  - Height: {y2 - y1}px")

        if result["text"] is not None and result["text"] != "unknown":
            print(f"  - License plate text: {result['text']}")
        else:
            print(
                f"  - License plate text: Not recognized (OCR failed or not available)"
            )
    else:
        print(f"❌ No license plate detected")

    # Draw results on image and save
    if result["bbox"] is not None:
        output_image = vehicle_frame.copy()
        x1, y1, x2, y2 = result["bbox"]

        # Draw bounding box
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw text
        if result["text"] and result["text"] != "unknown":
            text = result["text"]
            # Put text above the box
            cv2.putText(
                output_image,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                output_image,
                "Plate detected",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        # Save output image
        # output_path = f"test_output_{timestamp_str}.png"
        # cv2.imwrite(output_path, output_image)
        # print(f"\n💾 Output image saved: {output_path}")

    print("=" * 60)
    print("Test completed!")
    print("=" * 60)


def main():
    """Main function."""
    # Get image path from command line or use default
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Try to find a vehicle_frame image in screenshots
        screenshots_dir = Path("screenshots")
        if screenshots_dir.exists():
            vehicle_frames = list(screenshots_dir.glob("vehicle_frame_*.png"))
            if vehicle_frames:
                image_path = vehicle_frames[0]
                print(f"ℹ️  No image specified, using: {image_path}")
            else:
                print(
                    "❌ Error: No image specified and no vehicle_frame images found in screenshots/"
                )
                print("\nUsage: python test_plate_detection.py [image_path]")
                return
        else:
            print("❌ Error: No image specified and screenshots/ directory not found")
            print("\nUsage: python test_plate_detection.py [image_path]")
            return

    # Check for GPU flag
    use_gpu = None
    if len(sys.argv) > 2:
        if sys.argv[2].lower() in ["--gpu", "-g", "gpu"]:
            use_gpu = True
        elif sys.argv[2].lower() in ["--cpu", "-c", "cpu"]:
            use_gpu = False

    # Run test
    test_plate_detection(image_path, use_gpu=use_gpu)


if __name__ == "__main__":
    main()
