"""Image processing and encoding utilities."""
import base64
import cv2
import numpy as np
import math


def encode_image_base64(image):
    """
    Encode an image as base64.

    Args:
        image (numpy.ndarray): The image to be encoded.

    Returns:
        str: Base64-encoded image.
    """
    _, buffer = cv2.imencode(".jpg", image)
    image_base64 = base64.b64encode(buffer).decode()
    return image_base64


def decode_image_base64(image_base64):
    """
    Decode a base64-encoded image.

    Args:
        image_base64 (str): Base64-encoded image data.

    Returns:
        numpy.ndarray or None: Decoded image as a numpy array or None if decoding fails.
    """
    try:
        image_data = base64.b64decode(image_base64)
        image_np = np.frombuffer(image_data, dtype=np.uint8)
        image = cv2.imdecode(image_np, flags=cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        return None


def increase_brightness(image, factor=1.5):
    """
    Increases the brightness of an image by multiplying its pixels by a factor.

    :param image: The input image in numpy array format.
    :param factor: The brightness increase factor. A value greater than 1 will increase brightness.
    :return: The image with increased brightness.
    """
    brightened_image = cv2.convertScaleAbs(image, alpha=factor, beta=0)
    return brightened_image


def convert_meters_per_second_to_kmph(meters_per_second):
    """Convert speed from m/s to km/h"""
    kmph = meters_per_second * 3.6
    return kmph


def draw_plate_text_corner(frame, plates_dict):
    """
    Draw detected license plates at the top-left corner of the frame.
    Each vehicle's plate is displayed on a separate line.

    Args:
        frame (numpy.ndarray): Frame to draw on
        plates_dict (dict): Dictionary of {track_id: plate_text}
    """
    if not plates_dict:
        return frame

    # Draw background rectangle for text
    h, w = frame.shape[:2]
    num_plates = len(plates_dict)
    text_height = 30
    padding = 10
    bg_height = num_plates * text_height + padding * 2

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (400, bg_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Draw text for each detected plate
    y_offset = padding + 25
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2

    for idx, (track_id, plate_text) in enumerate(plates_dict.items()):
        if plate_text and plate_text != "unknown":
            text = f"Vehicle {track_id}: {plate_text}"
            cv2.putText(
                frame,
                text,
                (15, y_offset + idx * text_height),
                font,
                font_scale,
                (0, 255, 0),  # Green color
                font_thickness,
            )

    return frame


def map_direction_to_label(direction):
    """Map direction angle to label."""
    direction_ranges = {
        (-math.pi / 8, math.pi / 8): "Right",
        (math.pi / 8, 3 * math.pi / 8): "Bottom Right",
        (3 * math.pi / 8, 5 * math.pi / 8): "Bottom",
        (5 * math.pi / 8, 7 * math.pi / 8): "Bottom Left",
        (7 * math.pi / 8, -7 * math.pi / 8): "Left",
        (-7 * math.pi / 8, -5 * math.pi / 8): "Top Left",
        (-5 * math.pi / 8, -3 * math.pi / 8): "Top",
        (-3 * math.pi / 8, -math.pi / 8): "Top Right",
    }
    for angle_range, label in direction_ranges.items():
        if angle_range[0] <= direction <= angle_range[1]:
            return label
    return "Unknown"
