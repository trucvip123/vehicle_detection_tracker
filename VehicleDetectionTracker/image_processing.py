"""
Image processing utilities for vehicle detection and tracking.
"""
import cv2
import base64
import numpy as np

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
        print(f"Error decoding base64 image: {e}")
        return None

def increase_brightness(image, factor=1.5):
    """
    Increases the brightness of an image.
    
    Args:
        image: Input image
        factor: Brightness increase factor (>1 increases brightness)
        
    Returns:
        numpy.ndarray: Brightened image
    """
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)

def draw_tracking_line(frame, points, color, thickness=2):
    """
    Draw tracking line on frame.
    
    Args:
        frame: Frame to draw on
        points: Array of tracking points
        color: Line color
        thickness: Line thickness
    """
    cv2.polylines(
        frame,
        [points],
        isClosed=False,
        color=color,
        thickness=thickness
    )

def draw_license_plate(frame, box, text, color=(0, 255, 0), text_color=(0, 0, 255)):
    """
    Draw license plate box and text on frame.
    
    Args:
        frame: Frame to draw on
        box: (x1, y1, x2, y2) box coordinates
        text: License plate text to display
        color: Box color
        text_color: Text color
    """
    x1, y1, x2, y2 = box
    
    # Draw rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Draw text
    if text:
        cv2.putText(
            frame,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            text_color,
            2
        )

def draw_plates_corner(frame, plates_dict):
    """
    Draw detected license plates in corner of frame.
    
    Args:
        frame: Frame to draw on
        plates_dict: Dictionary of {track_id: plate_text}
    """
    if not plates_dict:
        return frame
        
    h, w = frame.shape[:2]
    num_plates = len(plates_dict)
    text_height = 30
    padding = 10
    bg_height = num_plates * text_height + padding * 2
    
    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (10, 10),
        (400, bg_height),
        (0, 0, 0),
        -1
    )
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw text for each plate
    y_offset = padding + 25
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    
    for track_id, plate_text in plates_dict.items():
        text = f"Vehicle {track_id}: {plate_text}"
        cv2.putText(
            frame,
            text,
            (20, y_offset),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness
        )
        y_offset += text_height
        
    return frame

def draw_vehicle_info(frame, box, track_id, plate_text=None, speed=None, color=None, model=None):
    """
    Draw vehicle information on frame near the bounding box.
    Information includes:
    - Track ID
    - License plate (if detected)
    - Speed (if calculated)
    - Vehicle color (if classified)
    - Vehicle model (if classified)
    
    Args:
        frame (numpy.ndarray): Frame to draw on
        box (list): Bounding box coordinates [x1, y1, x2, y2]
        track_id (int): Vehicle track ID
        plate_text (str, optional): License plate text
        speed (float, optional): Vehicle speed
        color (str, optional): Vehicle color
        model (str, optional): Vehicle model
    """
    # Get box coordinates
    x1, y1, x2, y2 = [int(i) for i in box]
    
    # Text settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    text_color = (255, 255, 255)  # White color
    
    # Draw track ID
    text = f"ID: {track_id}"
    cv2.putText(frame, text, (x1, y1 - 10), font, font_scale, text_color, thickness)
    
    # Calculate vertical offset for additional text
    y_offset = y1 - 10
    line_height = 30
    
    # Draw license plate if detected
    if plate_text and plate_text != "unknown":
        y_offset -= line_height
        cv2.putText(frame, f"Plate: {plate_text}", (x1, y_offset), font, font_scale, text_color, thickness)
        
    # Draw speed if calculated
    if speed is not None:
        y_offset -= line_height
        cv2.putText(frame, f"Speed: {speed:.1f} km/h", (x1, y_offset), font, font_scale, text_color, thickness)
        
    # Draw color if classified
    if color:
        y_offset -= line_height
        cv2.putText(frame, f"Color: {color}", (x1, y_offset), font, font_scale, text_color, thickness)
        
    # Draw model if classified
    if model:
        y_offset -= line_height
        cv2.putText(frame, f"Model: {model}", (x1, y_offset), font, font_scale, text_color, thickness)
    
    return frame