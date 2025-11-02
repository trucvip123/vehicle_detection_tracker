"""
Image processing utilities
"""
import cv2
import base64
import numpy as np


class ImageUtils:
    """Utility class for image operations"""
    
    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def increase_brightness(image, factor=1.5):
        """
        Increases the brightness of an image by multiplying its pixels by a factor.

        Args:
            image: The input image in numpy array format.
            factor: The brightness increase factor. A value greater than 1 will increase brightness.

        Returns:
            numpy.ndarray: The image with increased brightness.
        """
        brightened_image = cv2.convertScaleAbs(image, alpha=factor, beta=0)
        return brightened_image

