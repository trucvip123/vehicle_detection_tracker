"""Frame quality validation and assessment."""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional, Callable


def calculate_laplacian_variance(frame: np.ndarray) -> float:
    """
    Calculate Laplacian variance to detect blur.
    Higher variance indicates sharper (less blurry) image.

    Args:
        frame (np.ndarray): Input frame (BGR or grayscale)

    Returns:
        float: Laplacian variance (0-1000+, typically 0-500 for normal images)
    """
    if frame is None or frame.size == 0:
        return 0.0

    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    # Calculate Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(laplacian_var)


def calculate_brightness(frame: np.ndarray) -> float:
    """
    Calculate average brightness of frame (0-255).

    Args:
        frame (np.ndarray): Input frame

    Returns:
        float: Average brightness value (0-255)
    """
    if frame is None or frame.size == 0:
        return 0.0

    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    # Calculate average brightness
    brightness = np.mean(gray)
    return float(brightness)


def calculate_contrast(frame: np.ndarray) -> float:
    """
    Calculate contrast using standard deviation of pixel values.
    Higher std dev = higher contrast (0-128, typically 20-100 for normal images).

    Args:
        frame (np.ndarray): Input frame

    Returns:
        float: Contrast measure (standard deviation)
    """
    if frame is None or frame.size == 0:
        return 0.0

    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    # Calculate standard deviation (contrast)
    contrast = np.std(gray)
    return float(contrast)


def calculate_histogram_entropy(frame: np.ndarray) -> float:
    """
    Calculate histogram entropy to measure image information content.
    Higher entropy = more varied pixel values = more visual information.

    Args:
        frame (np.ndarray): Input frame

    Returns:
        float: Histogram entropy (0-8, typically 5-8 for normal images)
    """
    if frame is None or frame.size == 0:
        return 0.0

    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    # Calculate histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()

    # Calculate entropy
    entropy = 0.0
    for p in hist:
        if p > 0:
            entropy -= p * np.log2(p)

    return float(entropy)


def is_frame_mostly_black(frame: np.ndarray, threshold: float = 30.0) -> bool:
    """
    Check if frame is mostly black (very dark).

    Args:
        frame (np.ndarray): Input frame
        threshold (float): Brightness threshold (0-255), default 30

    Returns:
        bool: True if frame is mostly black
    """
    brightness = calculate_brightness(frame)
    return brightness < threshold


def is_frame_overexposed(frame: np.ndarray, threshold: float = 220.0) -> bool:
    """
    Check if frame is overexposed (washed out).

    Args:
        frame (np.ndarray): Input frame
        threshold (float): Brightness threshold (0-255), default 220

    Returns:
        bool: True if frame is overexposed
    """
    brightness = calculate_brightness(frame)
    return brightness > threshold


def is_frame_blurry(frame: np.ndarray, variance_threshold: float = 100.0) -> bool:
    """
    Check if frame is too blurry using Laplacian variance.

    Args:
        frame (np.ndarray): Input frame
        variance_threshold (float): Laplacian variance threshold, default 100

    Returns:
        bool: True if frame is too blurry
    """
    variance = calculate_laplacian_variance(frame)
    return variance < variance_threshold


def is_frame_low_contrast(frame: np.ndarray, contrast_threshold: float = 15.0) -> bool:
    """
    Check if frame has low contrast.

    Args:
        frame (np.ndarray): Input frame
        contrast_threshold (float): Contrast (std dev) threshold, default 15

    Returns:
        bool: True if frame has low contrast
    """
    contrast = calculate_contrast(frame)
    return contrast < contrast_threshold


def assess_frame_quality(
    frame: np.ndarray,
    brightness_min: float = 40.0,
    brightness_max: float = 210.0,
    blur_variance_min: float = 100.0,
    contrast_min: float = 15.0,
    entropy_min: float = 3.0,
    log_func: Optional[Callable[[str], None]] = None,
) -> Dict[str, float]:
    """
    Comprehensively assess frame quality with multiple metrics.

    Args:
        frame (np.ndarray): Input frame
        brightness_min (float): Minimum acceptable brightness (0-255)
        brightness_max (float): Maximum acceptable brightness (0-255)
        blur_variance_min (float): Minimum Laplacian variance for sharpness
        contrast_min (float): Minimum acceptable contrast (std dev)
        entropy_min (float): Minimum acceptable histogram entropy
        log_func (Callable): Optional logging function

    Returns:
        Dict with quality metrics:
        - brightness: 0-255
        - contrast: 0-128
        - blur_variance: 0-1000+
        - entropy: 0-8
        - quality_score: 0-100 (100 = excellent quality)
        - is_valid: bool (True if frame meets all thresholds)
        - issues: List of quality issues detected
    """
    if frame is None or frame.size == 0:
        return {
            "brightness": 0.0,
            "contrast": 0.0,
            "blur_variance": 0.0,
            "entropy": 0.0,
            "quality_score": 0.0,
            "is_valid": False,
            "issues": ["Frame is None or empty"],
        }

    brightness = calculate_brightness(frame)
    contrast = calculate_contrast(frame)
    blur_variance = calculate_laplacian_variance(frame)
    entropy = calculate_histogram_entropy(frame)

    issues = []

    # Check brightness
    if brightness < brightness_min:
        issues.append(f"Too dark (brightness={brightness:.1f})")
    elif brightness > brightness_max:
        issues.append(f"Overexposed (brightness={brightness:.1f})")

    # Check blur
    if blur_variance < blur_variance_min:
        issues.append(f"Too blurry (variance={blur_variance:.1f})")

    # Check contrast
    if contrast < contrast_min:
        issues.append(f"Low contrast (std_dev={contrast:.1f})")

    # Check entropy
    if entropy < entropy_min:
        issues.append(f"Low information content (entropy={entropy:.2f})")

    # Calculate quality score (0-100)
    # Normalize each metric to 0-1, then average
    brightness_score = 1.0 - abs(brightness - (brightness_min + brightness_max) / 2) / (
        (brightness_max - brightness_min) / 2
    )
    brightness_score = max(0.0, min(1.0, brightness_score))

    contrast_score = min(contrast / (contrast_min * 3), 1.0)
    blur_score = min(blur_variance / (blur_variance_min * 2), 1.0)
    entropy_score = min(entropy / (entropy_min * 2), 1.0)

    quality_score = (
        brightness_score * 0.25
        + contrast_score * 0.25
        + blur_score * 0.25
        + entropy_score * 0.25
    ) * 100

    is_valid = len(issues) == 0

    result = {
        "brightness": brightness,
        "contrast": contrast,
        "blur_variance": blur_variance,
        "entropy": entropy,
        "quality_score": quality_score,
        "is_valid": is_valid,
        "issues": issues,
    }

    if log_func and not is_valid:
        issues_str = "; ".join(issues)
        log_func(
            f"[QUALITY] Frame rejected: {issues_str} | Score: {quality_score:.1f}/100"
        )

    return result


def should_process_frame(
    frame: np.ndarray,
    quality_threshold: float = 50.0,
    brightness_min: float = 40.0,
    brightness_max: float = 210.0,
    blur_variance_min: float = 100.0,
    contrast_min: float = 15.0,
    entropy_min: float = 3.0,
    log_func: Optional[Callable[[str], None]] = None,
) -> Tuple[bool, Dict[str, float]]:
    """
    Decide whether a frame should be processed for detection.

    Args:
        frame (np.ndarray): Input frame
        quality_threshold (float): Minimum acceptable quality score (0-100)
        brightness_min (float): Minimum brightness threshold
        brightness_max (float): Maximum brightness threshold
        blur_variance_min (float): Minimum blur variance
        contrast_min (float): Minimum contrast threshold
        entropy_min (float): Minimum entropy threshold
        log_func (Callable): Optional logging function

    Returns:
        Tuple of (should_process: bool, quality_metrics: dict)
    """
    quality_metrics = assess_frame_quality(
        frame,
        brightness_min=brightness_min,
        brightness_max=brightness_max,
        blur_variance_min=blur_variance_min,
        contrast_min=contrast_min,
        entropy_min=entropy_min,
        log_func=log_func,
    )

    should_process = (
        quality_metrics["is_valid"]
        and quality_metrics["quality_score"] >= quality_threshold
    )

    return should_process, quality_metrics


def get_frame_quality_summary(
    quality_metrics: Dict[str, float],
) -> str:
    """
    Generate human-readable frame quality summary.

    Args:
        quality_metrics (Dict): Quality metrics from assess_frame_quality()

    Returns:
        str: Formatted summary string
    """
    summary = (
        f"[QUALITY] Brightness: {quality_metrics['brightness']:.1f}/255 | "
        f"Contrast: {quality_metrics['contrast']:.1f}/128 | "
        f"Blur Var: {quality_metrics['blur_variance']:.1f} | "
        f"Entropy: {quality_metrics['entropy']:.2f}/8 | "
        f"Score: {quality_metrics['quality_score']:.1f}/100"
    )
    if quality_metrics["issues"]:
        summary += f" | Issues: {'; '.join(quality_metrics['issues'])}"
    return summary
