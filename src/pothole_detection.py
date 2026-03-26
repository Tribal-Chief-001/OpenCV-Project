"""
Pothole Detection Pipeline

Detects potholes in road surface images using:
1. Preprocessing: CLAHE, bilateral filter, color space analysis
2. Surface analysis: Texture irregularity via Gabor filters (Module 5)
3. Edge detection: Canny + morphological operations
4. Segmentation: Thresholding + contour analysis
5. Scoring: Severity classification based on contour properties

Integrates concepts from Modules 1, 3, and 5 (Shape from Texture).
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional

from src.preprocessing import (
    clahe_enhancement, bilateral_filter, gaussian_blur,
    to_grayscale, to_hsv
)
from src.feature_extraction import canny_edge_detection
from src.segmentation import (
    morphological_cleanup, find_contours, contour_properties
)


# ============================================================
# Texture Analysis (Module 5: Shape from X)
# ============================================================

def compute_gabor_features(image: np.ndarray,
                           num_orientations: int = 8,
                           frequencies: List[float] = None) -> np.ndarray:
    """
    Compute Gabor filter responses for texture analysis.

    Gabor filters are bandpass filters that respond to specific
    frequencies and orientations — modeling how the human visual
    system processes texture. Road surface irregularities (potholes)
    produce distinctive texture patterns detectable by Gabor filters.

    Implements concepts from Module 5 (Shape from Texture).
    """
    if frequencies is None:
        frequencies = [0.05, 0.1, 0.2, 0.3]

    gray = to_grayscale(image)
    gray = gray.astype(np.float32) / 255.0

    responses = []
    for freq in frequencies:
        for theta_idx in range(num_orientations):
            theta = theta_idx * np.pi / num_orientations
            kernel = cv2.getGaborKernel(
                ksize=(31, 31),
                sigma=4.0,
                theta=theta,
                lambd=1.0 / freq,
                gamma=0.5,
                psi=0,
                ktype=cv2.CV_32F
            )
            filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
            responses.append(filtered)

    # Combine responses: take the maximum response across all filters
    combined = np.max(np.array(responses), axis=0)
    combined = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX)
    return combined.astype(np.uint8)


def texture_variance_map(image: np.ndarray,
                         window_size: int = 15) -> np.ndarray:
    """
    Compute local texture variance map.

    Areas with high variance indicate surface irregularities.
    Smooth road surfaces have low variance, while potholes
    and cracks show high local variance due to shadows and
    uneven surfaces.
    """
    gray = to_grayscale(image).astype(np.float32)

    # Local mean
    mean = cv2.blur(gray, (window_size, window_size))

    # Local variance = E[X^2] - E[X]^2
    sq_mean = cv2.blur(gray ** 2, (window_size, window_size))
    variance = sq_mean - mean ** 2

    # Normalize
    variance = cv2.normalize(variance, None, 0, 255, cv2.NORM_MINMAX)
    return variance.astype(np.uint8)


# ============================================================
# Pothole Detection Pipeline
# ============================================================

def detect_dark_regions(image: np.ndarray,
                        darkness_threshold: float = 0.4) -> np.ndarray:
    """
    Detect unusually dark regions that may indicate potholes.

    Potholes appear darker than surrounding road surface due to:
    - Depth causing shadows
    - Water accumulation
    - Exposed darker subsurface material
    """
    gray = to_grayscale(image)
    enhanced = clahe_enhancement(image)

    # Adaptive threshold for dark regions
    mean_intensity = np.mean(enhanced)
    threshold = int(mean_intensity * darkness_threshold)

    _, dark_mask = cv2.threshold(enhanced, threshold, 255, cv2.THRESH_BINARY_INV)

    # Clean up with morphological operations
    dark_mask = morphological_cleanup(dark_mask, kernel_size=5, operation="open")
    dark_mask = morphological_cleanup(dark_mask, kernel_size=7, operation="close")

    return dark_mask


def score_pothole_severity(properties: dict) -> dict:
    """
    Classify pothole severity based on contour properties.

    Scoring system:
    - Small (area < 500px²): Minor — cosmetic damage risk
    - Medium (500-2000px²): Moderate — vehicle damage risk
    - Large (area > 2000px²): Severe — hazardous, needs repair

    Also considers circularity and aspect ratio for confidence.
    """
    area = properties["area"]
    circularity = properties["circularity"]

    # Size-based severity
    if area < 500:
        severity = "Minor"
        severity_score = 1
        color = (0, 255, 255)  # Yellow
    elif area < 2000:
        severity = "Moderate"
        severity_score = 2
        color = (0, 165, 255)  # Orange
    else:
        severity = "Severe"
        severity_score = 3
        color = (0, 0, 255)    # Red

    # Confidence based on shape (potholes tend to be roughly circular)
    confidence = min(1.0, circularity * 2 + 0.3)

    return {
        "severity": severity,
        "severity_score": severity_score,
        "color": color,
        "confidence": confidence,
        "area_px": area,
    }


class PotholeDetector:
    """
    Complete pothole detection system.

    Combines multiple approaches:
    1. Dark region detection (intensity-based)
    2. Edge detection + morphological closing (structure-based)
    3. Texture analysis (Gabor variance — shape from texture)
    4. Contour analysis + severity scoring

    The multi-approach fusion improves robustness across different
    lighting conditions and road surfaces.
    """

    def __init__(self,
                 min_area: int = 200,
                 max_area: int = 50000,
                 darkness_threshold: float = 0.4):
        self.min_area = min_area
        self.max_area = max_area
        self.darkness_threshold = darkness_threshold

    def detect(self, image: np.ndarray) -> dict:
        """
        Run the full pothole detection pipeline.

        Returns dict with:
        - result_image: Annotated image with marked potholes
        - pothole_mask: Binary mask of detected potholes
        - potholes: List of detected potholes with properties and severity
        - texture_map: Gabor texture analysis result
        - count: Number of potholes detected
        - summary: Text summary of findings
        """
        h, w = image.shape[:2]

        # Step 1: Preprocessing
        denoised = bilateral_filter(image)

        # Step 2: Dark region detection
        dark_mask = detect_dark_regions(denoised, self.darkness_threshold)

        # Step 3: Edge-based detection
        enhanced = clahe_enhancement(denoised)
        edges = canny_edge_detection(enhanced)
        edge_closed = morphological_cleanup(edges, kernel_size=7, operation="close")

        # Step 4: Texture analysis
        texture_map = texture_variance_map(denoised, window_size=15)
        _, texture_mask = cv2.threshold(texture_map, 100, 255, cv2.THRESH_BINARY)

        # Step 5: Combine masks (intersection of dark regions + texture anomaly)
        combined_mask = cv2.bitwise_and(dark_mask, texture_mask)
        combined_mask = morphological_cleanup(combined_mask, kernel_size=5, operation="close")
        combined_mask = morphological_cleanup(combined_mask, kernel_size=3, operation="open")

        # Step 6: Find and analyze contours
        contours = find_contours(combined_mask, self.min_area, self.max_area)

        # Step 7: Score and annotate
        result_image = image.copy()
        potholes = []

        for cnt in contours:
            props = contour_properties(cnt)
            severity = score_pothole_severity(props)

            # Filter by reasonable aspect ratio (very elongated = probably not pothole)
            if props["aspect_ratio"] > 5 or props["aspect_ratio"] < 0.2:
                continue

            pothole_info = {**props, **severity}
            potholes.append(pothole_info)

            # Draw on result image
            x, y, bw, bh = props["bounding_box"]
            color = severity["color"]

            cv2.drawContours(result_image, [cnt], -1, color, 2)
            cv2.rectangle(result_image, (x, y), (x + bw, y + bh), color, 2)

            # Label
            label = f"{severity['severity']} ({props['area']:.0f}px)"
            cv2.putText(result_image, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Summary
        total = len(potholes)
        severe = sum(1 for p in potholes if p["severity"] == "Severe")
        moderate = sum(1 for p in potholes if p["severity"] == "Moderate")
        minor = sum(1 for p in potholes if p["severity"] == "Minor")

        summary = (
            f"Detected {total} potential pothole(s): "
            f"{severe} severe, {moderate} moderate, {minor} minor"
        )

        # Draw summary on image
        cv2.putText(result_image, summary, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return {
            "result_image": result_image,
            "pothole_mask": combined_mask,
            "potholes": potholes,
            "texture_map": texture_map,
            "count": total,
            "summary": summary,
        }
