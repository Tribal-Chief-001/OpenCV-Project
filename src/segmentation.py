"""
Module 3: Image Segmentation

Implements various segmentation approaches:
- Thresholding: Otsu's, adaptive, HSV-based color segmentation
- Region-based: Watershed segmentation, region growing
- Graph-based: GrabCut foreground/background segmentation
- Contour analysis: Detection, filtering, shape analysis
- Morphological operations: Erosion, dilation, opening, closing

Covers syllabus topics:
    Region Growing, Edge Based approaches to segmentation,
    Graph-Cut, Mean-Shift, Texture Segmentation, Object detection
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional

from src.preprocessing import to_grayscale, to_hsv


# ============================================================
# Thresholding Methods
# ============================================================

def otsu_threshold(image: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Apply Otsu's automatic thresholding.

    Otsu's method finds the optimal threshold that minimizes
    intra-class variance (between foreground and background).
    Works best on bimodal histograms.
    """
    gray = to_grayscale(image)
    threshold_val, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary, threshold_val


def adaptive_threshold(image: np.ndarray,
                       block_size: int = 11,
                       C: int = 2,
                       method: str = "gaussian") -> np.ndarray:
    """
    Apply adaptive thresholding for non-uniform illumination.

    Computes threshold locally for each pixel neighborhood,
    handling images with varying lighting conditions (common in
    outdoor road scenes).
    """
    gray = to_grayscale(image)
    adapt_method = (cv2.ADAPTIVE_THRESH_GAUSSIAN_C if method == "gaussian"
                    else cv2.ADAPTIVE_THRESH_MEAN_C)
    return cv2.adaptiveThreshold(gray, 255, adapt_method,
                                 cv2.THRESH_BINARY, block_size, C)


def color_segmentation(image: np.ndarray,
                       lower_hsv: np.ndarray,
                       upper_hsv: np.ndarray) -> np.ndarray:
    """
    Segment image by HSV color range.

    Useful for isolating specific colored objects (e.g., road markings,
    traffic signs) based on their hue, saturation, and value.
    """
    hsv = to_hsv(image)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    return mask


# ============================================================
# Morphological Operations
# ============================================================

def morphological_cleanup(binary: np.ndarray,
                          kernel_size: int = 5,
                          operation: str = "close") -> np.ndarray:
    """
    Apply morphological operations to clean up binary masks.

    - Opening (erode + dilate): Removes small noise/spots
    - Closing (dilate + erode): Fills small holes/gaps

    Critical post-processing step after thresholding or edge detection.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )

    ops = {
        "open": cv2.MORPH_OPEN,
        "close": cv2.MORPH_CLOSE,
        "erode": cv2.MORPH_ERODE,
        "dilate": cv2.MORPH_DILATE,
        "gradient": cv2.MORPH_GRADIENT,
    }

    return cv2.morphologyEx(binary, ops.get(operation, cv2.MORPH_CLOSE), kernel)


# ============================================================
# Watershed Segmentation
# ============================================================

def watershed_segmentation(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply watershed segmentation.

    Watershed treats the image as a topographic surface where
    intensity = elevation. It "floods" from markers (regional minima)
    and builds dams where different flood basins meet — these dams
    are the segmentation boundaries.

    Process:
    1. Threshold to get sure foreground (distance transform)
    2. Dilate to get sure background
    3. Unknown region = background - foreground
    4. Apply watershed with markers
    """
    gray = to_grayscale(image)

    # Otsu threshold
    _, thresh = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological opening to remove noise
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background (dilated opening)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Sure foreground (distance transform + threshold)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(
        dist_transform, 0.5 * dist_transform.max(), 255, 0
    )
    sure_fg = np.uint8(sure_fg)

    # Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labeling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Apply watershed
    img_color = image if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_color, markers)

    # Create boundary mask (watershed boundaries are marked as -1)
    boundary_mask = np.zeros_like(gray)
    boundary_mask[markers == -1] = 255

    return markers, boundary_mask


# ============================================================
# GrabCut Segmentation
# ============================================================

def grabcut_segmentation(image: np.ndarray,
                         rect: Optional[Tuple[int, int, int, int]] = None,
                         iterations: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply GrabCut foreground/background segmentation.

    GrabCut uses iterative graph-cut optimization with Gaussian Mixture
    Models (GMMs) to separate foreground from background. Initialized
    with a bounding rectangle around the object of interest.

    Args:
        image: Input BGR image
        rect: (x, y, width, height) bounding rectangle for initialization
        iterations: Number of GrabCut iterations

    Returns:
        mask: Segmentation mask (0=BG, 1=FG, 2=probable_BG, 3=probable_FG)
        result: Foreground-extracted image
    """
    if len(image.shape) != 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    if rect is None:
        h, w = image.shape[:2]
        margin = min(h, w) // 10
        rect = (margin, margin, w - 2 * margin, h - 2 * margin)

    cv2.grabCut(image, mask, rect, bgd_model, fgd_model,
                iterations, cv2.GC_INIT_WITH_RECT)

    # Create binary mask (foreground + probable foreground)
    fg_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
    result = image * fg_mask[:, :, np.newaxis]

    return mask, result


# ============================================================
# Contour Analysis
# ============================================================

def find_contours(binary: np.ndarray,
                  min_area: float = 100,
                  max_area: float = None) -> List[np.ndarray]:
    """
    Find and filter contours by area.

    Contours are curves joining continuous points having the same
    intensity — essentially the boundaries of objects. Filtering
    by area removes noise (small contours) and background (huge contours).
    """
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    filtered = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        filtered.append(cnt)

    return filtered


def contour_properties(contour: np.ndarray) -> dict:
    """
    Compute geometric properties of a contour.

    Returns area, perimeter, circularity, bounding box, centroid,
    aspect ratio, extent, and solidity — all useful for shape-based
    object classification.
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    x, y, w, h = cv2.boundingRect(contour)

    moments = cv2.moments(contour)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
    else:
        cx, cy = x + w // 2, y + h // 2

    # Circularity: how close to a circle (1.0 = perfect circle)
    circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0

    # Aspect ratio of bounding box
    aspect_ratio = float(w) / h if h > 0 else 0

    # Extent: ratio of contour area to bounding box area
    bbox_area = w * h
    extent = float(area) / bbox_area if bbox_area > 0 else 0

    # Solidity: ratio of contour area to convex hull area
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0

    return {
        "area": area,
        "perimeter": perimeter,
        "circularity": circularity,
        "bounding_box": (x, y, w, h),
        "centroid": (cx, cy),
        "aspect_ratio": aspect_ratio,
        "extent": extent,
        "solidity": solidity,
    }


def region_of_interest(image: np.ndarray,
                       vertices: np.ndarray) -> np.ndarray:
    """
    Apply a polygonal ROI mask to the image.

    Used in lane detection to focus only on the road region,
    removing sky, trees, and other irrelevant parts.
    """
    mask = np.zeros_like(image)

    if len(image.shape) == 3:
        fill_color = (255, 255, 255)
    else:
        fill_color = 255

    cv2.fillPoly(mask, [vertices], fill_color)
    return cv2.bitwise_and(image, mask)
