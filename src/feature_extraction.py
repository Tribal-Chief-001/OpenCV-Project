"""
Module 3: Feature Extraction

Implements various feature detection and descriptor extraction methods:
- Edge detection: Canny (auto-threshold), Laplacian of Gaussian (LOG)
- Line detection: Hough Transform (standard + probabilistic)
- Corner detection: Harris corners, Shi-Tomasi (Good Features to Track)
- Keypoint descriptors: SIFT, ORB
- Region descriptors: HOG (Histogram of Oriented Gradients)
- Feature matching: BFMatcher, FLANN-based matching

Covers syllabus topics:
    Edges - Canny, LOG, DOG; Line detectors (Hough Transform),
    Corners - Harris and Hessian Affine, Orientation Histogram,
    SIFT, SURF, HOG, GLOH, Scale-Space Analysis
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional

from src.preprocessing import to_grayscale


# ============================================================
# Edge Detection
# ============================================================

def canny_edge_detection(image: np.ndarray,
                         low_threshold: Optional[int] = None,
                         high_threshold: Optional[int] = None,
                         sigma: float = 0.33) -> np.ndarray:
    """
    Apply Canny edge detection with automatic threshold computation.

    If thresholds are not provided, uses the median intensity and sigma
    to compute optimal thresholds automatically (Otsu-inspired approach).

    The Canny detector works in 4 stages:
    1. Gaussian smoothing (noise reduction)
    2. Gradient computation (Sobel in x and y)
    3. Non-maximum suppression (thin edges)
    4. Hysteresis thresholding (connect strong + weak edges)
    """
    gray = to_grayscale(image)

    if low_threshold is None or high_threshold is None:
        median = np.median(gray)
        low_threshold = int(max(0, (1.0 - sigma) * median))
        high_threshold = int(min(255, (1.0 + sigma) * median))

    return cv2.Canny(gray, low_threshold, high_threshold)


def laplacian_of_gaussian(image: np.ndarray,
                          kernel_size: int = 5,
                          sigma: float = 1.0) -> np.ndarray:
    """
    Apply Laplacian of Gaussian (LOG) edge detection.

    LOG combines Gaussian smoothing with Laplacian second-derivative
    detection, finding zero-crossings that correspond to edges.
    More robust than raw Laplacian due to noise suppression.
    """
    gray = to_grayscale(image)
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    result = np.uint8(np.abs(laplacian))
    return result


def difference_of_gaussian(image: np.ndarray,
                           sigma1: float = 1.0,
                           sigma2: float = 2.0) -> np.ndarray:
    """
    Apply Difference of Gaussian (DOG) for edge/blob detection.

    DOG approximates the LOG by subtracting two Gaussian-blurred
    versions of the image at different scales. This is the basis
    of SIFT's scale-space analysis.
    """
    gray = to_grayscale(image)
    k1 = int(6 * sigma1) | 1  # Ensure odd kernel size
    k2 = int(6 * sigma2) | 1
    blur1 = cv2.GaussianBlur(gray, (k1, k1), sigma1)
    blur2 = cv2.GaussianBlur(gray, (k2, k2), sigma2)
    dog = cv2.subtract(blur1, blur2)
    return dog


# ============================================================
# Line Detection (Hough Transform)
# ============================================================

def hough_lines(edge_image: np.ndarray,
                rho: float = 1,
                theta: float = np.pi / 180,
                threshold: int = 50) -> Optional[np.ndarray]:
    """
    Apply Standard Hough Line Transform.

    Maps each edge point to a sinusoidal curve in (rho, theta) space.
    Intersections of curves correspond to collinear points — i.e., lines.

    Returns array of (rho, theta) pairs for detected lines.
    """
    return cv2.HoughLines(edge_image, rho, theta, threshold)


def hough_lines_probabilistic(edge_image: np.ndarray,
                               rho: float = 1,
                               theta: float = np.pi / 180,
                               threshold: int = 50,
                               min_line_length: int = 50,
                               max_line_gap: int = 10) -> Optional[np.ndarray]:
    """
    Apply Probabilistic Hough Line Transform.

    More efficient variant that returns actual line endpoints (x1,y1,x2,y2)
    instead of (rho, theta). Also provides control over minimum line length
    and maximum allowed gap between line segments.
    """
    return cv2.HoughLinesP(edge_image, rho, theta, threshold,
                           minLineLength=min_line_length,
                           maxLineGap=max_line_gap)


# ============================================================
# Corner Detection
# ============================================================

def harris_corners(image: np.ndarray,
                   block_size: int = 2,
                   ksize: int = 3,
                   k: float = 0.04,
                   threshold_ratio: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Harris corner detection.

    Harris detector computes a corner response function based on the
    eigenvalues of the structure tensor (second moment matrix).
    Corners have large response values (both eigenvalues are large).

    Returns:
        corners: Boolean mask of corner locations
        response: Harris corner response map (R values)
    """
    gray = to_grayscale(image).astype(np.float32)
    response = cv2.cornerHarris(gray, block_size, ksize, k)
    response = cv2.dilate(response, None)  # Dilate for marking

    threshold = threshold_ratio * response.max()
    corners = response > threshold
    return corners, response


def shi_tomasi_corners(image: np.ndarray,
                       max_corners: int = 100,
                       quality_level: float = 0.01,
                       min_distance: int = 10) -> np.ndarray:
    """
    Apply Shi-Tomasi corner detection (Good Features to Track).

    Improvement over Harris — uses min(eigenvalue1, eigenvalue2)
    as the corner quality measure instead of the Harris R function.
    Returns the strongest corner points, well-separated from each other.
    """
    gray = to_grayscale(image)
    corners = cv2.goodFeaturesToTrack(gray, max_corners,
                                      quality_level, min_distance)
    if corners is not None:
        corners = corners.reshape(-1, 2)
    return corners


# ============================================================
# Keypoint Descriptors (SIFT, ORB)
# ============================================================

def detect_sift_features(image: np.ndarray,
                         n_features: int = 500) -> Tuple[list, np.ndarray]:
    """
    Detect keypoints and compute SIFT descriptors.

    SIFT (Scale-Invariant Feature Transform) provides:
    - Scale invariance through scale-space extrema detection (DOG)
    - Rotation invariance through orientation assignment
    - 128-dimensional descriptor based on gradient histograms

    This is the gold standard for feature matching and is invariant
    to scale, rotation, and partially to affine transformations.
    """
    gray = to_grayscale(image)
    sift = cv2.SIFT_create(nfeatures=n_features)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors


def detect_orb_features(image: np.ndarray,
                        n_features: int = 500) -> Tuple[list, np.ndarray]:
    """
    Detect keypoints and compute ORB descriptors.

    ORB (Oriented FAST and Rotated BRIEF) is a fast, royalty-free
    alternative to SIFT/SURF. Uses FAST keypoint detector + BRIEF
    descriptor with rotation compensation. Binary descriptor makes
    matching very fast via Hamming distance.
    """
    gray = to_grayscale(image)
    orb = cv2.ORB_create(nfeatures=n_features)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors


def compute_hog_descriptor(image: np.ndarray,
                           win_size: Tuple[int, int] = (64, 128),
                           block_size: Tuple[int, int] = (16, 16),
                           block_stride: Tuple[int, int] = (8, 8),
                           cell_size: Tuple[int, int] = (8, 8),
                           n_bins: int = 9) -> np.ndarray:
    """
    Compute HOG (Histogram of Oriented Gradients) descriptor.

    HOG captures shape and appearance through gradient orientation
    distributions. The image is divided into cells, each producing
    a histogram of gradient directions. Cells are grouped into
    overlapping blocks for normalization.

    This is the fundamental descriptor used for pedestrian detection
    (Dalal & Triggs, 2005).
    """
    gray = to_grayscale(image)
    resized = cv2.resize(gray, win_size)
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride,
                            cell_size, n_bins)
    descriptor = hog.compute(resized)
    return descriptor.flatten()


# ============================================================
# Feature Matching
# ============================================================

def match_features_bf(desc1: np.ndarray,
                      desc2: np.ndarray,
                      descriptor_type: str = "sift",
                      ratio_threshold: float = 0.75) -> List[cv2.DMatch]:
    """
    Match features using Brute-Force matcher with Lowe's ratio test.

    The ratio test (Lowe, 2004) rejects ambiguous matches by comparing
    the distance to the best match vs the second-best match. If the
    ratio is above the threshold, the match is considered ambiguous
    and is rejected.

    Args:
        desc1, desc2: Feature descriptor arrays
        descriptor_type: "sift" (L2 norm) or "orb" (Hamming distance)
        ratio_threshold: Maximum ratio for ratio test (lower = stricter)
    """
    if descriptor_type.lower() in ["sift", "surf"]:
        norm_type = cv2.NORM_L2
    else:
        norm_type = cv2.NORM_HAMMING

    bf = cv2.BFMatcher(norm_type, crossCheck=False)
    matches = bf.knnMatch(desc1, desc2, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)

    return good_matches


def match_features_flann(desc1: np.ndarray,
                         desc2: np.ndarray,
                         ratio_threshold: float = 0.75) -> List[cv2.DMatch]:
    """
    Match features using FLANN (Fast Library for Approximate Nearest Neighbors).

    FLANN is faster than brute-force for large descriptor sets.
    Uses KD-tree for L2 descriptors (SIFT).
    """
    index_params = dict(algorithm=1, trees=5)  # KD-tree
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)

    return good_matches
