"""
Lane Detection Pipeline

Combines multiple modules to detect lane lines in road images/video:
1. Preprocessing: grayscale, CLAHE, Gaussian blur
2. Edge detection: Canny with auto-thresholding
3. ROI masking: Focus on road region only
4. Line detection: Probabilistic Hough Transform
5. Lane fitting: Separate left/right lanes, average lines
6. Overlay: Draw detected lanes on original image

Integrates concepts from Modules 1 and 3 of the syllabus.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List

from src.preprocessing import clahe_enhancement, gaussian_blur, to_grayscale
from src.feature_extraction import canny_edge_detection, hough_lines_probabilistic
from src.segmentation import region_of_interest


def create_road_roi(image: np.ndarray,
                    roi_ratio: float = 0.6) -> np.ndarray:
    """
    Create a trapezoidal ROI mask for the road region.

    The road in a forward-facing camera typically occupies a
    trapezoidal region in the lower portion of the frame.

    Args:
        image: Input edge/binary image
        roi_ratio: How far up the ROI extends (0.6 = top 40% masked)
    """
    h, w = image.shape[:2]
    vertices = np.array([
        [
            (int(w * 0.05), h),            # Bottom-left
            (int(w * 0.40), int(h * roi_ratio)),  # Top-left
            (int(w * 0.60), int(h * roi_ratio)),  # Top-right
            (int(w * 0.95), h),            # Bottom-right
        ]
    ], dtype=np.int32)

    return region_of_interest(image, vertices[0])


def separate_left_right_lines(lines: np.ndarray,
                               image_width: int) -> Tuple[List, List]:
    """
    Separate detected lines into left and right lane candidates.

    Uses the slope of each line:
    - Negative slope = left lane (line goes from bottom-left to top-right)
    - Positive slope = right lane (line goes from bottom-right to top-left)

    Also filters out near-horizontal lines (|slope| < 0.3) which are
    likely noise, not lane markings.
    """
    left_lines = []
    right_lines = []

    midpoint = image_width // 2

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue

        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        # Filter near-horizontal lines
        if abs(slope) < 0.3:
            continue

        # Length weighting (longer lines are more reliable)
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        if slope < 0 and x1 < midpoint and x2 < midpoint:
            left_lines.append((slope, intercept, length))
        elif slope > 0 and x1 > midpoint and x2 > midpoint:
            right_lines.append((slope, intercept, length))

    return left_lines, right_lines


def average_lane_line(lines: List,
                      y_start: int,
                      y_end: int) -> Optional[Tuple[int, int, int, int]]:
    """
    Compute a single averaged lane line from multiple detections.

    Averages slope and intercept (weighted by line length) to get
    the best estimate of the true lane line position.
    """
    if not lines:
        return None

    # Weighted average by line length
    total_weight = sum(l[2] for l in lines)
    if total_weight == 0:
        return None

    avg_slope = sum(l[0] * l[2] for l in lines) / total_weight
    avg_intercept = sum(l[1] * l[2] for l in lines) / total_weight

    if avg_slope == 0:
        return None

    x1 = int((y_start - avg_intercept) / avg_slope)
    x2 = int((y_end - avg_intercept) / avg_slope)

    return (x1, y_start, x2, y_end)


def draw_lane_overlay(image: np.ndarray,
                      left_line: Optional[Tuple],
                      right_line: Optional[Tuple],
                      color: Tuple[int, int, int] = (0, 255, 0),
                      thickness: int = 5,
                      fill_alpha: float = 0.3) -> np.ndarray:
    """
    Draw detected lane lines and filled lane region on the image.

    Creates a semi-transparent green overlay between the two lane
    lines to clearly show the detected driving lane.
    """
    overlay = image.copy()
    lane_image = np.zeros_like(image)

    # Draw individual lane lines
    if left_line is not None:
        cv2.line(lane_image, (left_line[0], left_line[1]),
                 (left_line[2], left_line[3]), color, thickness)

    if right_line is not None:
        cv2.line(lane_image, (right_line[0], right_line[1]),
                 (right_line[2], right_line[3]), color, thickness)

    # Fill the lane region between lines
    if left_line is not None and right_line is not None:
        pts = np.array([
            [left_line[0], left_line[1]],
            [left_line[2], left_line[3]],
            [right_line[2], right_line[3]],
            [right_line[0], right_line[1]],
        ], np.int32)
        cv2.fillPoly(lane_image, [pts], (0, 255, 0))

    # Blend with original
    result = cv2.addWeighted(overlay, 1 - fill_alpha, lane_image, fill_alpha, 0)

    # Draw solid lines on top
    if left_line is not None:
        cv2.line(result, (left_line[0], left_line[1]),
                 (left_line[2], left_line[3]), (0, 0, 255), thickness)
    if right_line is not None:
        cv2.line(result, (right_line[0], right_line[1]),
                 (right_line[2], right_line[3]), (0, 0, 255), thickness)

    return result


class LaneDetector:
    """
    Complete lane detection pipeline.

    Encapsulates the full process from raw image to lane overlay,
    with frame-to-frame smoothing for video stability.
    """

    def __init__(self, roi_ratio: float = 0.6,
                 smoothing_frames: int = 5):
        self.roi_ratio = roi_ratio
        self.smoothing_frames = smoothing_frames
        self.left_history = []
        self.right_history = []

    def _smooth_line(self, new_line, history):
        """Temporal smoothing using a rolling average of recent detections."""
        if new_line is not None:
            history.append(new_line)

        if len(history) > self.smoothing_frames:
            history.pop(0)

        if not history:
            return None

        # Average the recent lines
        avg = tuple(int(np.mean([h[i] for h in history])) for i in range(4))
        return avg

    def detect(self, image: np.ndarray) -> dict:
        """
        Run the full lane detection pipeline on a single image/frame.

        Returns dict with:
        - result_image: Original image with lane overlay
        - edges: Canny edge detection result
        - roi_edges: Edges after ROI masking
        - left_line, right_line: Detected lane line coordinates
        - lane_detected: Whether both lanes were found
        """
        h, w = image.shape[:2]

        # Step 1: Preprocessing
        enhanced = clahe_enhancement(image)
        blurred = gaussian_blur(enhanced, kernel_size=5)

        # Step 2: Edge detection
        edges = canny_edge_detection(blurred)

        # Step 3: ROI masking
        roi_edges = create_road_roi(edges, self.roi_ratio)

        # Step 4: Hough line detection
        lines = hough_lines_probabilistic(
            roi_edges,
            threshold=30,
            min_line_length=40,
            max_line_gap=150
        )

        left_line = None
        right_line = None

        if lines is not None and len(lines) > 0:
            # Step 5: Separate and average lanes
            left_lines, right_lines = separate_left_right_lines(lines, w)

            y_start = h
            y_end = int(h * self.roi_ratio)

            raw_left = average_lane_line(left_lines, y_start, y_end)
            raw_right = average_lane_line(right_lines, y_start, y_end)

            # Temporal smoothing
            left_line = self._smooth_line(raw_left, self.left_history)
            right_line = self._smooth_line(raw_right, self.right_history)

        # Step 6: Draw overlay
        result_image = draw_lane_overlay(image, left_line, right_line)

        return {
            "result_image": result_image,
            "edges": edges,
            "roi_edges": roi_edges,
            "left_line": left_line,
            "right_line": right_line,
            "lane_detected": left_line is not None and right_line is not None,
        }
