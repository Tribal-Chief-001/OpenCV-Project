"""
Module 4: Motion Analysis

Implements motion estimation and analysis techniques:
- Sparse optical flow: Lucas-Kanade (KLT tracker)
- Dense optical flow: Farneback method
- Background subtraction: MOG2 (Mixture of Gaussians)
- Object tracking using centroid-based tracker
- Speed estimation from optical flow magnitude

Covers syllabus topics:
    Background Subtraction and Modeling, Optical Flow, KLT,
    Spatio-Temporal Analysis, Dynamic Stereo,
    Motion parameter estimation
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
from collections import defaultdict

from src.preprocessing import to_grayscale


# ============================================================
# Lucas-Kanade Optical Flow (Sparse)
# ============================================================

class LucasKanadeTracker:
    """
    Sparse optical flow tracker using the Lucas-Kanade method.

    Tracks a set of feature points across frames. The KLT tracker
    assumes:
    1. Brightness constancy: pixel intensity doesn't change between frames
    2. Small motion: displacement is small between consecutive frames
    3. Spatial coherence: neighboring pixels have similar motion

    Uses a pyramidal approach for handling larger motions.
    """

    def __init__(self, max_corners: int = 100,
                 quality_level: float = 0.3,
                 min_distance: int = 7):
        # Parameters for Shi-Tomasi corner detection
        self.feature_params = dict(
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=7
        )

        # Parameters for Lucas-Kanade optical flow
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        self.prev_gray = None
        self.prev_points = None
        self.tracks = []

    def initialize(self, frame: np.ndarray) -> np.ndarray:
        """Initialize tracker with the first frame."""
        self.prev_gray = to_grayscale(frame)
        self.prev_points = cv2.goodFeaturesToTrack(
            self.prev_gray, mask=None, **self.feature_params
        )
        return self.prev_points

    def update(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Track points to the new frame.

        Returns:
            good_new: Successfully tracked points in new frame
            good_old: Corresponding points from previous frame
            flow_vectors: Motion vectors (dx, dy) for each tracked point
        """
        gray = to_grayscale(frame)

        if self.prev_points is None or len(self.prev_points) == 0:
            self.prev_points = cv2.goodFeaturesToTrack(
                gray, mask=None, **self.feature_params
            )
            self.prev_gray = gray
            return np.array([]), np.array([]), np.array([])

        # Compute optical flow
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_points, None, **self.lk_params
        )

        if next_points is None:
            self.prev_gray = gray
            return np.array([]), np.array([]), np.array([])

        # Select good points (status == 1)
        good_new = next_points[status.flatten() == 1]
        good_old = self.prev_points[status.flatten() == 1]

        # Compute flow vectors
        flow_vectors = good_new - good_old

        # Update state
        self.prev_gray = gray
        self.prev_points = good_new.reshape(-1, 1, 2)

        return good_new, good_old, flow_vectors

    def reinitialize(self, frame: np.ndarray, min_points: int = 20):
        """Re-detect features if too many points are lost."""
        if self.prev_points is None or len(self.prev_points) < min_points:
            self.initialize(frame)


# ============================================================
# Farneback Dense Optical Flow
# ============================================================

def compute_dense_optical_flow(prev_frame: np.ndarray,
                                curr_frame: np.ndarray,
                                pyr_scale: float = 0.5,
                                levels: int = 3,
                                winsize: int = 15,
                                iterations: int = 3,
                                poly_n: int = 5,
                                poly_sigma: float = 1.2) -> np.ndarray:
    """
    Compute dense optical flow using the Farneback method.

    Unlike Lucas-Kanade (sparse), Farneback computes flow for EVERY
    pixel using polynomial expansion to approximate the neighborhood
    of each pixel. This gives a complete motion field.

    Returns flow array of shape (H, W, 2) where flow[:,:,0] = dx
    and flow[:,:,1] = dy.
    """
    prev_gray = to_grayscale(prev_frame)
    curr_gray = to_grayscale(curr_frame)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale, levels, winsize, iterations,
        poly_n, poly_sigma, 0
    )
    return flow


def flow_to_color(flow: np.ndarray) -> np.ndarray:
    """
    Convert optical flow to an HSV color visualization.

    - Hue = direction of motion
    - Saturation = 255 (constant)
    - Value = magnitude of motion

    This creates an intuitive color-coded motion map where different
    colors represent different motion directions.
    """
    magnitude, angle = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])

    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[:, :, 0] = angle * 180 / np.pi / 2  # Hue: direction
    hsv[:, :, 1] = 255                        # Saturation: full
    hsv[:, :, 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value: magnitude

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def compute_motion_magnitude(flow: np.ndarray) -> np.ndarray:
    """
    Compute per-pixel motion magnitude from dense flow.

    Returns a grayscale image where brighter = more motion.
    """
    magnitude = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
    normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)


# ============================================================
# Background Subtraction (MOG2)
# ============================================================

class BackgroundSubtractor:
    """
    Background subtraction using MOG2 (Mixture of Gaussians 2).

    Models each background pixel as a mixture of K Gaussian
    distributions. Pixels that don't fit any background Gaussian
    are classified as foreground (moving objects).

    MOG2 is adaptive — it updates the background model over time,
    handling gradual illumination changes.
    """

    def __init__(self, history: int = 500,
                 var_threshold: float = 16,
                 detect_shadows: bool = True):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def apply(self, frame: np.ndarray,
              learning_rate: float = -1) -> np.ndarray:
        """
        Apply background subtraction to get foreground mask.

        Returns binary mask where white = foreground (moving object).
        Applies morphological operations to clean up noise.
        """
        fg_mask = self.bg_subtractor.apply(frame, learningRate=learning_rate)

        # Remove shadows (marked as 127 in MOG2)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Morphological cleanup
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)

        return fg_mask

    def get_background(self) -> np.ndarray:
        """Get the current background model."""
        return self.bg_subtractor.getBackgroundImage()


# ============================================================
# Simple Centroid Tracker
# ============================================================

class CentroidTracker:
    """
    Simple multi-object tracker based on centroid distance.

    Assigns unique IDs to detected objects and tracks them across
    frames by matching centroids using minimum Euclidean distance.

    Tracks are maintained with a history of positions for
    trajectory visualization and speed estimation.
    """

    def __init__(self, max_disappeared: int = 30):
        self.next_id = 0
        self.objects: Dict[int, np.ndarray] = {}
        self.disappeared: Dict[int, int] = {}
        self.trajectories: Dict[int, List[np.ndarray]] = defaultdict(list)
        self.max_disappeared = max_disappeared

    def register(self, centroid: np.ndarray):
        """Register a new object with a unique ID."""
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.trajectories[self.next_id].append(centroid.copy())
        self.next_id += 1

    def deregister(self, object_id: int):
        """Remove a tracked object."""
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, detections: List[Tuple[int, int]]) -> Dict[int, np.ndarray]:
        """
        Update tracker with new detections.

        Matches new detections to existing tracks using minimum
        Euclidean distance. Handles appearing/disappearing objects.
        """
        if len(detections) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects

        input_centroids = np.array(detections)

        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = np.array(list(self.objects.values()))

            # Compute distance matrix
            from scipy.spatial.distance import cdist
            distances = cdist(object_centroids, input_centroids)

            # Find minimum distance assignments
            rows = distances.min(axis=1).argsort()
            cols = distances.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if distances[row, col] > 100:  # Max distance threshold
                    continue

                obj_id = object_ids[row]
                self.objects[obj_id] = input_centroids[col]
                self.trajectories[obj_id].append(input_centroids[col].copy())
                self.disappeared[obj_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            # Handle unmatched existing objects
            for row in range(len(object_ids)):
                if row not in used_rows:
                    obj_id = object_ids[row]
                    self.disappeared[obj_id] += 1
                    if self.disappeared[obj_id] > self.max_disappeared:
                        self.deregister(obj_id)

            # Register new detections
            for col in range(len(input_centroids)):
                if col not in used_cols:
                    self.register(input_centroids[col])

        return self.objects


# ============================================================
# Speed Estimation
# ============================================================

def estimate_speed(flow: np.ndarray,
                   fps: float = 30.0,
                   pixels_per_meter: float = 10.0) -> np.ndarray:
    """
    Estimate approximate speed from optical flow magnitude.

    Converts pixel displacement per frame to meters per second
    using the given calibration factor.

    Note: This is an approximation — accurate speed estimation
    requires proper camera calibration and known scene geometry
    (perspective effects make distant objects appear slower).

    Args:
        flow: Dense optical flow (H, W, 2)
        fps: Camera frame rate
        pixels_per_meter: Calibration factor (pixels per real-world meter)
    """
    magnitude = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)

    # Convert: pixels/frame → meters/second
    speed_mps = (magnitude * fps) / pixels_per_meter

    # Convert to km/h
    speed_kmph = speed_mps * 3.6

    return speed_kmph
