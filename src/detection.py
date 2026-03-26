"""
Module 4: Object Detection

Implements pedestrian and vehicle detection using:
- HOG + SVM pedestrian detector (OpenCV's pre-trained model)
- Haar cascade-based vehicle detection
- Non-maximum suppression (NMS) for overlapping detections
- Multi-scale detection for size invariance

Covers syllabus topics:
    Classification: Discriminant Function, Supervised;
    Classifiers: KNN, ANN models; Object detection
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


# ============================================================
# HOG + SVM Pedestrian Detection
# ============================================================

class PedestrianDetector:
    """
    Pedestrian detector using HOG descriptors + pre-trained SVM.

    Based on the Dalal & Triggs (2005) method:
    1. Compute HOG descriptors over a sliding window
    2. Classify each window using a linear SVM
    3. Apply non-maximum suppression to merge overlapping detections

    OpenCV provides a pre-trained SVM model (cv2.HOGDescriptor_getDefaultPeopleDetector)
    trained on the INRIA pedestrian dataset.
    """

    def __init__(self, win_stride: Tuple[int, int] = (8, 8),
                 padding: Tuple[int, int] = (8, 8),
                 scale: float = 1.05):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.win_stride = win_stride
        self.padding = padding
        self.scale = scale

    def detect(self, image: np.ndarray,
               hit_threshold: float = 0,
               group_threshold: int = 2) -> List[dict]:
        """
        Detect pedestrians in an image.

        Returns list of detections, each with:
        - bbox: (x, y, w, h) bounding box
        - confidence: detection weight/confidence
        """
        boxes, weights = self.hog.detectMultiScale(
            image,
            winStride=self.win_stride,
            padding=self.padding,
            scale=self.scale,
            hitThreshold=hit_threshold,
            groupThreshold=group_threshold
        )

        detections = []
        for (x, y, w, h), weight in zip(boxes, weights):
            detections.append({
                "bbox": (int(x), int(y), int(w), int(h)),
                "confidence": float(weight),
                "label": "pedestrian"
            })

        return detections


# ============================================================
# Haar Cascade Vehicle Detection
# ============================================================

class VehicleDetector:
    """
    Vehicle detector using Haar cascade classifier.

    Haar cascades use a series of simple rectangular features
    (edge, line, center-surround) evaluated at multiple scales.
    The cascade structure (AdaBoost) allows rapid rejection of
    non-vehicle windows.
    """

    def __init__(self, cascade_path: Optional[str] = None):
        if cascade_path is None:
            # Use OpenCV's built-in car cascade
            cascade_path = cv2.data.haarcascades + "haarcascade_car.xml"

        self.cascade = cv2.CascadeClassifier(cascade_path)

        if self.cascade.empty():
            print(f"[WARNING] Could not load cascade from {cascade_path}")
            print("         Falling back to frontal face cascade for demo purposes")
            self.cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

    def detect(self, image: np.ndarray,
               scale_factor: float = 1.1,
               min_neighbors: int = 3,
               min_size: Tuple[int, int] = (30, 30)) -> List[dict]:
        """
        Detect vehicles in an image.

        Multi-scale detection: the image is progressively downscaled and
        the fixed-size detector is applied at each scale level.
        """
        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        boxes = self.cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size
        )

        detections = []
        for (x, y, w, h) in boxes:
            detections.append({
                "bbox": (int(x), int(y), int(w), int(h)),
                "confidence": 1.0,
                "label": "vehicle"
            })

        return detections


# ============================================================
# Non-Maximum Suppression
# ============================================================

def non_maximum_suppression(detections: List[dict],
                            overlap_threshold: float = 0.3) -> List[dict]:
    """
    Apply Non-Maximum Suppression (NMS) to remove overlapping boxes.

    When multiple overlapping detections exist for the same object,
    NMS keeps only the one with the highest confidence and removes
    the rest (if IoU > threshold).

    This is a critical post-processing step for any sliding-window
    or multi-scale detector.
    """
    if len(detections) == 0:
        return []

    boxes = np.array([d["bbox"] for d in detections], dtype=np.float32)
    scores = np.array([d["confidence"] for d in detections])

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        intersection = w * h

        iou = intersection / (areas[i] + areas[order[1:]] - intersection)
        remaining = np.where(iou <= overlap_threshold)[0]
        order = order[remaining + 1]

    return [detections[i] for i in keep]


# ============================================================
# Combined Detector
# ============================================================

class RoadObjectDetector:
    """
    Combined detector for road safety analysis.

    Runs both pedestrian and vehicle detectors and merges results.
    """

    def __init__(self):
        self.pedestrian_detector = PedestrianDetector()
        self.vehicle_detector = VehicleDetector()

    def detect_all(self, image: np.ndarray,
                   nms_threshold: float = 0.3) -> List[dict]:
        """Detect both pedestrians and vehicles, apply NMS."""
        pedestrians = self.pedestrian_detector.detect(image)
        vehicles = self.vehicle_detector.detect(image)

        all_detections = pedestrians + vehicles

        # Apply NMS separately per class
        ped_nms = non_maximum_suppression(
            [d for d in all_detections if d["label"] == "pedestrian"],
            nms_threshold
        )
        veh_nms = non_maximum_suppression(
            [d for d in all_detections if d["label"] == "vehicle"],
            nms_threshold
        )

        return ped_nms + veh_nms
