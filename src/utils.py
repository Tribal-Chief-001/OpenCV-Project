"""
Visualization Utilities

Common drawing and display functions used across all modules:
- Bounding box drawing with labels
- Side-by-side image comparison
- Keypoint and match visualization
- Histogram plotting
- Video frame annotation
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


def draw_bounding_boxes(image: np.ndarray,
                        detections: List[dict],
                        colors: dict = None) -> np.ndarray:
    """
    Draw bounding boxes with labels and confidence scores.

    Args:
        image: Input image
        detections: List of dicts with 'bbox', 'label', 'confidence'
        colors: Dict mapping label → BGR color
    """
    if colors is None:
        colors = {
            "pedestrian": (0, 255, 0),    # Green
            "vehicle": (255, 165, 0),      # Orange
            "pothole": (0, 0, 255),        # Red
        }

    result = image.copy()

    for det in detections:
        x, y, w, h = det["bbox"]
        label = det.get("label", "object")
        confidence = det.get("confidence", 0)
        color = colors.get(label, (255, 255, 255))

        # Draw box
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)

        # Draw label background
        text = f"{label}: {confidence:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(result, (x, y - text_h - 8), (x + text_w + 4, y), color, -1)
        cv2.putText(result, text, (x + 2, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return result


def create_comparison(images: List[np.ndarray],
                      titles: List[str],
                      save_path: Optional[str] = None,
                      figsize: Tuple[int, int] = (16, 5)) -> None:
    """
    Create a side-by-side comparison of multiple images.

    Handles both color and grayscale images correctly.
    """
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, img, title in zip(axes, images, titles):
        if len(img.shape) == 3:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(img, cmap='gray')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[+] Saved comparison to {save_path}")

    plt.close(fig)


def plot_histogram(image: np.ndarray,
                   title: str = "Intensity Histogram",
                   save_path: Optional[str] = None) -> None:
    """Plot intensity histogram for grayscale or BGR image."""
    fig, ax = plt.subplots(figsize=(8, 4))

    if len(image.shape) == 3:
        colors = ('b', 'g', 'r')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            ax.plot(hist, color=color, linewidth=1.5)
        ax.set_xlabel("Pixel Value")
        ax.set_ylabel("Frequency")
        ax.legend(["Blue", "Green", "Red"])
    else:
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        ax.plot(hist, color='gray', linewidth=1.5)
        ax.fill_between(range(256), hist.flatten(), alpha=0.3, color='gray')
        ax.set_xlabel("Pixel Value")
        ax.set_ylabel("Frequency")

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlim([0, 256])
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[+] Saved histogram to {save_path}")

    plt.close(fig)


def draw_keypoints_custom(image: np.ndarray,
                          keypoints: list,
                          color: Tuple[int, int, int] = (0, 255, 0),
                          radius: int = 4) -> np.ndarray:
    """Draw keypoints on image with custom styling."""
    result = image.copy()
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        size = int(kp.size / 2) if kp.size > 0 else radius
        cv2.circle(result, (x, y), size, color, 1)
        # Draw orientation line if available
        if kp.angle >= 0:
            angle_rad = np.deg2rad(kp.angle)
            x2 = int(x + size * np.cos(angle_rad))
            y2 = int(y + size * np.sin(angle_rad))
            cv2.line(result, (x, y), (x2, y2), color, 1)
    return result


def draw_optical_flow_arrows(image: np.ndarray,
                             flow: np.ndarray,
                             step: int = 16,
                             scale: float = 3,
                             color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    Draw optical flow as arrows on the image.

    Samples flow vectors at regular grid positions and draws
    arrows showing motion direction and magnitude.
    """
    result = image.copy()
    h, w = image.shape[:2]

    for y in range(0, h, step):
        for x in range(0, w, step):
            fx, fy = flow[y, x]
            magnitude = np.sqrt(fx ** 2 + fy ** 2)
            if magnitude > 1:  # Only draw significant motion
                end_x = int(x + fx * scale)
                end_y = int(y + fy * scale)
                cv2.arrowedLine(result, (x, y), (end_x, end_y),
                                color, 1, tipLength=0.3)

    return result


def draw_trajectories(image: np.ndarray,
                      trajectories: dict,
                      max_points: int = 30) -> np.ndarray:
    """Draw object tracking trajectories as colored paths."""
    result = image.copy()

    # Generate unique colors for each track
    np.random.seed(42)
    colors = {tid: tuple(int(c) for c in np.random.randint(0, 255, 3))
              for tid in trajectories}

    for track_id, points in trajectories.items():
        color = colors.get(track_id, (0, 255, 0))
        recent = points[-max_points:]

        for i in range(1, len(recent)):
            pt1 = tuple(int(v) for v in recent[i - 1])
            pt2 = tuple(int(v) for v in recent[i])
            thickness = max(1, int(i / len(recent) * 3))
            cv2.line(result, pt1, pt2, color, thickness)

        # Draw current position
        if recent:
            current = tuple(int(v) for v in recent[-1])
            cv2.circle(result, current, 5, color, -1)
            cv2.putText(result, f"ID:{track_id}", (current[0] + 8, current[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return result


def add_info_panel(image: np.ndarray,
                   info_lines: List[str],
                   position: str = "top") -> np.ndarray:
    """
    Add an information panel to the image.

    Creates a semi-transparent bar at the top or bottom with text lines.
    """
    result = image.copy()
    h, w = image.shape[:2]
    line_height = 25
    panel_height = len(info_lines) * line_height + 20

    if position == "top":
        y_start = 0
    else:
        y_start = h - panel_height

    # Semi-transparent background
    overlay = result.copy()
    cv2.rectangle(overlay, (0, y_start), (w, y_start + panel_height),
                  (0, 0, 0), -1)
    result = cv2.addWeighted(overlay, 0.6, result, 0.4, 0)

    # Draw text lines
    for i, line in enumerate(info_lines):
        y = y_start + (i + 1) * line_height
        cv2.putText(result, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    return result
