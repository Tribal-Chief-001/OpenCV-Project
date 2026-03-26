"""
Module 1: Digital Image Formation & Low-Level Processing

Implements fundamental image preprocessing techniques:
- Grayscale conversion and color space transforms
- Histogram equalization (standard + CLAHE)
- Gaussian blur and bilateral filtering
- Fourier domain filtering (high-pass edge enhancement)
- Perspective transform (bird's-eye view)
- Image normalization and resizing

Covers syllabus topics:
    Fourier Transform, Convolution and Filtering,
    Image Enhancement, Restoration, Histogram Processing
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def to_hsv(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to HSV color space."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Apply standard histogram equalization to enhance contrast.

    Spreads pixel intensities across the full [0, 255] range,
    improving visibility in under/over-exposed images.
    """
    gray = to_grayscale(image)
    return cv2.equalizeHist(gray)


def clahe_enhancement(image: np.ndarray,
                      clip_limit: float = 3.0,
                      tile_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Unlike standard histogram equalization, CLAHE operates on small tiles
    and limits contrast amplification, preventing noise over-amplification.
    This is crucial for road images where lighting varies across the frame.

    Args:
        image: Input image (BGR or grayscale)
        clip_limit: Threshold for contrast limiting (higher = more contrast)
        tile_size: Size of grid tiles for localized equalization
    """
    gray = to_grayscale(image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return clahe.apply(gray)


def gaussian_blur(image: np.ndarray,
                  kernel_size: int = 5) -> np.ndarray:
    """
    Apply Gaussian blur for noise reduction.

    Uses a Gaussian kernel to smooth the image — essential as a
    preprocessing step before edge detection (reduces false edges).
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def bilateral_filter(image: np.ndarray,
                     d: int = 9,
                     sigma_color: float = 75,
                     sigma_space: float = 75) -> np.ndarray:
    """
    Apply bilateral filter — smooths while preserving edges.

    Unlike Gaussian blur, bilateral filtering considers both spatial
    proximity and color similarity, making it ideal for denoising
    road images without blurring edge features like lane lines.
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def fourier_highpass_filter(image: np.ndarray,
                            cutoff: int = 30) -> np.ndarray:
    """
    Apply high-pass filter in the Fourier domain to enhance edges.

    Process:
    1. Compute 2D DFT (Discrete Fourier Transform)
    2. Shift zero-frequency to center
    3. Apply circular mask to remove low-frequency components
    4. Inverse DFT to get edge-enhanced image

    This implements the Fourier Transform concepts from Module 1.
    """
    gray = to_grayscale(image)
    rows, cols = gray.shape

    # Compute 2D DFT
    dft = np.fft.fft2(gray.astype(np.float32))
    dft_shift = np.fft.fftshift(dft)

    # Create high-pass mask (block low frequencies at center)
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.float32)
    mask[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff] = 0

    # Apply mask and inverse DFT
    filtered = dft_shift * mask
    f_ishift = np.fft.ifftshift(filtered)
    result = np.fft.ifft2(f_ishift)
    result = np.abs(result)

    # Normalize to 0-255 range
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    return result.astype(np.uint8)


def perspective_transform(image: np.ndarray,
                          src_points: Optional[np.ndarray] = None,
                          dst_points: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply perspective transform to get bird's-eye view of the road.

    This is critical for lane detection — viewing the road from above
    makes lane lines parallel, simplifying line fitting.

    Implements Projective Transformation from Module 1.

    Args:
        image: Input image
        src_points: 4 source points (if None, uses default trapezoid)
        dst_points: 4 destination points (if None, uses image corners)

    Returns:
        warped: Transformed image
        M_inv: Inverse transform matrix (to unwarp results back)
    """
    h, w = image.shape[:2]

    if src_points is None:
        # Default: trapezoid covering the road region
        src_points = np.float32([
            [w * 0.15, h],        # Bottom-left
            [w * 0.45, h * 0.6],  # Top-left
            [w * 0.55, h * 0.6],  # Top-right
            [w * 0.85, h],        # Bottom-right
        ])

    if dst_points is None:
        dst_points = np.float32([
            [w * 0.2, h],
            [w * 0.2, 0],
            [w * 0.8, 0],
            [w * 0.8, h],
        ])

    M = cv2.getPerspectiveTransform(src_points, dst_points)
    M_inv = cv2.getPerspectiveTransform(dst_points, src_points)
    warped = cv2.warpPerspective(image, M, (w, h))
    return warped, M_inv


def resize_image(image: np.ndarray,
                 width: Optional[int] = None,
                 height: Optional[int] = None) -> np.ndarray:
    """Resize image while maintaining aspect ratio."""
    h, w = image.shape[:2]

    if width is None and height is None:
        return image

    if width is not None and height is not None:
        return cv2.resize(image, (width, height))

    if width is not None:
        ratio = width / w
        new_h = int(h * ratio)
        return cv2.resize(image, (width, new_h))

    ratio = height / h
    new_w = int(w * ratio)
    return cv2.resize(image, (new_w, height))


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1] float range."""
    return image.astype(np.float32) / 255.0


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """Convert [0, 1] float image back to [0, 255] uint8."""
    return (image * 255).clip(0, 255).astype(np.uint8)


def get_fourier_spectrum(image: np.ndarray) -> np.ndarray:
    """
    Compute and return the Fourier magnitude spectrum for visualization.

    Useful for understanding the frequency content of road images.
    """
    gray = to_grayscale(image)
    dft = np.fft.fft2(gray.astype(np.float32))
    dft_shift = np.fft.fftshift(dft)
    magnitude = np.log1p(np.abs(dft_shift))
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return magnitude.astype(np.uint8)


def preprocess_pipeline(image: np.ndarray,
                        target_width: int = 640) -> np.ndarray:
    """
    Standard preprocessing pipeline for road images.

    Steps:
    1. Resize to standard width (maintains aspect ratio)
    2. Bilateral filter (denoise while preserving edges)
    3. CLAHE enhancement (adaptive contrast)

    This pipeline is used as input to all downstream modules.
    """
    resized = resize_image(image, width=target_width)
    denoised = bilateral_filter(resized)
    enhanced = clahe_enhancement(denoised)
    return enhanced
