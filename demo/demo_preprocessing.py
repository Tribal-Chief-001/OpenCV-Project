#!/usr/bin/env python3
"""
Demo: Image Preprocessing Pipeline

Demonstrates Module 1 concepts:
- Histogram equalization (standard + CLAHE)
- Gaussian and bilateral filtering
- Fourier domain filtering
- Perspective transform

Usage:
    python demo/demo_preprocessing.py --input samples/road.jpg --output results/
"""

import argparse
import os
import sys
import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import (
    histogram_equalization, clahe_enhancement, gaussian_blur,
    bilateral_filter, fourier_highpass_filter, perspective_transform,
    get_fourier_spectrum, preprocess_pipeline
)
from src.utils import create_comparison, plot_histogram


def main():
    parser = argparse.ArgumentParser(description="Image Preprocessing Demo")
    parser.add_argument("--input", "-i", required=True, help="Input image path")
    parser.add_argument("--output", "-o", default="results/", help="Output directory")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load image
    image = cv2.imread(args.input)
    if image is None:
        print(f"[ERROR] Could not load image: {args.input}")
        sys.exit(1)

    print(f"[+] Loaded image: {args.input} ({image.shape[1]}x{image.shape[0]})")

    # 1. Histogram Equalization
    print("[+] Applying histogram equalization...")
    hist_eq = histogram_equalization(image)
    clahe = clahe_enhancement(image)

    create_comparison(
        [image, cv2.cvtColor(hist_eq, cv2.COLOR_GRAY2BGR),
         cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR)],
        ["Original", "Histogram Equalization", "CLAHE"],
        save_path=os.path.join(args.output, "01_histogram_equalization.png")
    )

    # 2. Filtering
    print("[+] Applying filters...")
    gauss = gaussian_blur(image, kernel_size=7)
    bilateral = bilateral_filter(image)

    create_comparison(
        [image, gauss, bilateral],
        ["Original", "Gaussian Blur (k=7)", "Bilateral Filter"],
        save_path=os.path.join(args.output, "02_filtering.png")
    )

    # 3. Fourier Analysis
    print("[+] Computing Fourier analysis...")
    spectrum = get_fourier_spectrum(image)
    highpass = fourier_highpass_filter(image, cutoff=30)

    create_comparison(
        [image, cv2.cvtColor(spectrum, cv2.COLOR_GRAY2BGR),
         cv2.cvtColor(highpass, cv2.COLOR_GRAY2BGR)],
        ["Original", "Fourier Spectrum", "High-Pass Filter (edges)"],
        save_path=os.path.join(args.output, "03_fourier_analysis.png")
    )

    # 4. Perspective Transform
    print("[+] Applying perspective transform...")
    warped, _ = perspective_transform(image)

    create_comparison(
        [image, warped],
        ["Original", "Bird's-Eye View (Perspective Transform)"],
        save_path=os.path.join(args.output, "04_perspective_transform.png")
    )

    # 5. Full Pipeline
    print("[+] Running full preprocessing pipeline...")
    pipeline_result = preprocess_pipeline(image)

    create_comparison(
        [image, cv2.cvtColor(pipeline_result, cv2.COLOR_GRAY2BGR)],
        ["Original", "Preprocessed (resize + bilateral + CLAHE)"],
        save_path=os.path.join(args.output, "05_pipeline_result.png")
    )

    # 6. Histograms
    print("[+] Generating histograms...")
    plot_histogram(image, "Original Histogram",
                   save_path=os.path.join(args.output, "06_histogram_original.png"))
    plot_histogram(cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR), "After CLAHE",
                   save_path=os.path.join(args.output, "06_histogram_clahe.png"))

    print(f"\n[✓] All results saved to {args.output}")
    print("    Preprocessing demo complete!")


if __name__ == "__main__":
    main()
