#!/usr/bin/env python3
"""
Demo: Pothole Detection

Demonstrates Modules 1 + 3 + 5 concepts:
- Texture analysis using Gabor filters (Shape from Texture)
- Dark region detection + edge-based segmentation
- Contour analysis and severity scoring
- Multi-method fusion for robustness

Usage:
    python demo/demo_potholes.py --input samples/pothole.jpg --output results/
"""

import argparse
import os
import sys
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pothole_detection import PotholeDetector
from src.utils import create_comparison


def main():
    parser = argparse.ArgumentParser(description="Pothole Detection Demo")
    parser.add_argument("--input", "-i", required=True, help="Input road image")
    parser.add_argument("--output", "-o", default="results/", help="Output directory")
    parser.add_argument("--sensitivity", type=float, default=0.6,
                        help="Detection sensitivity (0.0 to 1.0, lower is stricter)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    image = cv2.imread(args.input)
    if image is None:
        print(f"[ERROR] Could not load: {args.input}")
        sys.exit(1)

    print(f"[+] Processing image: {args.input}")
    print(f"    Sensitivity: {args.sensitivity}")

    detector = PotholeDetector(darkness_threshold=args.sensitivity)
    result = detector.detect(image)

    # Save step-by-step visualization
    create_comparison(
        [image,
         cv2.cvtColor(result["texture_map"], cv2.COLOR_GRAY2BGR),
         cv2.cvtColor(result["pothole_mask"], cv2.COLOR_GRAY2BGR),
         result["result_image"]],
        ["Original", "Texture Variance Map", "Detection Mask", "Annotated Result"],
        save_path=os.path.join(args.output, "potholes_pipeline.png"),
        figsize=(20, 5)
    )

    cv2.imwrite(os.path.join(args.output, "potholes_result.png"),
                result["result_image"])

    # Print results
    print(f"\n[+] {result['summary']}")
    for i, p in enumerate(result["potholes"]):
        print(f"    Pothole #{i+1}: {p['severity']} "
              f"(area={p['area_px']:.0f}px², "
              f"confidence={p['confidence']:.2f})")

    print(f"\n[✓] Results saved to {args.output}")


if __name__ == "__main__":
    main()
