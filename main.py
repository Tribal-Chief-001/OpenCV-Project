#!/usr/bin/env python3
"""
Smart Road Safety Analyzer — Main CLI Entry Point

A computer vision system for analyzing road safety conditions.
Processes road images and dashcam footage to detect:
- Lane lines
- Potholes (with severity scoring)
- Pedestrians and vehicles
- Motion patterns and speed estimation

Usage:
    python main.py <command> --input <file> [--output <dir>]

Commands:
    preprocess  - Run image preprocessing pipeline
    features    - Extract and visualize features (SIFT, Harris, edges)
    lanes       - Detect lane lines
    potholes    - Detect and score potholes
    detect      - Detect pedestrians and vehicles
    motion      - Analyze motion (video only)
    full        - Run complete analysis pipeline

Examples:
    python main.py preprocess --input samples/road.jpg
    python main.py lanes --input samples/road.jpg --output results/
    python main.py full --input samples/dashcam.mp4
"""

import argparse
import os
import sys
import time


def cmd_preprocess(args):
    """Run preprocessing pipeline."""
    from demo.demo_preprocessing import main as run
    sys.argv = ['', '--input', args.input, '--output', args.output]
    run()


def cmd_features(args):
    """Extract and visualize features."""
    import cv2
    from src.preprocessing import preprocess_pipeline
    from src.feature_extraction import (
        canny_edge_detection, detect_sift_features, detect_orb_features,
        harris_corners, laplacian_of_gaussian
    )
    from src.utils import draw_keypoints_custom, create_comparison

    os.makedirs(args.output, exist_ok=True)
    image = cv2.imread(args.input)
    if image is None:
        print(f"[ERROR] Could not load: {args.input}")
        return

    print(f"[+] Feature Extraction: {args.input}")

    # Edges
    canny = canny_edge_detection(image)
    log = laplacian_of_gaussian(image)

    # Keypoints
    sift_kp, sift_desc = detect_sift_features(image, n_features=500)
    orb_kp, orb_desc = detect_orb_features(image, n_features=500)

    sift_img = draw_keypoints_custom(image, sift_kp, color=(0, 255, 0))
    orb_img = draw_keypoints_custom(image, orb_kp, color=(0, 165, 255))

    # Corners
    corners, response = harris_corners(image)
    corner_img = image.copy()
    corner_img[corners] = [0, 0, 255]

    create_comparison(
        [cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR),
         cv2.cvtColor(log, cv2.COLOR_GRAY2BGR),
         sift_img, orb_img, corner_img],
        ["Canny Edges", "LOG Edges",
         f"SIFT ({len(sift_kp)})", f"ORB ({len(orb_kp)})", "Harris Corners"],
        save_path=os.path.join(args.output, "features_overview.png"),
        figsize=(25, 5)
    )

    print(f"[+] SIFT: {len(sift_kp)} keypoints | ORB: {len(orb_kp)} keypoints")
    print(f"[✓] Results saved to {args.output}")


def cmd_lanes(args):
    """Run lane detection."""
    from demo.demo_lanes import main as run
    sys.argv = ['', '--input', args.input, '--output', args.output]
    run()


def cmd_potholes(args):
    """Run pothole detection."""
    from demo.demo_potholes import main as run
    sys.argv = ['', '--input', args.input, '--output', args.output]
    run()


def cmd_detect(args):
    """Run object detection."""
    from demo.demo_detection import main as run
    sys.argv = ['', '--input', args.input, '--output', args.output]
    run()


def cmd_motion(args):
    """Run motion analysis."""
    from demo.demo_motion import main as run
    sys.argv = ['', '--input', args.input, '--output', args.output]
    run()


def cmd_full(args):
    """Run full pipeline."""
    from demo.demo_full_pipeline import main as run
    sys.argv = ['', '--input', args.input, '--output', args.output]
    run()


def main():
    parser = argparse.ArgumentParser(
        description="🚗 Smart Road Safety Analyzer — Computer Vision CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s preprocess --input road.jpg
  %(prog)s lanes --input highway.jpg --output results/
  %(prog)s potholes --input pothole.jpg
  %(prog)s detect --input street.jpg
  %(prog)s motion --input traffic.mp4
  %(prog)s full --input dashcam.mp4 --output results/

Project: CSE3010 Computer Vision — BYOP
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Analysis command")

    # Common arguments
    for name, help_text, func in [
        ("preprocess", "Image preprocessing and enhancement", cmd_preprocess),
        ("features", "Feature extraction and visualization", cmd_features),
        ("lanes", "Lane line detection", cmd_lanes),
        ("potholes", "Pothole detection and severity scoring", cmd_potholes),
        ("detect", "Pedestrian and vehicle detection", cmd_detect),
        ("motion", "Motion analysis (video)", cmd_motion),
        ("full", "Complete road safety analysis", cmd_full),
    ]:
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("--input", "-i", required=True, help="Input file path")
        sub.add_argument("--output", "-o", default="results/", help="Output directory")
        sub.set_defaults(func=func)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # Validate input
    if not os.path.exists(args.input):
        print(f"[ERROR] File not found: {args.input}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("  🚗 Smart Road Safety Analyzer")
    print(f"  Command: {args.command}")
    print(f"  Input:   {args.input}")
    print(f"  Output:  {args.output}")
    print("=" * 60)

    start = time.time()
    args.func(args)
    elapsed = time.time() - start

    print(f"\n⏱  Total time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
