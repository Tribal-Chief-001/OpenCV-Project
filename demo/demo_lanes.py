#!/usr/bin/env python3
"""
Demo: Lane Detection

Demonstrates Modules 1 + 3 concepts:
- Canny edge detection with auto-thresholding
- Hough Line Transform (probabilistic)
- ROI masking and lane fitting
- Temporal smoothing for video

Usage:
    python demo/demo_lanes.py --input samples/road.jpg --output results/
    python demo/demo_lanes.py --input samples/driving.mp4 --output results/ --video
"""

import argparse
import os
import sys
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lane_detection import LaneDetector
from src.utils import create_comparison


def process_image(image_path, output_dir):
    """Process a single image for lane detection."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not load: {image_path}")
        return

    print(f"[+] Processing image: {image_path}")
    detector = LaneDetector()
    result = detector.detect(image)

    # Save step-by-step results
    create_comparison(
        [image, cv2.cvtColor(result["edges"], cv2.COLOR_GRAY2BGR),
         cv2.cvtColor(result["roi_edges"], cv2.COLOR_GRAY2BGR), result["result_image"]],
        ["Original", "Canny Edges", "ROI Masked", "Lane Detection"],
        save_path=os.path.join(output_dir, "lanes_pipeline.png"),
        figsize=(20, 5)
    )

    cv2.imwrite(os.path.join(output_dir, "lanes_result.png"), result["result_image"])

    status = "DETECTED" if result["lane_detected"] else "PARTIAL/NOT DETECTED"
    print(f"[+] Lane status: {status}")
    print(f"[✓] Results saved to {output_dir}")


def process_video(video_path, output_dir):
    """Process a video for lane detection."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = os.path.join(output_dir, "lanes_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    detector = LaneDetector(smoothing_frames=8)
    frame_count = 0

    print(f"[+] Processing video: {video_path} ({total_frames} frames)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = detector.detect(frame)
        writer.write(result["result_image"])
        frame_count += 1

        if frame_count % 30 == 0:
            print(f"    Frame {frame_count}/{total_frames}...")

    cap.release()
    writer.release()

    print(f"[✓] Output video saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Lane Detection Demo")
    parser.add_argument("--input", "-i", required=True, help="Input image/video")
    parser.add_argument("--output", "-o", default="results/", help="Output directory")
    parser.add_argument("--video", "-v", action="store_true", help="Process as video")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.video or args.input.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        process_video(args.input, args.output)
    else:
        process_image(args.input, args.output)


if __name__ == "__main__":
    main()
