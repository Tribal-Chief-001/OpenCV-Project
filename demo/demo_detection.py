#!/usr/bin/env python3
"""
Demo: Object Detection (Pedestrians & Vehicles)

Demonstrates Module 4 concepts:
- HOG + SVM pedestrian detection
- Haar cascade vehicle detection
- Non-maximum suppression
- Multi-class detection pipeline

Usage:
    python demo/demo_detection.py --input samples/street.jpg --output results/
"""

import argparse
import os
import sys
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detection import RoadObjectDetector
from src.utils import draw_bounding_boxes, create_comparison, add_info_panel


def process_image(image_path, output_dir):
    """Detect objects in a single image."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not load: {image_path}")
        return

    print(f"[+] Processing: {image_path}")

    detector = RoadObjectDetector()
    detections = detector.detect_all(image)

    # Draw results
    result = draw_bounding_boxes(image, detections)

    # Add info panel
    ped_count = sum(1 for d in detections if d["label"] == "pedestrian")
    veh_count = sum(1 for d in detections if d["label"] == "vehicle")
    info = [
        f"Pedestrians: {ped_count} | Vehicles: {veh_count}",
        f"Total detections: {len(detections)}"
    ]
    result = add_info_panel(result, info)

    # Save
    create_comparison(
        [image, result],
        ["Original", f"Detection ({len(detections)} objects)"],
        save_path=os.path.join(output_dir, "detection_comparison.png")
    )
    cv2.imwrite(os.path.join(output_dir, "detection_result.png"), result)

    print(f"[+] Found: {ped_count} pedestrians, {veh_count} vehicles")
    for d in detections:
        print(f"    {d['label']}: bbox={d['bbox']}, conf={d['confidence']:.2f}")

    print(f"[✓] Results saved to {output_dir}")


def process_video(video_path, output_dir):
    """Detect objects in a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = os.path.join(output_dir, "detection_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    detector = RoadObjectDetector()
    frame_count = 0

    print(f"[+] Processing video: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect_all(frame)
        result = draw_bounding_boxes(frame, detections)

        ped_count = sum(1 for d in detections if d["label"] == "pedestrian")
        veh_count = sum(1 for d in detections if d["label"] == "vehicle")
        result = add_info_panel(result,
                                [f"Frame {frame_count} | Peds: {ped_count} | Cars: {veh_count}"])

        writer.write(result)
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"    Frame {frame_count}...")

    cap.release()
    writer.release()
    print(f"[✓] Output saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Object Detection Demo")
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
