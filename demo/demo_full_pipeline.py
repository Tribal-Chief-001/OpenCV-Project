#!/usr/bin/env python3
"""
Demo: Full Road Safety Analysis Pipeline

Runs all analysis modules on a single input:
1. Preprocessing & enhancement
2. Lane detection
3. Pothole detection (on road images)
4. Pedestrian & vehicle detection
5. Motion analysis (for video)

Usage:
    python demo/demo_full_pipeline.py --input samples/road.jpg --output results/
    python demo/demo_full_pipeline.py --input samples/dashcam.mp4 --output results/
"""

import argparse
import os
import sys
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import preprocess_pipeline, clahe_enhancement
from src.feature_extraction import (
    canny_edge_detection, detect_sift_features, harris_corners
)
from src.lane_detection import LaneDetector
from src.pothole_detection import PotholeDetector
from src.detection import RoadObjectDetector
from src.motion_analysis import (
    compute_dense_optical_flow, flow_to_color,
    BackgroundSubtractor, estimate_speed
)
from src.utils import (
    draw_bounding_boxes, draw_keypoints_custom,
    draw_optical_flow_arrows, add_info_panel, create_comparison
)


def analyze_image(image_path, output_dir):
    """Complete analysis pipeline for a single image."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not load: {image_path}")
        return

    h, w = image.shape[:2]
    print(f"[+] FULL PIPELINE - Image: {image_path} ({w}x{h})")
    print("=" * 60)

    results = {}

    # 1. Preprocessing
    print("\n[1/5] Preprocessing...")
    t = time.time()
    preprocessed = preprocess_pipeline(image)
    print(f"      Done ({time.time()-t:.2f}s)")

    # 2. Feature Extraction
    print("[2/5] Feature Extraction...")
    t = time.time()
    edges = canny_edge_detection(image)
    keypoints, descriptors = detect_sift_features(image, n_features=300)
    kp_image = draw_keypoints_custom(image, keypoints, color=(0, 255, 0))
    corners, _ = harris_corners(image)
    corner_image = image.copy()
    corner_image[corners] = [0, 0, 255]
    print(f"      SIFT: {len(keypoints)} keypoints | Harris corners detected")
    print(f"      Done ({time.time()-t:.2f}s)")

    # 3. Lane Detection
    print("[3/5] Lane Detection...")
    t = time.time()
    lane_detector = LaneDetector()
    lane_result = lane_detector.detect(image)
    status = "✓ Detected" if lane_result["lane_detected"] else "✗ Not detected"
    print(f"      Lanes: {status}")
    print(f"      Done ({time.time()-t:.2f}s)")

    # 4. Pothole Detection
    print("[4/5] Pothole Detection...")
    t = time.time()
    pothole_detector = PotholeDetector()
    pothole_result = pothole_detector.detect(image)
    print(f"      {pothole_result['summary']}")
    print(f"      Done ({time.time()-t:.2f}s)")

    # 5. Object Detection
    print("[5/5] Object Detection...")
    t = time.time()
    obj_detector = RoadObjectDetector()
    detections = obj_detector.detect_all(image)
    det_image = draw_bounding_boxes(image, detections)
    ped = sum(1 for d in detections if d["label"] == "pedestrian")
    veh = sum(1 for d in detections if d["label"] == "vehicle")
    print(f"      Pedestrians: {ped} | Vehicles: {veh}")
    print(f"      Done ({time.time()-t:.2f}s)")

    # Save comprehensive results
    print("\n[+] Saving results...")

    # Pipeline overview
    create_comparison(
        [image, cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR),
         cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)],
        ["Original", "Preprocessed", "Edge Detection"],
        save_path=os.path.join(output_dir, "full_01_preprocessing.png")
    )

    create_comparison(
        [kp_image, corner_image],
        [f"SIFT ({len(keypoints)} keypoints)", "Harris Corners"],
        save_path=os.path.join(output_dir, "full_02_features.png")
    )

    create_comparison(
        [lane_result["result_image"], pothole_result["result_image"], det_image],
        ["Lane Detection", "Pothole Detection", "Object Detection"],
        save_path=os.path.join(output_dir, "full_03_analysis.png"),
        figsize=(18, 5)
    )

    # Save individual results
    cv2.imwrite(os.path.join(output_dir, "full_lanes.png"),
                lane_result["result_image"])
    cv2.imwrite(os.path.join(output_dir, "full_potholes.png"),
                pothole_result["result_image"])
    cv2.imwrite(os.path.join(output_dir, "full_detection.png"), det_image)

    print(f"\n{'='*60}")
    print(f"[✓] Full pipeline complete! Results saved to {output_dir}")


def analyze_video(video_path, output_dir, max_frames=500):
    """Complete analysis pipeline for video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = os.path.join(output_dir, "full_pipeline_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    lane_detector = LaneDetector(smoothing_frames=8)
    obj_detector = RoadObjectDetector()
    bg_subtractor = BackgroundSubtractor()
    prev_frame = None
    frame_count = 0

    print(f"[+] FULL PIPELINE - Video: {video_path}")
    print(f"    {total} frames | {fps:.1f} FPS | {width}x{height}")
    print("=" * 60)

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        result = frame.copy()

        # Lane detection
        lane_result = lane_detector.detect(frame)
        result = lane_result["result_image"]

        # Object detection (every 3rd frame for speed)
        if frame_count % 3 == 0:
            detections = obj_detector.detect_all(frame)
            result = draw_bounding_boxes(result, detections)

        # Motion analysis
        if prev_frame is not None:
            flow = compute_dense_optical_flow(prev_frame, frame)
            avg_speed = np.mean(estimate_speed(flow, fps=fps))

            fg_mask = bg_subtractor.apply(frame)
            moving_px = np.count_nonzero(fg_mask)
            motion_pct = (moving_px / (width * height)) * 100

            result = add_info_panel(result, [
                f"Lane: {'✓' if lane_result['lane_detected'] else '✗'} | "
                f"Speed: {avg_speed:.1f} km/h | Motion: {motion_pct:.1f}%",
                f"Frame: {frame_count}/{min(total, max_frames)}"
            ])

        writer.write(result)
        prev_frame = frame.copy()
        frame_count += 1

        if frame_count % 50 == 0:
            print(f"    Frame {frame_count}/{min(total, max_frames)}...")

    cap.release()
    writer.release()

    print(f"\n{'='*60}")
    print(f"[✓] Full pipeline complete! Output: {output_path}")
    print(f"    Processed {frame_count} frames")


def main():
    parser = argparse.ArgumentParser(description="Full Road Safety Pipeline")
    parser.add_argument("--input", "-i", required=True, help="Input image/video")
    parser.add_argument("--output", "-o", default="results/", help="Output directory")
    parser.add_argument("--max-frames", type=int, default=500, help="Max video frames")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.input.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        analyze_video(args.input, args.output, args.max_frames)
    else:
        analyze_image(args.input, args.output)


if __name__ == "__main__":
    main()
