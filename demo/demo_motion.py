#!/usr/bin/env python3
"""
Demo: Motion Analysis & Optical Flow

Demonstrates Module 4 concepts:
- Lucas-Kanade sparse optical flow (KLT tracker)
- Farneback dense optical flow
- MOG2 background subtraction
- Object tracking with centroid tracker
- Speed estimation

Usage:
    python demo/demo_motion.py --input samples/traffic.mp4 --output results/
"""

import argparse
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.motion_analysis import (
    LucasKanadeTracker, compute_dense_optical_flow,
    flow_to_color, compute_motion_magnitude,
    BackgroundSubtractor, CentroidTracker, estimate_speed
)
from src.segmentation import find_contours
from src.utils import (
    draw_optical_flow_arrows, draw_trajectories,
    add_info_panel, create_comparison
)


def main():
    parser = argparse.ArgumentParser(description="Motion Analysis Demo")
    parser.add_argument("--input", "-i", required=True, help="Input video")
    parser.add_argument("--output", "-o", default="results/", help="Output directory")
    parser.add_argument("--mode", "-m", default="all",
                        choices=["sparse", "dense", "background", "all"],
                        help="Analysis mode")
    parser.add_argument("--max-frames", type=int, default=300,
                        help="Max frames to process")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"[ERROR] Could not open: {args.input}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize components
    lk_tracker = LucasKanadeTracker()
    bg_subtractor = BackgroundSubtractor()
    centroid_tracker = CentroidTracker(max_disappeared=15)

    # Output video writers
    writers = {}
    if args.mode in ["sparse", "all"]:
        writers["sparse"] = cv2.VideoWriter(
            os.path.join(args.output, "motion_sparse.mp4"),
            cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if args.mode in ["dense", "all"]:
        writers["dense"] = cv2.VideoWriter(
            os.path.join(args.output, "motion_dense.mp4"),
            cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if args.mode in ["background", "all"]:
        writers["bg"] = cv2.VideoWriter(
            os.path.join(args.output, "motion_background.mp4"),
            cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    prev_frame = None
    frame_count = 0
    sample_frames = {}

    print(f"[+] Processing video: {args.input}")
    print(f"    Mode: {args.mode} | FPS: {fps:.1f}")

    while frame_count < args.max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # --- Sparse Optical Flow (Lucas-Kanade) ---
        if args.mode in ["sparse", "all"]:
            if frame_count == 0:
                lk_tracker.initialize(frame)
            else:
                good_new, good_old, flow_vectors = lk_tracker.update(frame)
                sparse_vis = frame.copy()

                if len(good_new) > 0:
                    for new, old in zip(good_new, good_old):
                        a, b = int(new[0]), int(new[1])
                        c, d = int(old[0]), int(old[1])
                        cv2.line(sparse_vis, (a, b), (c, d), (0, 255, 0), 2)
                        cv2.circle(sparse_vis, (a, b), 4, (0, 0, 255), -1)

                    avg_motion = np.mean(np.abs(flow_vectors), axis=0)
                    sparse_vis = add_info_panel(sparse_vis, [
                        f"Lucas-Kanade Tracker | Points: {len(good_new)}",
                        f"Avg Motion: dx={avg_motion[0]:.1f}, dy={avg_motion[1]:.1f}"
                    ])

                writers["sparse"].write(sparse_vis)
                lk_tracker.reinitialize(frame, min_points=20)

                if frame_count == 30:
                    sample_frames["sparse"] = sparse_vis

        # --- Dense Optical Flow (Farneback) ---
        if args.mode in ["dense", "all"] and prev_frame is not None:
            flow = compute_dense_optical_flow(prev_frame, frame)
            flow_color = flow_to_color(flow)
            magnitude = compute_motion_magnitude(flow)

            dense_vis = draw_optical_flow_arrows(frame, flow, step=20)

            avg_speed = np.mean(estimate_speed(flow, fps=fps))
            dense_vis = add_info_panel(dense_vis, [
                f"Farneback Dense Flow | Avg Magnitude: {np.mean(magnitude):.1f}",
                f"Est. Speed: {avg_speed:.1f} km/h (uncalibrated)"
            ])

            writers["dense"].write(dense_vis)

            if frame_count == 30:
                sample_frames["dense"] = dense_vis
                sample_frames["flow_color"] = flow_color

        # --- Background Subtraction (MOG2) ---
        if args.mode in ["background", "all"]:
            fg_mask = bg_subtractor.apply(frame)

            # Find moving objects
            contours = find_contours(fg_mask, min_area=500)
            centroids = []
            bg_vis = frame.copy()

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                cx, cy = x + w // 2, y + h // 2
                centroids.append((cx, cy))
                cv2.rectangle(bg_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Update tracker
            objects = centroid_tracker.update(centroids)
            bg_vis = draw_trajectories(bg_vis, centroid_tracker.trajectories)
            bg_vis = add_info_panel(bg_vis, [
                f"MOG2 Background Subtraction | Moving Objects: {len(objects)}",
                f"Tracked IDs: {list(objects.keys())}"
            ])

            writers["bg"].write(bg_vis)

            if frame_count == 60:
                sample_frames["bg"] = bg_vis

        prev_frame = frame.copy()
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"    Frame {frame_count}...")

    cap.release()
    for w in writers.values():
        w.release()

    # Save sample comparison
    if sample_frames:
        frames = list(sample_frames.values())[:4]
        titles = list(sample_frames.keys())[:4]
        create_comparison(
            frames, titles,
            save_path=os.path.join(args.output, "motion_comparison.png"),
            figsize=(20, 5)
        )

    print(f"\n[✓] Motion analysis complete! Results saved to {args.output}")
    print(f"    Processed {frame_count} frames")


if __name__ == "__main__":
    main()
