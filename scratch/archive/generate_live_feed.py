# -*- coding: utf-8 -*-
"""
Generate high-fidelity Image-Based Live Feed snapshots with calibrated 6+6 physical topology
in all 3 modes: Spatial (Clean), Behavioral, and Unified Debug.
"""
import os
import sys
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND = os.path.join(ROOT, "backend")
sys.path.insert(0, ROOT)
sys.path.insert(0, BACKEND)

from core.engine import ClassroomEngine
from core.visualizer import render_classroom_frame

ARTIFACT_DIR = r"C:\Users\rajea\.gemini\antigravity-ide\brain\46e4c7fc-e84f-4e27-9ca4-2f199b24196a"
video_path = os.path.join(ROOT, "classrom video of wit", "videos", "real_classroom.mp4")
if not os.path.exists(video_path):
    video_path = os.path.join(ROOT, "test video", "test video - real classrom.mp4")


def main():
    engine = ClassroomEngine(max_seats=15)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return

    # Calibrate structure
    calib_frames = []
    for _ in range(30):
        ret, f = cap.read()
        if not ret:
            break
        calib_frames.append(f)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    engine.calibrate_structure(calib_frames, W, H)

    # Reset video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    target_frames = [30, 90, 180, 270]

    current_idx = 0
    while cap.isOpened() and current_idx <= max(target_frames):
        ret, frame = cap.read()
        if not ret:
            break
        current_idx += 1

        # Run detection
        faces = engine.process_single_frame(frame, W, H, persist=True)

        if current_idx in target_frames:
            # Mode 3: Unified Debug (primary output)
            rendered_debug = render_classroom_frame(frame, faces, engine, mode="debug", frame_idx=current_idx, fps=fps)
            out_debug = os.path.join(ARTIFACT_DIR, f"live_feed_frame_{current_idx}.jpg")
            cv2.imwrite(out_debug, rendered_debug, [cv2.IMWRITE_JPEG_QUALITY, 94])
            print(f"Saved: {out_debug} (Detections: {len(faces)})")

            if current_idx == 30:
                # Mode 1: Spatial (Clean)
                rendered_spatial = render_classroom_frame(frame, faces, engine, mode="spatial", frame_idx=current_idx, fps=fps)
                out_spatial = os.path.join(ARTIFACT_DIR, f"live_feed_mode1_spatial_frame_30.jpg")
                cv2.imwrite(out_spatial, rendered_spatial, [cv2.IMWRITE_JPEG_QUALITY, 94])
                print(f"Saved: {out_spatial}")

                # Mode 2: Behavioral
                rendered_behav = render_classroom_frame(frame, faces, engine, mode="behavioral", frame_idx=current_idx, fps=fps)
                out_behav = os.path.join(ARTIFACT_DIR, f"live_feed_mode2_behavioral_frame_30.jpg")
                cv2.imwrite(out_behav, rendered_behav, [cv2.IMWRITE_JPEG_QUALITY, 94])
                print(f"Saved: {out_behav}")

    cap.release()
    print("Done generating live feed frames across all 3 modes.")


if __name__ == "__main__":
    main()
