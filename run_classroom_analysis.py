# -*- coding: utf-8 -*-
"""
SmartClass AI — Comprehensive Classroom Video Analyzer
======================================================
Executes end-to-end classroom analysis on a video file and produces the complete
output bundle matching structure_experiment + every classroom student's longitudinal data:
- 00_clean_frame.jpg, 01_furniture_boxes_gd.jpg, 01_people.jpg, 02_furniture.jpg
- 03_depth.jpg, 04_seats.jpg, 05_final_structure.jpg, 06_validation_overlay.jpg
- classroom_topology.json
- classroom_analysis_report.json
- students_data.json
- session_summary.json
- students/ (Individual student reports + occupant crop photos)
- classroom_dashboard.html (Interactive standalone UI)
- CLASSROOM_ANALYSIS_REPORT.md (Markdown report)
"""
from __future__ import annotations
import os
import sys
import io
import time
import argparse
import cv2
import numpy as np

# Force UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(ROOT, "backend")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from core.engine import ClassroomEngine
from core.behavior_tracker import BehaviorTracker
from core.pipeline.comprehensive_analyzer import ComprehensiveExporter


def run_analysis(
    video_path: str,
    output_dir: str = os.path.join(ROOT, "outputs", "structure_experiment"),
    max_frames: int = 150,
    frame_skip: int = 3,
    save_video: bool = False,
):
    print("=" * 70)
    print(" SmartClass AI — Comprehensive Classroom Video Analysis Engine")
    print("=" * 70)
    print(f" Video Path   : {video_path}")
    print(f" Output Dir   : {output_dir}")
    print(f" Max Frames   : {max_frames if max_frames > 0 else 'ALL'}")
    print(f" Frame Skip   : {frame_skip}")
    print("=" * 70)

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur_total = round(total_fr / fps_src, 2)

    print(f" Source Video : {W}x{H} @ {fps_src:.1f} FPS | {total_fr} frames (~{dur_total}s)")

    # 1. Initialize Engine & Calibrate Structure
    print("\n[Step 1] Initializing Physical Structure Topology...")
    engine = ClassroomEngine(max_seats=12)

    # Use pure Structure AI calibration
    engine.calibrate_structure([], W, H)

    # Behavior trackers per seat
    t0 = time.time()
    behavior_trackers = {
        sid: BehaviorTracker(session_start_time=t0)
        for sid in range(1, 13)
    }

    # Best student occupant crops
    student_crops: dict[str, np.ndarray] = {}
    best_crop_confs: dict[str, float] = {}

    limit = max_frames if max_frames > 0 else total_fr
    processed = 0
    frame_idx = 0

    video_writer = None
    if save_video:
        annotated_path = os.path.join(output_dir, "classroom_annotated.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(annotated_path, fourcc, fps_src / frame_skip, (W, H))

    print(f"\n[Step 2] Processing Video & Tracking Student Engagement ({limit} frames)...")

    while True:
        ret, frame = cap.read()
        if not ret or (max_frames > 0 and frame_idx >= max_frames):
            break
        frame_idx += 1

        if frame_idx % frame_skip != 0:
            continue

        processed += 1
        now_ts = time.time()

        # Run inference
        faces = engine.process_single_frame(frame, W, H, persist=True)

        for f in faces:
            sid = f.get("id")
            slabel = f.get("seat_label") or f"S{sid}"
            status = f.get("status", "OK")

            if sid in behavior_trackers:
                act = f.get("activity", "neutral")
                behavior_trackers[sid].record(act, now_ts)

            # Extract clearest occupant crop for this seat
            if status == "OK" and "bbox" in f:
                bbox = f["bbox"]
                # Use person_bbox or face bbox with margin
                pbox = f.get("person_bbox", bbox)
                x1, y1, x2, y2 = [int(v) for v in pbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)

                conf = float(f.get("confidence", 0.8))
                if (x2 - x1) > 60 and (y2 - y1) > 80:
                    if slabel not in best_crop_confs or conf > best_crop_confs[slabel]:
                        crop = frame[y1:y2, x1:x2].copy()
                        student_crops[slabel] = crop
                        best_crop_confs[slabel] = conf

        if video_writer:
            from core.visualizer import draw_student_overlay
            disp = frame.copy()
            for face in faces:
                draw_student_overlay(disp, face)
            video_writer.write(disp)

        if processed % 15 == 0 or frame_idx >= limit:
            pct = (frame_idx / limit) * 100.0
            print(f"  Progress: {pct:5.1f}% | Processed {processed} frames | Active Occupants: {len(faces)}/12", flush=True)

    cap.release()
    if video_writer:
        video_writer.release()

    elapsed = time.time() - t0
    effective_dur = round(frame_idx / fps_src, 2)

    print("\n[Step 3] Exporting Full Structure & Longitudinal Student Data Bundle...")
    exporter = ComprehensiveExporter(out_dir=output_dir)

    session_meta = {
        "session_id": f"classroom_session_{int(time.time())}",
        "video_path": video_path,
        "width": W,
        "height": H,
        "fps": fps_src,
        "duration": effective_dur,
        "frame_skip": frame_skip,
        "processed_frames": processed,
    }

    report = exporter.export_all(
        engine=engine,
        behavior_trackers=behavior_trackers,
        student_crops=student_crops,
        session_metadata=session_meta,
    )

    print("\n" + "=" * 70)
    print(" COMPLETE CLASSROOM ANALYSIS GENERATED SUCCESSFULLY!")
    print("=" * 70)
    print(f" Output Location: {output_dir}")
    print("\n Artifacts Generated:")
    print("   [1] 00_clean_frame.jpg          - Clean reference frame")
    print("   [2] 01_furniture_boxes_gd.jpg   - Furniture boxes candidate overlay")
    print("   [3] 01_people.jpg               - People detection with head & floor anchors")
    print("   [4] 02_furniture.jpg            - Segmented furniture polygons & masks")
    print("   [5] 03_depth.jpg                - Depth Anything V2 relative depth map")
    print("   [6] 04_seats.jpg                - Synthesized physical seats geometry")
    print("   [7] 05_final_structure.jpg      - Benches + rows + seat layout")
    print("   [8] 06_validation_overlay.jpg   - People mapped to physical seats")
    print("   [9] classroom_topology.json     - Ground Truth 1 physical topology")
    print("  [10] students_data.json          - EVERY student's comprehensive record")
    print("  [11] classroom_analysis_report.json - Master combined analysis report")
    print("  [12] session_summary.json        - Executive KPI summary")
    print("  [13] students/                   - Individual student reports & occupant crops")
    print("  [14] classroom_dashboard.html    - Standalone interactive HTML dashboard")
    print("  [15] CLASSROOM_ANALYSIS_REPORT.md - Formatted Markdown report")
    print("=" * 70)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartClass AI — Comprehensive Classroom Video Analyzer")
    parser.add_argument("video", nargs="?", default=None, help="Video file path")
    parser.add_argument("--output", "-o", default=os.path.join(ROOT, "outputs", "structure_experiment"), help="Output directory")
    parser.add_argument("--max-frames", "-n", type=int, default=120, help="Max frames to process (default: 120)")
    parser.add_argument("--frame-skip", "-s", type=int, default=3, help="Frame skip factor (default: 3)")
    parser.add_argument("--save-video", action="store_true", help="Save annotated mp4 video")
    args = parser.parse_args()

    default_vid = os.path.join(ROOT, "test video", "test video - real classrom.mp4")
    v_path = args.video if args.video else default_vid

    run_analysis(
        video_path=v_path,
        output_dir=args.output,
        max_frames=args.max_frames,
        frame_skip=args.frame_skip,
        save_video=args.save_video,
    )
