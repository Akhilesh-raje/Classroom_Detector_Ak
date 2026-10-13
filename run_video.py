# -*- coding: utf-8 -*-
"""
SmartClass AI — Video File Runner (Clean Orchestrator)
"""
import sys
import os
import io
import time
import cv2
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "backend"))
sys.path.insert(0, backend_path)

from core.config import YOLO_CONF, PERSIST, SMALL_BBOX_FRAC
from core.engine import ClassroomEngine
from core.student_state import StudentState
from core.visualizer import draw_student_overlay, draw_global_hud, C
from core.behavior_tracker import BehaviorTracker
from core.pipeline.comprehensive_analyzer import ComprehensiveExporter

FRAME_SKIP = 3
MAX_SEATS = 12
VIDEO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "test video"))
OUTPUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "annotated_output.mp4"))
EXPORT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs", "structure_experiment"))
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
DISPLAY_W = 1280

def find_video(directory):
    if not os.path.isdir(directory):
        return None
    for f in sorted(os.listdir(directory)):
        if f.lower().endswith(VIDEO_EXTS):
            return os.path.join(directory, f)
    return None

import argparse

def run():
    parser = argparse.ArgumentParser(description="SmartClass AI — Video File Runner (Clean Orchestrator)")
    parser.add_argument("video", nargs="?", default=None, help="Path to video file")
    parser.add_argument("--output", "-o", default=OUTPUT_PATH, help="Path for annotated output video")
    parser.add_argument("--headless", action="store_true", help="Run without UI display window")
    parser.add_argument("--max-seats", type=int, default=MAX_SEATS, help="Maximum seats to track")
    parser.add_argument("--frame-skip", type=int, default=FRAME_SKIP, help="Frame skip factor for AI processing")
    parser.add_argument("--rows", type=int, default=None, help="Optional row count hint (default: auto-detected)")
    parser.add_argument("--cols", type=int, default=None, help="Optional column count hint (default: auto-detected)")
    parser.add_argument("--grid", action="store_true", help="Force synthetic grid topology instead of auto-discovering from video")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to process (0 = all)")
    parser.add_argument("--export-dir", default=EXPORT_DIR, help="Directory for full structure & student report exports")
    parser.add_argument("--no-export", action="store_true", help="Disable full artifact bundle export")
    args = parser.parse_args()

    print("=" * 60)
    print("SmartClass AI — Video File Runner (Clean Orchestrator)")
    print("=" * 60)

    if args.video:
        video_path = os.path.abspath(args.video)
        if not os.path.isfile(video_path):
            print(f"ERROR: File not found: '{video_path}'")
            return
    else:
        video_path = find_video(VIDEO_DIR)
        if not video_path:
            print(f"ERROR: No video found in '{VIDEO_DIR}'")
            return

    output_path = os.path.abspath(args.output)
    print(f"Video  : {os.path.basename(video_path)}")
    print(f"Output : {output_path}")
    print(f"Mode   : {'Headless' if args.headless else 'Interactive Preview'}")
    print("=" * 60)

    cap = cv2.VideoCapture(video_path)
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps_src, (W, H))

    WIN = "SmartClass AI — Video Processing Preview"
    if not args.headless:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, DISPLAY_W, int(DISPLAY_W * H / W))

    engine = ClassroomEngine(max_seats=args.max_seats)

    # Topology initialization: Auto-discovery (default) or synthetic grid
    if args.grid and args.rows and args.cols:
        from core.topology import create_classroom_grid_topology
        topo = create_classroom_grid_topology(rows=args.rows, cols=args.cols, frame_size=(W, H))
        engine.apply_topology(topo)
        print(f"  Locked Synthetic Grid Topology: {args.rows} Rows x {args.cols} Columns ({len(topo.seats)} Seats)")
    else:
        print("  Calibrating video spatial topology (auto-discovery)...", flush=True)
        calib_frames = []
        sample_limit = min(150, total_fr)
        for f_i in range(sample_limit):
            ret, cf = cap.read()
            if not ret:
                break
            if f_i % 2 == 0 and len(calib_frames) < 60:
                calib_frames.append(cf)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        from core.topology import TopologyConfig
        cfg_cols = [args.cols] * args.rows if (args.rows and args.cols) else None
        topo_cfg = TopologyConfig(mode="auto", rows=args.rows, cols_per_row=cfg_cols)
        engine.calibrate(calib_frames, W, H, config=topo_cfg)

        if engine.topology:
            print(f"  Discovered Seating Topology: {engine.topology.rows} Rows, {len(engine.topology.seats)} Physical Seats")
            for rg in engine.topology.row_geometries:
                seats_in_row = engine.topology.get_seats_for_row(rg.row_idx)
                seat_labels = [s.label for s in seats_in_row]
                print(f"    {rg.label} ({rg.semantic}): {len(seats_in_row)} seats -> {seat_labels}")

    total_seats = len(engine.topology.seats) if engine.topology else args.max_seats
    layout_name = f"{engine.topology.rows} Rows x {max(engine.topology.cols_per_row)} Cols" if engine.topology else "Auto Layout"

    frame_idx = 0
    processed = 0
    detections = 0
    last_faces = []
    t0 = time.time()
    fps_buf = []

    behavior_trackers = {sid: BehaviorTracker(session_start_time=t0) for sid in range(1, total_seats + 1)}
    student_crops = {}
    best_confs = {}

    limit_frames = args.max_frames if args.max_frames > 0 else total_fr

    while True:
        ret, frame = cap.read()
        if not ret or (args.max_frames > 0 and frame_idx >= args.max_frames):
            break
        frame_idx += 1
        t_frame = time.time()

        if frame_idx % args.frame_skip == 0:
            processed += 1
            faces = engine.process_single_frame(frame, W, H, persist=True)
            last_faces = faces
            detections += len(faces)

            now_ts = time.time()
            for f in faces:
                sid = f.get("id")
                slabel = f.get("seat_label") or f"S{sid}"
                if sid in behavior_trackers:
                    behavior_trackers[sid].record(f.get("activity", "neutral"), now_ts)
                if f.get("status") == "OK" and "bbox" in f:
                    pbox = f.get("person_bbox", f["bbox"])
                    x1, y1, x2, y2 = [int(v) for v in pbox]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(W, x2), min(H, y2)
                    conf = float(f.get("confidence", 0.8))
                    if (x2 - x1) > 60 and (y2 - y1) > 80:
                        if slabel not in best_confs or conf > best_confs[slabel]:
                            student_crops[slabel] = frame[y1:y2, x1:x2].copy()
                            best_confs[slabel] = conf

        for face in last_faces:
            draw_student_overlay(frame, face)

        elapsed = time.time() - t0
        fps_buf.append(time.time() - t_frame)
        if len(fps_buf) > 20:
            fps_buf.pop(0)
        live_fps = 1.0 / max(0.001, sum(fps_buf) / len(fps_buf))

        draw_global_hud(frame, last_faces, {
            "total_frames": limit_frames,
            "elapsed": elapsed,
            "fps": live_fps,
            "mode": f"{layout_name} ({len(last_faces)}/{total_seats} students)",
            "frame_idx": frame_idx,
        })

        out.write(frame)

        if not args.headless:
            cv2.imshow(WIN, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nQuit by user.")
                break
        else:
            if frame_idx % 30 == 0 or frame_idx == limit_frames:
                pct = frame_idx / max(1, limit_frames) * 100
                det_labels = [f.get("seat_label", "?") for f in last_faces if f.get("status") == "OK"]
                print(f"  [{pct:5.1f}%] Frame {frame_idx}/{limit_frames} | Students ({len(det_labels)}/{total_seats}): {det_labels} | Live FPS: {live_fps:.1f}", flush=True)

    if not args.headless:
        cv2.destroyAllWindows()
    cap.release()
    out.release()

    print("\n" + "=" * 60)
    print(f"RESULTS — {layout_name.upper()} ({total_seats} STUDENTS)")
    print("=" * 60)
    print(f"  Total frames read  : {frame_idx}")
    print(f"  Frames AI processed: {processed}")
    print(f"  Unique seat labels : {[engine._seat_id_to_label.get(i, f'S{i}') for i in sorted(engine.student_states.keys())]}")
    print(f"  Total time         : {time.time() - t0:.2f}s")
    print(f"  Annotated video    : {output_path}")
    if engine.student_states:
        print()
        print(f"  {'Seat':<8} {'Internal':<10} {'Avg Attn':>9} {'Peak':>7} {'Activity':<28}")
        print("  " + "-" * 62)
        for sid, st in sorted(engine.student_states.items()):
            seat_lbl = engine._seat_id_to_label.get(sid, f"S{sid}")
            rec_act, note = st.reconciled_activity()
            display = f"{rec_act}" + (f" ({note})" if note else "")
            print(f"  {seat_lbl:<8} (Seat {sid}) {st.avg:>8.1f}%  {st.peak:>6.1f}%  {display:<28}")
    print("=" * 60)

    # ── Export Comprehensive Artifacts & Student Reports ──────────────────────
    if not args.no_export and args.export_dir:
        print(f"\nExporting complete classroom analytics to: {args.export_dir}")
        exporter = ComprehensiveExporter(out_dir=args.export_dir)
        session_meta = {
            "session_id": f"session_{int(time.time())}",
            "video_path": video_path,
            "width": W,
            "height": H,
            "fps": fps_src,
            "duration": round(frame_idx / fps_src, 2),
            "frame_skip": args.frame_skip,
            "processed_frames": processed,
        }
        exporter.export_all(
            engine=engine,
            behavior_trackers=behavior_trackers,
            student_crops=student_crops,
            session_metadata=session_meta,
        )

if __name__ == "__main__":
    run()
