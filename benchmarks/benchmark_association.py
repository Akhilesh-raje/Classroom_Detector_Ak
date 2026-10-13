# -*- coding: utf-8 -*-
"""
SmartClass AI — Association Benchmark Suite (V2.1)
===================================================
Evaluates and audits seat association quality against the Four Truths.

Reads benchmark CSV with format:
  frame,track_id,head_x,head_y,bottom_y,row,seat,cost,status,prev_seat

Computes and reports:
  1. Row Stability (% frames track stays in same row)           [Target: >98%]
  2. Seat Stability (% frames track stays in same seat)         [Target: >95%]
  3. ID Switch Rate (seat track_id changes per 100 frames)       [Target: <2.0]
  4. Unassigned Rate (% of tracks marked UNASSIGNED)            [Target: <5.0%]
  5. Double Booking Rate (frames where two tracks share seat)   [Target: 0]
  6. Ghost Re-detection Rate (% ghost seats re-detected <= 8f)   [Informational]
  7. Average Cost (px distance between track and seat anchor)   [Informational]

Can run in two modes:
  1. Offline mode: analyzes existing CSV
     python benchmark_association.py --csv benchmark_association_log.csv
  2. Online mode: processes N frames of a video, outputs CSV, and audits metrics
     python benchmark_association.py --video "test video/test video - real classrom.mp4" --frames 300
"""
import os
import sys
import csv
import argparse
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND_PATH = os.path.join(HERE, "backend")
if BACKEND_PATH not in sys.path:
    sys.path.insert(0, BACKEND_PATH)


def compute_metrics(csv_path: str) -> dict:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Benchmark CSV not found: {csv_path}")

    rows_by_frame = defaultdict(list)
    tracks_by_id = defaultdict(list)
    seats_by_label = defaultdict(list)

    total_records = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            total_records += 1
            f_idx = int(r["frame"])
            tid = r["track_id"]
            seat = r["seat"]
            row = r["row"]
            status = r["status"]
            cost = float(r["cost"]) if r["cost"] != "" else 0.0

            entry = {
                "frame": f_idx,
                "track_id": tid,
                "seat": seat,
                "row": row,
                "status": status,
                "cost": cost,
                "prev_seat": r.get("prev_seat", seat),
            }
            rows_by_frame[f_idx].append(entry)
            if tid != "GHOST":
                tracks_by_id[tid].append(entry)
            if seat != "UNASSIGNED":
                seats_by_label[seat].append(entry)

    if not rows_by_frame:
        return {"error": "No records in CSV"}

    total_frames = max(rows_by_frame.keys()) - min(rows_by_frame.keys()) + 1

    # ── 1. Row Stability ──────────────────────────────────────────────────────
    # For active tracks seen in consecutive frames, did row change?
    row_transitions = 0
    row_maintained = 0
    for tid, obs_list in tracks_by_id.items():
        obs_list.sort(key=lambda x: x["frame"])
        for i in range(1, len(obs_list)):
            prev_obs = obs_list[i - 1]
            curr_obs = obs_list[i]
            # Only consider adjacent frames
            if curr_obs["frame"] == prev_obs["frame"] + 1:
                if prev_obs["status"] != "UNASSIGNED" and curr_obs["status"] != "UNASSIGNED":
                    row_transitions += 1
                    if prev_obs["row"] == curr_obs["row"]:
                        row_maintained += 1

    row_stability = (row_maintained / max(1, row_transitions)) * 100.0

    # ── 2. Seat Stability ─────────────────────────────────────────────────────
    # For active tracks in consecutive frames, did seat change?
    seat_transitions = 0
    seat_maintained = 0
    for tid, obs_list in tracks_by_id.items():
        for i in range(1, len(obs_list)):
            prev_obs = obs_list[i - 1]
            curr_obs = obs_list[i]
            if curr_obs["frame"] == prev_obs["frame"] + 1:
                if prev_obs["status"] != "UNASSIGNED" and curr_obs["status"] != "UNASSIGNED":
                    seat_transitions += 1
                    if prev_obs["seat"] == curr_obs["seat"]:
                        seat_maintained += 1

    seat_stability = (seat_maintained / max(1, seat_transitions)) * 100.0

    # ── 3. ID Switch Rate per 100 frames ──────────────────────────────────────
    # How many times does a physical seat's occupant track_id change?
    total_id_switches = 0
    for seat_lbl, obs_list in seats_by_label.items():
        obs_list.sort(key=lambda x: x["frame"])
        prev_tid = None
        for obs in obs_list:
            curr_tid = obs["track_id"]
            if curr_tid == "GHOST":
                continue
            if prev_tid is not None and curr_tid != prev_tid:
                total_id_switches += 1
            prev_tid = curr_tid

    id_switch_rate = (total_id_switches / max(1, total_frames)) * 100.0

    # ── 4. Unassigned Rate ────────────────────────────────────────────────────
    unassigned_count = sum(
        1 for records in rows_by_frame.values()
        for r in records if r["status"] == "UNASSIGNED"
    )
    total_active_dets = sum(
        1 for records in rows_by_frame.values()
        for r in records if r["status"] in ("OK", "UNASSIGNED")
    )
    unassigned_rate = (unassigned_count / max(1, total_active_dets)) * 100.0

    # ── 5. Double Booking Rate ────────────────────────────────────────────────
    double_booked_frames = 0
    for f_idx, records in rows_by_frame.items():
        claimed_seats = [
            r["seat"] for r in records
            if r["status"] == "OK" and r["seat"] != "UNASSIGNED"
        ]
        if len(claimed_seats) != len(set(claimed_seats)):
            double_booked_frames += 1

    # ── 6. Ghost Re-detection Rate ───────────────────────────────────────────
    # Ghost sequences: how many ghost sequences ended with a re-detection vs expiry?
    ghost_events = 0
    ghost_recovered = 0
    for seat_lbl, obs_list in seats_by_label.items():
        in_ghost = False
        ghost_start_tid = None
        for obs in obs_list:
            if obs["status"] == "GHOST":
                if not in_ghost:
                    in_ghost = True
                    ghost_events += 1
            else:
                if in_ghost:
                    ghost_recovered += 1
                    in_ghost = False

    ghost_recovery_rate = (ghost_recovered / max(1, ghost_events)) * 100.0 if ghost_events > 0 else 100.0

    # ── 7. Average Matching Cost ──────────────────────────────────────────────
    costs = [
        r["cost"] for records in rows_by_frame.values()
        for r in records if r["status"] == "OK"
    ]
    avg_cost = float(np.mean(costs)) if costs else 0.0

    return {
        "total_frames": total_frames,
        "total_records": total_records,
        "unique_tracks": len(tracks_by_id),
        "physical_seats": len(seats_by_label),
        "row_stability": row_stability,
        "seat_stability": seat_stability,
        "id_switch_rate": id_switch_rate,
        "unassigned_rate": unassigned_rate,
        "double_booked_frames": double_booked_frames,
        "ghost_recovery_rate": ghost_recovery_rate,
        "avg_cost_px": avg_cost,
    }


def print_report(metrics: dict):
    print("\n" + "=" * 68)
    print("      SMARTCLASS AI — V2.1 SEAT ASSOCIATION BENCHMARK REPORT      ")
    print("=" * 68)

    if "error" in metrics:
        print(f"Error: {metrics['error']}")
        return

    print(f" Analyzed Frames     : {metrics['total_frames']}")
    print(f" Total Observations  : {metrics['total_records']}")
    print(f" Unique Tracks Seen  : {metrics['unique_tracks']}")
    print(f" Physical Seats Mapped: {metrics['physical_seats']}")
    print("-" * 68)

    rows = [
        ("Row Stability", f"{metrics['row_stability']:.2f}%", "> 98.0%", metrics['row_stability'] >= 98.0),
        ("Seat Stability", f"{metrics['seat_stability']:.2f}%", "> 95.0%", metrics['seat_stability'] >= 95.0),
        ("ID Switch Rate / 100f", f"{metrics['id_switch_rate']:.2f}", "< 2.00", metrics['id_switch_rate'] <= 2.0),
        ("Unassigned Rate", f"{metrics['unassigned_rate']:.2f}%", "< 5.0%", metrics['unassigned_rate'] <= 5.0),
        ("Double-Booking Errors", f"{metrics['double_booked_frames']}", "0", metrics['double_booked_frames'] == 0),
        ("Ghost Recovery Rate", f"{metrics['ghost_recovery_rate']:.1f}%", "Informational", True),
        ("Avg Match Cost", f"{metrics['avg_cost_px']:.1f} px", "Informational", True),
    ]

    print(f" {'Metric':<24} | {'Measured':<14} | {'Target':<14} | {'Status':<6}")
    print("-" * 68)
    all_pass = True
    for name, val, target, passed in rows:
        if target != "Informational":
            status_str = "PASS" if passed else "FAIL"
            if not passed:
                all_pass = False
        else:
            status_str = "INFO"
        print(f" {name:<24} | {val:<14} | {target:<14} | {status_str:<6}")

    print("=" * 68)
    if all_pass:
        print("  VERDICT: ALL V2.1 BENCHMARK TARGETS MET (READY FOR V3)\n")
    else:
        print("  VERDICT: BENCHMARK COMPLETED — REVIEW FAILING METRICS\n")


def run_benchmark_on_video(video_path: str, max_frames: int = 300, output_csv: str = "benchmark_association_log.csv"):
    import cv2
    from core.engine import ClassroomEngine
    from core.config import YOLO_CONF

    print(f"Running V2.1 Association Benchmark on: {video_path}")
    print(f"Max frames to process: {max_frames}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    engine = ClassroomEngine(max_seats=20)

    # 1. Calibration on first 30 frames
    print("Collecting 30 calibration frames for topology discovery...", flush=True)
    calib_frames = []
    for _ in range(30):
        ret, frame = cap.read()
        if not ret: break
        calib_frames.append(frame)

    engine.calibrate(calib_frames, W, H)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if not engine.topology or not engine.topology.seats:
        print("ERROR: Topology discovery failed to locate seats!")
        return

    print(f"Discovered Topology: {engine.topology.rows} rows, {engine.topology.cols_per_row} seats/row.")

    # 2. Process frames
    print(f"Processing {max_frames} frames through Decoupled Association Pipeline...", flush=True)
    for f_idx in range(max_frames):
        ret, frame = cap.read()
        if not ret: break
        engine.process_single_frame(frame, W, H, persist=True)
        if (f_idx + 1) % 50 == 0 or (f_idx + 1) == max_frames:
            print(f"  Frame {f_idx + 1}/{max_frames} processed (records: {len(engine.benchmark_records)})", flush=True)

    cap.release()

    # 3. Export CSV and compute metrics
    engine.export_benchmark_csv(output_csv)
    print(f"Benchmark CSV exported to: {output_csv}")

    metrics = compute_metrics(output_csv)
    print_report(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartClass AI Association Benchmark")
    parser.add_argument("--video", type=str, default=None, help="Video file path to benchmark")
    parser.add_argument("--frames", type=int, default=300, help="Number of frames to process")
    parser.add_argument("--csv", type=str, default="benchmark_association_log.csv", help="CSV path to save or evaluate")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate existing CSV without running video")
    args = parser.parse_args()

    default_video = os.path.join(HERE, "test video", "test video - real classrom.mp4")
    target_video = args.video or (default_video if os.path.isfile(default_video) else None)

    if args.eval_only and os.path.isfile(args.csv):
        metrics = compute_metrics(args.csv)
        print_report(metrics)
    elif target_video:
        run_benchmark_on_video(target_video, max_frames=args.frames, output_csv=args.csv)
    elif os.path.isfile(args.csv):
        metrics = compute_metrics(args.csv)
        print_report(metrics)
    else:
        print(f"No CSV or video found. Specify --video <path> or --csv <path>.")
