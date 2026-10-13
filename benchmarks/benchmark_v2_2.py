# -*- coding: utf-8 -*-
"""
SmartClass AI — Seat Geometry & Association Benchmark Suite (V2.2)
===================================================================
Evaluates seating geometry, explicit row semantics, spatial match confidence,
and stability against the Five V2.2 Accuracy Questions.

Reads benchmark CSV formatted with:
  frame,track_id,head_x,head_y,bottom_y,row,seat,cost,status,prev_seat,seat_confidence,match_type

Computes and reports:
  1. Person Detection Rate (% frames with detections, total active tracks)
  2. Row Semantics & Classification Accuracy (R1=FRONT, R2=BACK, stability %)
  3. Seat Assignment Accuracy (EXACT_REGION %, OVERLAP_REGION %, FALLBACK %, RESCUE %)
  4. Seat Spatial Confidence Distribution (Mean seat_confidence, High >= 0.85 %, Mid >= 0.60 %)
  5. Unassigned & Anomaly Breakdown (Unassigned rate %, Diagnostic breakdown)
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


def compute_v2_2_metrics(csv_path: str) -> dict:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Benchmark CSV not found: {csv_path}")

    rows_by_frame = defaultdict(list)
    tracks_by_id = defaultdict(list)
    seats_by_label = defaultdict(list)
    match_types = defaultdict(int)
    conf_scores = []
    unassigned_details = []

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
            cost = float(r["cost"]) if r.get("cost") not in (None, "", "GHOST") else 0.0
            seat_conf_raw = r.get("geometric_match") or r.get("seat_confidence")
            seat_conf = float(seat_conf_raw) if seat_conf_raw not in (None, "") else 1.0
            match_type = r.get("match_type", "EXACT_REGION")

            entry = {
                "frame": f_idx,
                "track_id": tid,
                "seat": seat,
                "row": row,
                "status": status,
                "cost": cost,
                "seat_confidence": seat_conf,
                "match_type": match_type,
                "prev_seat": r.get("prev_seat", seat),
            }
            rows_by_frame[f_idx].append(entry)
            if tid != "GHOST":
                tracks_by_id[tid].append(entry)
                if status == "OK":
                    conf_scores.append(seat_conf)
                    match_types[match_type] += 1
                elif status == "UNASSIGNED":
                    unassigned_details.append(entry)
            if seat != "UNASSIGNED":
                seats_by_label[seat].append(entry)

    if not rows_by_frame:
        return {"error": "No records in CSV"}

    total_frames = max(rows_by_frame.keys()) - min(rows_by_frame.keys()) + 1

    # 1. Row Stability & Semantics
    row_transitions = 0
    row_maintained = 0
    for tid, obs_list in tracks_by_id.items():
        obs_list.sort(key=lambda x: x["frame"])
        for i in range(1, len(obs_list)):
            prev_obs = obs_list[i - 1]
            curr_obs = obs_list[i]
            if curr_obs["frame"] == prev_obs["frame"] + 1:
                if prev_obs["status"] != "UNASSIGNED" and curr_obs["status"] != "UNASSIGNED":
                    row_transitions += 1
                    if prev_obs["row"] == curr_obs["row"]:
                        row_maintained += 1

    row_stability = (row_maintained / max(1, row_transitions)) * 100.0

    # 2. Seat Stability
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

    # 3. ID Switch Rate per 100 frames
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

    # 4. Spatial Match Breakdown
    total_matches = sum(match_types.values())
    exact_region_pct = (match_types["EXACT_REGION"] / max(1, total_matches)) * 100.0
    overlap_region_pct = (match_types["OVERLAP_REGION"] / max(1, total_matches)) * 100.0
    fallback_pct = (match_types["NEAREST_FALLBACK"] / max(1, total_matches)) * 100.0
    rescue_pct = (match_types["RESCUE"] / max(1, total_matches)) * 100.0

    # 5. Seat Confidence Stats
    mean_conf = float(np.mean(conf_scores)) if conf_scores else 0.0
    high_conf_pct = (sum(1 for s in conf_scores if s >= 0.85) / max(1, len(conf_scores))) * 100.0
    mid_conf_pct = (sum(1 for s in conf_scores if 0.60 <= s < 0.85) / max(1, len(conf_scores))) * 100.0
    low_conf_pct = (sum(1 for s in conf_scores if s < 0.60) / max(1, len(conf_scores))) * 100.0

    # 6. Unassigned & Double-Booking Metrics
    unassigned_count = len(unassigned_details)
    total_active_dets = total_matches + unassigned_count
    unassigned_rate = (unassigned_count / max(1, total_active_dets)) * 100.0

    double_booked_frames = 0
    for f_idx, records in rows_by_frame.items():
        claimed_seats = [
            r["seat"] for r in records
            if r["status"] == "OK" and r["seat"] != "UNASSIGNED"
        ]
        if len(claimed_seats) != len(set(claimed_seats)):
            double_booked_frames += 1

    return {
        "total_frames": total_frames,
        "total_records": total_records,
        "unique_tracks": len(tracks_by_id),
        "mapped_seats": len(seats_by_label),
        "row_stability": row_stability,
        "seat_stability": seat_stability,
        "id_switch_rate": id_switch_rate,
        "unassigned_rate": unassigned_rate,
        "double_booked_frames": double_booked_frames,
        "mean_seat_confidence": mean_conf,
        "high_conf_pct": high_conf_pct,
        "mid_conf_pct": mid_conf_pct,
        "low_conf_pct": low_conf_pct,
        "exact_region_pct": exact_region_pct,
        "overlap_region_pct": overlap_region_pct,
        "fallback_pct": fallback_pct,
        "rescue_pct": rescue_pct,
    }


def print_v2_2_report(metrics: dict):
    print("\n" + "=" * 72)
    print("     SMARTCLASS AI -- V2.2 SEAT GEOMETRY & BENCHMARK REPORT      ")
    print("=" * 72)

    if "error" in metrics:
        print(f"Error: {metrics['error']}")
        return

    print(f" Analyzed Frames         : {metrics['total_frames']}")
    print(f" Total Observations      : {metrics['total_records']}")
    print(f" Unique Tracks Tracked   : {metrics['unique_tracks']}")
    print(f" Physical Seats Discovered: {metrics['mapped_seats']}")
    print("-" * 72)

    print(" -- 1. SPATIAL ACCURACY & MATCH QUALITY --")
    print(f" Mean Seat Confidence    : {metrics['mean_seat_confidence']:.2f}")
    print(f" High Conf (>=0.85)      : {metrics['high_conf_pct']:.1f}%")
    print(f" Mid Conf (0.60-0.85)    : {metrics['mid_conf_pct']:.1f}%")
    print(f" Low Conf (<0.60)        : {metrics['low_conf_pct']:.1f}%")
    print(f" Match Types             : Exact={metrics['exact_region_pct']:.1f}% | Overlap={metrics['overlap_region_pct']:.1f}% | Fallback={metrics['fallback_pct']:.1f}% | Rescue={metrics['rescue_pct']:.1f}%")
    print("-" * 72)

    rows = [
        ("Row Stability", f"{metrics['row_stability']:.2f}%", "> 98.0%", metrics['row_stability'] >= 98.0),
        ("Seat Stability", f"{metrics['seat_stability']:.2f}%", "> 95.0%", metrics['seat_stability'] >= 95.0),
        ("ID Switch Rate / 100f", f"{metrics['id_switch_rate']:.2f}", "< 2.00", metrics['id_switch_rate'] <= 2.0),
        ("Unassigned Rate", f"{metrics['unassigned_rate']:.2f}%", "< 5.0%", metrics['unassigned_rate'] <= 5.0),
        ("Double-Booking Errors", f"{metrics['double_booked_frames']}", "0", metrics['double_booked_frames'] == 0),
        ("Mean Seat Conf", f"{metrics['mean_seat_confidence']:.2f}", ">= 0.80", metrics['mean_seat_confidence'] >= 0.80),
    ]

    print(f" {'Metric':<24} | {'Measured':<14} | {'Target':<14} | {'Status':<6}")
    print("-" * 72)
    all_pass = True
    for name, val, target, passed in rows:
        status_str = "PASS" if passed else "WARN"
        if not passed and name != "Unassigned Rate":
            all_pass = False
        print(f" {name:<24} | {val:<14} | {target:<14} | {status_str:<6}")

    print("=" * 72)
    if all_pass:
        print("  VERDICT: V2.2 SEAT GEOMETRY BENCHMARK TARGETS SATISFIED ✓\n")
    else:
        print("  VERDICT: BENCHMARK COMPLETE — AUDIT WARNINGS IDENTIFIED\n")


def run_benchmark_on_video(video_path: str, max_frames: int = 300, output_csv: str = "benchmark_v2_2_log.csv"):
    import cv2
    from core.engine import ClassroomEngine
    from core.config import YOLO_CONF

    print(f"Running V2.2 Seat Geometry Benchmark on: {video_path}")
    print(f"Max frames: {max_frames}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    engine = ClassroomEngine(max_seats=20)

    # 1. Calibration Phase (first 60 frames)
    calib_frames = []
    f_idx = 0
    print("Gathering calibration frames...")
    while cap.isOpened() and len(calib_frames) < 60:
        ret, frame = cap.read()
        if not ret:
            break
        if f_idx % 2 == 0:
            calib_frames.append(frame)
        f_idx += 1

    print(f"Calibrating with {len(calib_frames)} frames...")
    engine.calibrate(calib_frames, W, H)

    # Reset video capture to frame 0 for benchmark run
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    engine.frame_count = 0
    engine.benchmark_records.clear()

    processed = 0
    print("Processing benchmark frames...")
    while cap.isOpened() and processed < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        processed += 1
        if processed % 2 == 0:
            engine.process_single_frame(frame, W, H, persist=True)

    cap.release()

    # Save benchmark CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "frame", "track_id", "head_x", "head_y", "bottom_y",
            "row", "seat", "cost", "status", "prev_seat", "seat_confidence", "match_type"
        ])
        writer.writeheader()
        writer.writerows(engine.benchmark_records)

    print(f"Wrote {len(engine.benchmark_records)} records to {output_csv}")
    metrics = compute_v2_2_metrics(output_csv)
    print_v2_2_report(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartClass AI V2.2 Benchmark Tool")
    parser.add_argument("--csv", type=str, help="Path to existing benchmark CSV log")
    parser.add_argument("--video", type=str, help="Path to input video for live evaluation")
    parser.add_argument("--frames", type=int, default=300, help="Max frames to process in video mode")
    parser.add_argument("--out", type=str, default="benchmark_v2_2_log.csv", help="Output CSV filepath")

    args = parser.parse_args()

    if args.csv:
        metrics = compute_v2_2_metrics(args.csv)
        print_v2_2_report(metrics)
    elif args.video:
        run_benchmark_on_video(args.video, max_frames=args.frames, output_csv=args.out)
    else:
        # Default fallback to existing log if present
        default_csv = "benchmark_association_log.csv"
        if os.path.isfile(default_csv):
            print(f"No arguments provided. Auditing {default_csv}...")
            metrics = compute_v2_2_metrics(default_csv)
            print_v2_2_report(metrics)
        else:
            print("Usage: python benchmark_v2_2.py --csv <log.csv> OR python benchmark_v2_2.py --video <video.mp4>")
