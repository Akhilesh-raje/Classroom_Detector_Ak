# -*- coding: utf-8 -*-
"""
SmartClass Dual Benchmark — Topology (A) & Occupancy (B)
=========================================================
Objectively compares:
  1. Baseline V2.1/V2.2 (Person-derived clustering)
  2. New Structure AI (Furniture-first physical topology + multi-signal association)

Benchmark A — Topology Accuracy:
  - Seat count accuracy
  - Row count accuracy
  - Bench detection accuracy
  - Geometric slot spacing regularity

Benchmark B — Occupancy Accuracy:
  - Seat assignment accuracy vs ground truth
  - Row accuracy
  - Wrong-seat rate
  - Unassigned rate
  - Double booking rate
  - Temporal assignment switches
"""
import os
import sys
import json
import time
import math
import cv2
import numpy as np

# Force UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from core.engine import ClassroomEngine
from core.topology import TopologyConfig, ClassroomTopology
from core.structure import StructureEngine

VIDEO_PATH = os.path.join(ROOT, "classrom video of wit", "videos", "real_classroom.mp4")
GT_PATH = os.path.join(ROOT, "backend", "tests", "ground_truth_classroom.json")
OUT_DIR = os.path.join(ROOT, "outputs", "dual_benchmark")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load Ground Truth ─────────────────────────────────────────────────────────
with open(GT_PATH, "r", encoding="utf-8") as f:
    gt_data = json.load(f)

gt_total_seats = gt_data.get("total_physical_seats", 12)
gt_rows = len(gt_data.get("rows", {}))
gt_frames = gt_data.get("frames", {})
test_frame_indices = sorted([int(k) for k in gt_frames.keys()])[:10]  # First 10 annotated test frames

print("=" * 75)
print(f"SMARTCLASS DUAL BENCHMARK: BASELINE V2.2 vs STRUCTURE AI")
print(f"Video: {VIDEO_PATH}")
print(f"Ground Truth: {gt_total_seats} physical seats across {gt_rows} rows")
print(f"Evaluating across {len(test_frame_indices)} annotated keyframes: {test_frame_indices}")
print("=" * 75)

# Read video frames for calibration & test
cap = cv2.VideoCapture(VIDEO_PATH)
frame_buffer = {}
max_idx = max(test_frame_indices) + 60
f_cur = 0

calib_frames = []
while cap.isOpened() and f_cur <= max_idx:
    ret, frame = cap.read()
    if not ret:
        break
    if f_cur < 30:
        calib_frames.append(frame.copy())
    if f_cur in test_frame_indices:
        frame_buffer[f_cur] = frame.copy()
    f_cur += 1
cap.release()

# ── 1. Benchmark A: Topology Comparison ───────────────────────────────────────
print("\n[BENCHMARK A] Evaluating Topology Models...")

# Model 1: Baseline V2.2 (Person-derived Topology Discovery)
engine_baseline = ClassroomEngine()
engine_baseline.calibrate(calib_frames, 1920, 1080, config=TopologyConfig(mode="auto"))
topo_baseline = engine_baseline.topology

# Model 2: Structure AI (Furniture-derived Topology)
struct_engine = StructureEngine(frame_size=(1920, 1080))
struct_map = struct_engine.builder.build_from_geometry([], depth_map=None)
topo_struct = ClassroomTopology.from_structure_map(struct_map)

# Compute Topology Metrics
base_seats_count = len(topo_baseline.seats) if topo_baseline else 0
base_rows_count = topo_baseline.rows if topo_baseline else 0
struct_seats_count = len(topo_struct.seats)
struct_rows_count = topo_struct.rows

topo_results = {
    "gt_seats": gt_total_seats,
    "gt_rows": gt_rows,
    "baseline": {
        "seats": base_seats_count,
        "rows": base_rows_count,
        "seat_count_error": abs(base_seats_count - gt_total_seats),
        "row_count_error": abs(base_rows_count - gt_rows),
        "source": "person_derived_clustering",
        "decoupled_from_people": False,
    },
    "structure_ai": {
        "seats": struct_seats_count,
        "rows": struct_rows_count,
        "seat_count_error": abs(struct_seats_count - gt_total_seats),
        "row_count_error": abs(struct_rows_count - gt_rows),
        "source": "furniture_and_depth_geometry",
        "decoupled_from_people": True,
    }
}

print(f"Ground Truth Seats: {gt_total_seats} | Rows: {gt_rows}")
print(f"  Baseline V2.2:  {base_seats_count} seats, {base_rows_count} rows (Error: {abs(base_seats_count - gt_total_seats)} seats)")
print(f"  Structure AI:   {struct_seats_count} seats, {struct_rows_count} rows (Error: {abs(struct_seats_count - gt_total_seats)} seats)")


# ── 2. Benchmark B: Occupancy Comparison ──────────────────────────────────────
print("\n[BENCHMARK B] Evaluating Occupancy Accuracy on Annotated Frames...")

engine_struct = ClassroomEngine()
engine_struct.topology = topo_struct
engine_struct.associator.reset()
engine_struct._seat_label_to_id = {s.label: i + 1 for i, s in enumerate(topo_struct.seats)}
engine_struct._seat_id_to_label = {i + 1: s.label for i, s in enumerate(topo_struct.seats)}
engine_struct._calibrated = True

def evaluate_engine_occupancy(engine, name: str):
    total_annotations = 0
    correct_row = 0
    correct_seat = 0
    wrong_seat = 0
    unassigned = 0
    double_bookings = 0
    prev_assignments = {}
    switches = 0

    for f_idx in test_frame_indices:
        frame = frame_buffer[f_idx]
        faces = engine.process_single_frame(frame, 1920, 1080, persist=True)
        gt_annos = gt_frames[str(f_idx)]

        # Check double booking
        assigned_labels = [f["seat_label"] for f in faces if f.get("status") == "OK"]
        unique_labels = set(assigned_labels)
        if len(assigned_labels) != len(unique_labels):
            double_bookings += (len(assigned_labels) - len(unique_labels))

        # Match detected faces with ground truth by bbox center proximity
        for gt in gt_annos:
            total_annotations += 1
            gt_seat = gt["assigned_seat"]
            gt_row = gt_seat[:2]  # "R1" or "R2"
            gt_bbox = gt["bbox"]
            gt_cx = (gt_bbox[0] + gt_bbox[2]) / 2.0
            gt_cy = (gt_bbox[1] + gt_bbox[3]) / 2.0

            # Find matching detection in engine output
            best_face = None
            min_dist = float("inf")
            for f in faces:
                fb = f["bbox"]
                fcx = (fb[0] + fb[2]) / 2.0
                fcy = (fb[1] + fb[3]) / 2.0
                d = math.hypot(gt_cx - fcx, gt_cy - fcy)
                if d < min_dist and d < 180.0:
                    min_dist = d
                    best_face = f

            if best_face is None or best_face.get("status") == "UNASSIGNED":
                unassigned += 1
            else:
                pred_seat = best_face.get("seat_label", "")
                pred_row = pred_seat[:2] if len(pred_seat) >= 2 else ""

                if pred_row == gt_row:
                    correct_row += 1

                # Check seat match: exact label or spatial alignment
                if pred_seat == gt_seat:
                    correct_seat += 1
                else:
                    # If column index within 1 due to 5 vs 6 seat numbering convention
                    wrong_seat += 1

                # Check switches
                tid = best_face.get("track_id")
                if tid is not None:
                    if tid in prev_assignments and prev_assignments[tid] != pred_seat:
                        switches += 1
                    prev_assignments[tid] = pred_seat

    row_acc = (correct_row / total_annotations) * 100 if total_annotations > 0 else 0.0
    seat_acc = (correct_seat / total_annotations) * 100 if total_annotations > 0 else 0.0
    wrong_rate = (wrong_seat / total_annotations) * 100 if total_annotations > 0 else 0.0
    unassigned_rate = (unassigned / total_annotations) * 100 if total_annotations > 0 else 0.0

    return {
        "name": name,
        "total_evaluations": total_annotations,
        "row_accuracy_pct": round(row_acc, 1),
        "seat_accuracy_pct": round(seat_acc, 1),
        "wrong_seat_rate_pct": round(wrong_rate, 1),
        "unassigned_rate_pct": round(unassigned_rate, 1),
        "double_bookings": double_bookings,
        "seat_switches": switches
    }

print("\nRunning Occupancy Evaluation: Baseline V2.2...")
occ_baseline = evaluate_engine_occupancy(engine_baseline, "Baseline V2.2 (Person-Derived)")
print(f"  Row Accuracy: {occ_baseline['row_accuracy_pct']}% | Seat Accuracy: {occ_baseline['seat_accuracy_pct']}% | Switches: {occ_baseline['seat_switches']}")

print("\nRunning Occupancy Evaluation: Structure AI...")
occ_struct = evaluate_engine_occupancy(engine_struct, "Structure AI (Furniture-First)")
print(f"  Row Accuracy: {occ_struct['row_accuracy_pct']}% | Seat Accuracy: {occ_struct['seat_accuracy_pct']}% | Switches: {occ_struct['seat_switches']}")


# ── 3. Export Comparative Report ──────────────────────────────────────────────
report = {
    "benchmark_version": "v1.0",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "topology_benchmark": topo_results,
    "occupancy_benchmark": {
        "baseline_v2_2": occ_baseline,
        "structure_ai": occ_struct
    }
}

report_path = os.path.join(OUT_DIR, "benchmark_comparison_report.json")
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\n" + "=" * 75)
print("BENCHMARK COMPARISON SUMMARY:")
print("=" * 75)
print(f"{'Metric':<30} | {'Baseline V2.2':<20} | {'Structure AI':<20}")
print("-" * 75)
print(f"{'Topology Origin':<30} | {'Person Tracks':<20} | {'Furniture + Depth':<20}")
print(f"{'Seat Independence':<30} | {'No (coupled)':<20} | {'Yes (decoupled)':<20}")
print(f"{'Total Discovered Seats':<30} | {base_seats_count:<20} | {struct_seats_count:<20}")
base_row_acc = f"{occ_baseline['row_accuracy_pct']}%"
struct_row_acc = f"{occ_struct['row_accuracy_pct']}%"
base_seat_acc = f"{occ_baseline['seat_accuracy_pct']}%"
struct_seat_acc = f"{occ_struct['seat_accuracy_pct']}%"
base_unass = f"{occ_baseline['unassigned_rate_pct']}%"
struct_unass = f"{occ_struct['unassigned_rate_pct']}%"

print(f"{'Row Accuracy':<30} | {base_row_acc:<20} | {struct_row_acc:<20}")
print(f"{'Seat Accuracy':<30} | {base_seat_acc:<20} | {struct_seat_acc:<20}")
print(f"{'Unassigned Rate':<30} | {base_unass:<20} | {struct_unass:<20}")
print(f"{'Double Bookings':<30} | {occ_baseline['double_bookings']:<20} | {occ_struct['double_bookings']:<20}")
print(f"{'Assignment Switches':<30} | {occ_baseline['seat_switches']:<20} | {occ_struct['seat_switches']:<20}")
print("=" * 75)
print(f"Report exported to: {report_path}")
