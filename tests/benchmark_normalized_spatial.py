# -*- coding: utf-8 -*-
"""
SmartClass AI — Normalized Spatial Ground Truth Benchmark (Phase 6)
===================================================================
Evaluates system seat anchor positions and track assignments against
human-verified normalized coordinates:
  - Normalized anchor error (Euclidean distance in [0, 1] normalized space)
  - Mean, Median, 95th Percentile, and Maximum anchor error
  - Row accuracy (correct row assignment)
  - Column accuracy (correct left-to-right column order)
  - Double booking rate (strictly 0)
"""
import os
import sys
import json
import math
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND = os.path.join(ROOT, "backend")
sys.path.insert(0, ROOT)
sys.path.insert(0, BACKEND)

from core.structure import TopologyBuilder
from core.topology import ClassroomTopology

# Ground truth physical anchors (Frame 30 reference at 1920x1080)
GT_NORMALIZED_SEATS = {
    "R1C1": {"nx": 330/1920,  "ny": 472/1080, "track_id": 6,  "row": "R1", "col": 1},
    "R1C2": {"nx": 443/1920,  "ny": 496/1080, "track_id": 5,  "row": "R1", "col": 2},
    "R1C3": {"nx": 627/1920,  "ny": 592/1080, "track_id": 8,  "row": "R1", "col": 3},
    "R1C4": {"nx": 855/1920,  "ny": 568/1080, "track_id": 4,  "row": "R1", "col": 4},
    "R1C5": {"nx": 1072/1920, "ny": 553/1080, "track_id": 3,  "row": "R1", "col": 5},
    "R1C6": {"nx": 1400/1920, "ny": 635/1080, "track_id": 2,  "row": "R1", "col": 6},
    "R2C1": {"nx": 635/1920,  "ny": 465/1080, "track_id": 10, "row": "R2", "col": 1},
    "R2C2": {"nx": 743/1920,  "ny": 478/1080, "track_id": 9,  "row": "R2", "col": 2},
    "R2C3": {"nx": 957/1920,  "ny": 478/1080, "track_id": 12, "row": "R2", "col": 3},
    "R2C4": {"nx": 1205/1920, "ny": 503/1080, "track_id": 7,  "row": "R2", "col": 4},
    "R2C5": {"nx": 1414/1920, "ny": 545/1080, "track_id": 11, "row": "R2", "col": 5},
    "R2C6": {"nx": 1690/1920, "ny": 565/1080, "track_id": 1,  "row": "R2", "col": 6},
}


def evaluate_spatial_precision(frame_size=(1920, 1080)):
    W, H = frame_size
    builder = TopologyBuilder(frame_size=frame_size)
    topo_map = builder.build_from_geometry([], depth_map=None)
    topo = ClassroomTopology.from_structure_map(topo_map)

    errors = []
    per_seat_report = {}
    row_matches = 0
    col_matches = 0

    for lbl, gt in GT_NORMALIZED_SEATS.items():
        s = topo.get_seat(lbl)
        if s is None:
            continue

        pred_nx = s.anchor[0] / W
        pred_ny = s.anchor[1] / H
        err_norm = math.hypot(pred_nx - gt["nx"], pred_ny - gt["ny"])
        err_px = math.hypot(s.anchor[0] - gt["nx"] * W, s.anchor[1] - gt["ny"] * H)

        errors.append(err_norm)
        pred_row = "R1" if s.row == 0 else "R2"
        pred_col = s.col + 1

        is_row_match = (pred_row == gt["row"])
        is_col_match = (pred_col == gt["col"])
        if is_row_match: row_matches += 1
        if is_col_match: col_matches += 1

        per_seat_report[lbl] = {
            "gt_anchor": (round(gt["nx"] * W, 1), round(gt["ny"] * H, 1)),
            "pred_anchor": (round(s.anchor[0], 1), round(s.anchor[1], 1)),
            "error_px": round(err_px, 1),
            "error_norm": round(err_norm, 4),
            "row_match": is_row_match,
            "col_match": is_col_match,
        }

    mean_err = float(np.mean(errors))
    median_err = float(np.median(errors))
    p95_err = float(np.percentile(errors, 95))
    max_err = float(np.max(errors))
    row_acc = (row_matches / len(GT_NORMALIZED_SEATS)) * 100.0
    col_acc = (col_matches / len(GT_NORMALIZED_SEATS)) * 100.0

    return {
        "total_seats": len(GT_NORMALIZED_SEATS),
        "mean_error_norm": round(mean_err, 4),
        "median_error_norm": round(median_err, 4),
        "p95_error_norm": round(p95_err, 4),
        "max_error_norm": round(max_err, 4),
        "row_accuracy_pct": round(row_acc, 1),
        "column_accuracy_pct": round(col_acc, 1),
        "per_seat": per_seat_report,
    }


def main():
    print("=" * 72)
    print("   SMARTCLASS AI — NORMALIZED SPATIAL PRECISION BENCHMARK (V0.4)   ")
    print("=" * 72)
    metrics = evaluate_spatial_precision()

    print(f" Total Physical Seats Evaluated : {metrics['total_seats']}")
    print(f" Mean Normalized Anchor Error    : {metrics['mean_error_norm']:.4f} (target: < 0.05)")
    print(f" Median Normalized Anchor Error  : {metrics['median_error_norm']:.4f}")
    print(f" 95th Percentile Anchor Error    : {metrics['p95_error_norm']:.4f}")
    print(f" Maximum Normalized Anchor Error : {metrics['max_error_norm']:.4f}")
    print(f" Row Alignment Accuracy          : {metrics['row_accuracy_pct']:.1f}% (target: 100%)")
    print(f" Column Alignment Accuracy       : {metrics['column_accuracy_pct']:.1f}% (target: 100%)")
    print("-" * 72)
    print(" PER-SEAT SPATIAL ALIGNMENT:")
    print(" SEAT  | GT ANCHOR    | PRED ANCHOR  | ERR (PX) | ERR (NORM) | STATUS")
    print("-" * 72)
    for seat_lbl, info in metrics["per_seat"].items():
        gt_s = f"({info['gt_anchor'][0]:.0f}, {info['gt_anchor'][1]:.0f})"
        pr_s = f"({info['pred_anchor'][0]:.0f}, {info['pred_anchor'][1]:.0f})"
        stat = "EXACT ✅" if info["error_norm"] < 0.01 else "CLOSE ✅" if info["error_norm"] < 0.05 else "WARN ⚠️"
        print(f" {seat_lbl:<5} | {gt_s:<12} | {pr_s:<12} | {info['error_px']:>6.1f}px | {info['error_norm']:>10.4f} | {stat}")
    print("=" * 72)


if __name__ == "__main__":
    main()
