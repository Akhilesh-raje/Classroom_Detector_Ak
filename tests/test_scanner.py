"""
SmartClass AI — Classroom Scanner Tests
Tests all scanner logic without needing a camera.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scan_classroom import (
    cluster_into_rows, generate_seat_names,
    deduplicate_detections, iou, build_layout_json,
    build_grid_image, draw_seat, draw_row_lines
)
import numpy as np
import json

print("=== Testing scan_classroom.py logic ===")

# ── Mock 6 seats in 2 rows ────────────────────────────────────────────────────
mock_dets = [
    {"bbox":[50,100,200,250],  "cls":56,"cls_name":"chair","confidence":0.85,"center":[125.0,175.0],"area":22500},
    {"bbox":[250,100,400,250], "cls":56,"cls_name":"chair","confidence":0.82,"center":[325.0,175.0],"area":22500},
    {"bbox":[450,100,600,250], "cls":56,"cls_name":"chair","confidence":0.79,"center":[525.0,175.0],"area":22500},
    {"bbox":[50,350,200,500],  "cls":56,"cls_name":"chair","confidence":0.88,"center":[125.0,425.0],"area":22500},
    {"bbox":[250,350,400,500], "cls":56,"cls_name":"chair","confidence":0.84,"center":[325.0,425.0],"area":22500},
    {"bbox":[450,350,600,500], "cls":56,"cls_name":"chair","confidence":0.81,"center":[525.0,425.0],"area":22500},
]

# ── Test 1: Row clustering ────────────────────────────────────────────────────
rows = cluster_into_rows(mock_dets, frame_h=600)
assert len(rows) == 2,    f"Expected 2 rows, got {len(rows)}"
assert len(rows[0]) == 3, f"Expected 3 seats in row 1, got {len(rows[0])}"
assert len(rows[1]) == 3, f"Expected 3 seats in row 2, got {len(rows[1])}"
print(f"  [OK] cluster_into_rows: {len(rows)} rows, {[len(r) for r in rows]} seats per row")

# ── Test 2: Left-to-right ordering ───────────────────────────────────────────
for r_idx, row in enumerate(rows):
    xs = [d["center"][0] for d in row]
    assert xs == sorted(xs), f"Row {r_idx+1} not sorted left-to-right: {xs}"
print("  [OK] rows sorted left-to-right")

# ── Test 3: Seat naming — desk layout (1 seat per bench = R1C1 format) ────────
# Use seats_per_bench=1 to get the desk/column format (backward-compatible).
# This also tests that the function accepts the seats_per_bench parameter.
rows_copy = cluster_into_rows(mock_dets, frame_h=600)  # fresh copy, generate_seat_names mutates dets
generate_seat_names(rows_copy, seats_per_bench=1, ollama_available=False)
labels_desk = [d["seat_label"] for row in rows_copy for d in row]
assert labels_desk == ["R1C1","R1C2","R1C3","R2C1","R2C2","R2C3"], \
    f"Desk labels wrong: {labels_desk}"
print(f"  [OK] seat labels (desk/1-per-bench): {labels_desk}")

# Test bench layout (default, 2 seats per bench = R1B1-L/R format)
rows_bench = cluster_into_rows(mock_dets, frame_h=600)
generate_seat_names(rows_bench, seats_per_bench=2, ollama_available=False)
# Each physical bench detection expands to 2 sub-seats; 3 detections × 2 = 6 sub-seats per row
all_bench_labels = [lbl for row in rows_bench for d in row for lbl in d.get("bench_labels", [])]
assert len(all_bench_labels) == 12, f"Expected 12 bench sub-labels, got {len(all_bench_labels)}"
assert all_bench_labels[0] == "R1B1-L", f"First label should be R1B1-L, got {all_bench_labels[0]}"
assert all_bench_labels[1] == "R1B1-R", f"Second label should be R1B1-R, got {all_bench_labels[1]}"
print(f"  [OK] seat labels (bench/2-per-bench): first 4={all_bench_labels[:4]}")

# Test 3-seat bench (L/M/R)
rows_3 = cluster_into_rows(mock_dets[:3], frame_h=600)
generate_seat_names(rows_3, seats_per_bench=3, ollama_available=False)
labels_3 = rows_3[0][0].get("bench_labels", [])
assert labels_3 == ["R1B1-L","R1B1-M","R1B1-R"], f"3-seat bench labels wrong: {labels_3}"
print(f"  [OK] seat labels (bench/3-per-bench): {labels_3}")

# ── Test 4: IoU ───────────────────────────────────────────────────────────────
b1 = [0,0,100,100]
b2 = [50,50,150,150]
b3 = [200,200,300,300]
assert 0.1 < iou(b1,b2) < 0.5, f"IoU b1/b2 wrong: {iou(b1,b2)}"
assert iou(b1,b3) == 0.0,       f"IoU b1/b3 should be 0"
print(f"  [OK] IoU: overlapping={iou(b1,b2):.3f}, non-overlapping={iou(b1,b3):.3f}")

# IoU edge cases
assert iou([0,0,0,0],[0,0,100,100]) == 0.0, "zero-area box should give 0"
assert abs(iou([0,0,100,100],[0,0,100,100]) - 1.0) < 1e-6, "identical boxes should give 1.0"
print("  [OK] IoU edge cases (zero-area, identical)")

# ── Test 5: Deduplication ─────────────────────────────────────────────────────
dup_dets = mock_dets + [
    {"bbox":[55,105,205,255],"cls":56,"cls_name":"chair","confidence":0.60,"center":[130.0,180.0],"area":22500},
]
deduped = deduplicate_detections(dup_dets, iou_thresh=0.4)
assert len(deduped) == 6, f"Expected 6 after dedup, got {len(deduped)}"
print(f"  [OK] deduplicate: {len(dup_dets)} → {len(deduped)} (removed 1 duplicate)")

# ── Test 6: Layout JSON ───────────────────────────────────────────────────────
# Use desk layout (seats_per_bench=1) — 3 detections = 3 seat records per row = 6 total
rows_json = cluster_into_rows(mock_dets, frame_h=600)
generate_seat_names(rows_json, seats_per_bench=1)
layout = build_layout_json(rows_json, 640, 480, cam_idx=1,
                           timestamp="2026-09-21 12:00:00", seats_per_bench=1)
assert layout["total_seats"] == 6,    f"Expected 6, got {layout['total_seats']}"
assert layout["total_rows"]  == 2,    f"Expected 2, got {layout['total_rows']}"
assert layout["max_cols"]    == 3,    f"Expected 3, got {layout['max_cols']}"
assert layout["version"]     == "3.0",f"Expected version 3.0, got {layout['version']}"
assert layout["layout_type"] == "desk"
assert layout["seats"][0]["seat_label"] == "R1C1", \
    f"First seat should be R1C1, got {layout['seats'][0]['seat_label']}"
assert layout["seats"][5]["seat_label"] == "R2C3", \
    f"Last seat should be R2C3, got {layout['seats'][5]['seat_label']}"
# JSON round-trip
l2 = json.loads(json.dumps(layout))
assert l2["total_seats"] == 6
print(f"  [OK] build_layout_json (desk): {layout['total_seats']} seats, JSON round-trip OK")

# Bench layout — 3 detections × 2 sub-seats = 6 records per row = 12 total
rows_bjson = cluster_into_rows(mock_dets, frame_h=600)
generate_seat_names(rows_bjson, seats_per_bench=2)
layout_b = build_layout_json(rows_bjson, 640, 480, cam_idx=1,
                             timestamp="2026-09-21 12:00:00", seats_per_bench=2)
assert layout_b["total_seats"] == 12,   f"Expected 12 bench seats, got {layout_b['total_seats']}"
assert layout_b["layout_type"] == "bench"
assert "R1B1-L" in layout_b["grid_cells"]
assert "R2B3-R" in layout_b["grid_cells"]
print(f"  [OK] build_layout_json (bench): {layout_b['total_seats']} seats, grid_cells OK")

# ── Test 7: Draw functions ────────────────────────────────────────────────────
dummy_frame = np.zeros((600, 640, 3), dtype=np.uint8)
draw_row_lines(dummy_frame, rows)
for r_idx, row in enumerate(rows):
    for det in row:
        draw_seat(dummy_frame, det, r_idx)
print("  [OK] draw_seat + draw_row_lines: no crash")

# ── Test 8: Grid image ────────────────────────────────────────────────────────
grid = build_grid_image(dummy_frame, rows, 640, 600)
assert grid.shape == dummy_frame.shape
print("  [OK] build_grid_image: no crash")

# ── Test 9: Edge cases ────────────────────────────────────────────────────────
empty_rows = cluster_into_rows([], frame_h=600)
assert empty_rows == []
print("  [OK] empty detections handled")

single = [mock_dets[0]]
single_rows = cluster_into_rows(single, frame_h=600)
assert len(single_rows) == 1 and len(single_rows[0]) == 1
print("  [OK] single seat handled")

# ── Test 10: Syntax ───────────────────────────────────────────────────────────
import ast
src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scan_classroom.py")
ast.parse(open(src_path, encoding='utf-8').read())
print("  [OK] scan_classroom.py syntax clean")

print()
print("=" * 50)
print("ALL SCANNER TESTS PASSED")
print("=" * 50)
