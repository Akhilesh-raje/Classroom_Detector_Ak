"""
SmartClass AI — Classroom Scanner Tests
Tests all scanner logic without needing a camera.
"""
import sys
sys.path.insert(0, 'backend')
sys.path.insert(0, '.')

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
assert len(rows) == 2,       f"Expected 2 rows, got {len(rows)}"
assert len(rows[0]) == 3,    f"Expected 3 seats in row 1, got {len(rows[0])}"
assert len(rows[1]) == 3,    f"Expected 3 seats in row 2, got {len(rows[1])}"
print(f"  [OK] cluster_into_rows: {len(rows)} rows, {[len(r) for r in rows]} seats per row")

# ── Test 2: Left-to-right ordering ───────────────────────────────────────────
for r_idx, row in enumerate(rows):
    xs = [d["center"][0] for d in row]
    assert xs == sorted(xs), f"Row {r_idx+1} not sorted left-to-right: {xs}"
print("  [OK] rows sorted left-to-right")

# ── Test 3: Seat naming ───────────────────────────────────────────────────────
names = generate_seat_names(rows, ollama_available=False)
labels = [d["seat_label"] for row in rows for d in row]
assert labels == ["R1S1","R1S2","R1S3","R2S1","R2S2","R2S3"], f"Labels wrong: {labels}"
print(f"  [OK] seat labels: {labels}")

# ── Test 4: IoU ───────────────────────────────────────────────────────────────
b1 = [0,0,100,100]
b2 = [50,50,150,150]
b3 = [200,200,300,300]
assert 0.1 < iou(b1,b2) < 0.5, f"IoU b1/b2 wrong: {iou(b1,b2)}"
assert iou(b1,b3) == 0.0,       f"IoU b1/b3 should be 0"
print(f"  [OK] IoU: overlapping={iou(b1,b2):.3f}, non-overlapping={iou(b1,b3):.3f}")

# ── Test 5: Deduplication ─────────────────────────────────────────────────────
dup_dets = mock_dets + [
    {"bbox":[55,105,205,255],"cls":56,"cls_name":"chair","confidence":0.60,"center":[130.0,180.0],"area":22500},
]
deduped = deduplicate_detections(dup_dets, iou_thresh=0.4)
assert len(deduped) == 6, f"Expected 6 after dedup, got {len(deduped)}"
print(f"  [OK] deduplicate: {len(dup_dets)} → {len(deduped)} (removed 1 duplicate)")

# ── Test 6: Layout JSON ───────────────────────────────────────────────────────
layout = build_layout_json(rows, 640, 480, cam_idx=1, timestamp="2025-01-01 12:00:00")
assert layout["total_seats"] == 6
assert layout["total_rows"]  == 2
assert layout["max_cols"]    == 3
assert len(layout["seats"])  == 6
assert layout["seats"][0]["seat_label"] == "R1S1"
assert layout["seats"][5]["seat_label"] == "R2S3"
# JSON round-trip
j  = json.dumps(layout)
l2 = json.loads(j)
assert l2["total_seats"] == 6
print(f"  [OK] build_layout_json: {layout['total_seats']} seats, JSON round-trip OK")

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
# Empty detections
empty_rows = cluster_into_rows([], frame_h=600)
assert empty_rows == []
print("  [OK] empty detections handled")

# Single seat
single = [mock_dets[0]]
single_rows = cluster_into_rows(single, frame_h=600)
assert len(single_rows) == 1 and len(single_rows[0]) == 1
print("  [OK] single seat handled")

# ── Test 10: Syntax ───────────────────────────────────────────────────────────
import ast
ast.parse(open("scan_classroom.py").read())
print("  [OK] scan_classroom.py syntax clean")

print()
print("=" * 50)
print("ALL SCANNER TESTS PASSED")
print("=" * 50)
print()
print("To run the scanner with your 4K camera:")
print("  python scan_classroom.py")
print()
print("Controls:")
print("  SPACE = scan current frame")
print("  S     = save layout to classroom_layout.json")
print("  R     = retry / clear")
print("  Q     = quit (auto-saves if unsaved)")
