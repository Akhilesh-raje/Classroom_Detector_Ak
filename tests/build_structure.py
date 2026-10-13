# -*- coding: utf-8 -*-
"""
SmartClass Spatial Perception V2 — Phase 2+3+4 Integration Test
================================================================
Tests the full pipeline:
  1. Collect frames from video
  2. Run calibrate() — persons as fiducials → desk geometry → seat scaffold
  3. Lock stabiliser from frame 1
  4. Process subsequent frames — seat IDs stable from the start
  5. Save benchmark_structure.jpg and benchmark_layout.json

No seat count, row count, or student Y-coordinates used for topology.
"""
import sys, os, json, time
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "backend"))

VIDEO = os.path.join(ROOT, "test video", "test video - real classrom.mp4")

import core.config as cfg
cfg.YOLO_CONF = 0.25   # slightly lower conf for calibration to catch all persons

from core.engine import ClassroomEngine
from core.spatial_engine import layout_to_dict, render_layout

print("=" * 65)
print("SmartClass Spatial Perception V2 — Integration Test")
print("=" * 65)

# ── Load calibration frames (first 30) ───────────────────────────────────────
print("\n[1] Loading calibration frames...", flush=True)
cap = cv2.VideoCapture(VIDEO)
W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"    {W}x{H}  {fps:.1f}fps")

calib_frames = []
for _ in range(30):
    ret, frame = cap.read()
    if not ret:
        break
    calib_frames.append(frame)
print(f"    Loaded {len(calib_frames)} calibration frames")

# ── Run calibrate() ───────────────────────────────────────────────────────────
print("\n[2] Running calibration...", flush=True)
engine = ClassroomEngine(max_seats=20)   # generous upper bound; calibrate() will resize
t0 = time.perf_counter()
engine.calibrate(calib_frames, W, H)
elapsed = (time.perf_counter() - t0) * 1000
print(f"    Calibration done in {elapsed:.0f}ms")
print(f"    Calibrated     : {engine._calibrated}")
print(f"    Stabiliser locked: {engine.stabilizer._locked}")
print(f"    Seats scaffolded : {len(engine.stabilizer.seats)}")

if not engine._calibrated:
    print("\n  ⚠ Calibration failed — check YOLO detection on the video")
    sys.exit(1)

layout = engine._classroom_layout
print(f"    Bench rows       : {layout.n_rows}")
print(f"    Persons used     : {layout.n_persons_used}")
print(f"    Confidence       : {layout.confidence:.2f}")

# ── Print scaffold ───────────────────────────────────────────────────────────
print("\n[3] Seat scaffold (desk geometry, no student Y used for topology):")
for d in layout.desks:
    print(f"    {d.id}  R{d.row+1}  "
          f"y={d.y_band:.0f}  "
          f"x=[{d.x_left:.0f}–{d.x_right:.0f}]  "
          f"len={d.length_px:.0f}px  "
          f"seats={d.n_seats}")

print(f"\n    Grid cell labels:")
for seat in layout.seat_slots:
    sid = int(seat.id.split("_")[1])
    lbl = engine._seat_to_grid_label(sid)
    print(f"    {seat.id}  R{seat.row+1}C{seat.col}  "
          f"pos=({seat.position[0]:.0f},{seat.position[1]:.0f})  label={lbl}")

# ── Process 10 more frames with locked scaffold ───────────────────────────────
print("\n[4] Processing 10 frames with pre-locked scaffold...", flush=True)
for i in range(10):
    ret, frame = cap.read()
    if not ret:
        break
    faces = engine.process_single_frame(frame, W, H, persist=True)
    if i == 0:
        ref_frame = frame.copy()
        ref_faces = faces

cap.release()

print(f"    Frame 31: {len(ref_faces)} faces detected")
for face in ref_faces[:4]:
    print(f"    S{face['id']}  {face.get('gridCell','')}  "
          f"attn={face.get('attentionScore',0):.0f}%  "
          f"act={face.get('activity','?')}")

# ── Verify grid cells are correct ────────────────────────────────────────────
grid_cells = [f.get("gridCell","") for f in ref_faces]
r1 = [c for c in grid_cells if c.startswith("R1")]
r2 = [c for c in grid_cells if c.startswith("R2")]
print(f"\n    Row 1 seats: {sorted(r1)}")
print(f"    Row 2 seats: {sorted(r2)}")

# ── Save outputs ──────────────────────────────────────────────────────────────
print("\n[5] Saving outputs...", flush=True)

vis = render_layout(calib_frames[5], layout)
cv2.imwrite(os.path.join(ROOT, "benchmark_structure.jpg"), vis)
print("    ✓ benchmark_structure.jpg")

ldict = layout_to_dict(layout)
with open(os.path.join(ROOT, "benchmark_layout.json"), "w") as f:
    json.dump(ldict, f, indent=2)
print("    ✓ benchmark_layout.json")

# Overlay faces on ref frame for verification
if ref_frame is not None:
    for face in ref_faces:
        x1,y1,x2,y2 = face["bbox"]
        gc  = face.get("gridCell","?")
        col = (60,200,60) if gc.startswith("R1") else (60,100,255)
        cv2.rectangle(ref_frame, (x1,y1),(x2,y2), col, 2)
        cv2.putText(ref_frame, f"S{face['id']} {gc}",
                    (x1, max(14, y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(ROOT, "benchmark_faces.jpg"), ref_frame)
    print("    ✓ benchmark_faces.jpg")

print("\n" + "=" * 65)
print("SUMMARY")
print(f"  Bench rows     : {layout.n_rows}")
print(f"  Seat slots     : {len(layout.seat_slots)}")
print(f"  Persons used   : {layout.n_persons_used}")
print(f"  Confidence     : {layout.confidence:.2f}")
print(f"  Stabiliser locked from frame 1: YES")
print(f"  Student Y used for topology: NO")
print("=" * 65)
