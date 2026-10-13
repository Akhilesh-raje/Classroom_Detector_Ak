# -*- coding: utf-8 -*-
"""
Generate a fully-annotated test image showing:
  - Desk geometry (bench lines from calibration)
  - All detected faces with attention scores
  - Activity labels + grid cell positions
  - Global stats HUD
"""
import sys, os, time
import cv2
import numpy as np

HERE  = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "backend"))

VIDEO = os.path.join(ROOT, "test video", "test video - real classrom.mp4")
OUT   = os.path.join(ROOT, "test_result.jpg")

import core.config as cfg
cfg.YOLO_CONF = 0.25

from core.engine import ClassroomEngine
from core.spatial_engine import render_layout

# ── Palette ───────────────────────────────────────────────────────────────────
ROW_COLORS   = [(60, 200, 60), (60, 100, 255), (255, 140, 40), (200, 60, 220)]
ATTN_HIGH    = ( 60, 220,  60)   # ≥80%
ATTN_MED     = ( 40, 180, 255)   # 60–79%
ATTN_LOW     = (  0,  60, 220)   # <60%
TEXT_WHITE   = (255, 255, 255)
TEXT_BLACK   = ( 10,  10,  10)
PANEL_BG     = (  0,   0,   0)

def attn_color(score):
    if score >= 80: return ATTN_HIGH
    if score >= 60: return ATTN_MED
    return ATTN_LOW

def alpha_overlay(img, x1, y1, x2, y2, color, alpha=0.55):
    sub = img[y1:y2, x1:x2]
    rect = np.full_like(sub, color)
    img[y1:y2, x1:x2] = cv2.addWeighted(sub, 1-alpha, rect, alpha, 0)

# ── Load and calibrate ────────────────────────────────────────────────────────
print("Loading video...", flush=True)
cap = cv2.VideoCapture(VIDEO)
W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Collect calibration frames (first 60 for better coverage)
calib = []
for _ in range(60):
    ret, f = cap.read()
    if not ret: break
    calib.append(f)
print(f"  {len(calib)} calibration frames loaded")

engine = ClassroomEngine(max_seats=20)
print("Calibrating...", flush=True)
engine.calibrate(calib, W, H)
print(f"  {len(engine.stabilizer.seats)} seats locked, "
      f"{engine._classroom_layout.n_rows} rows")

# Process 20 more frames to build up EMA attention history
print("Processing frames for attention data...", flush=True)
ref_frame = None
ref_faces = None
for i in range(5):   # 5 frames is enough for EMA to settle
    ret, frame = cap.read()
    if not ret: break
    faces = engine.process_single_frame(frame, W, H, persist=True)
    ref_frame = frame.copy()
    ref_faces = faces

cap.release()
print(f"  {len(ref_faces)} faces tracked")

# ── Build the annotated image ─────────────────────────────────────────────────
vis = ref_frame.copy()

# 1. Draw desk geometry from calibration (faint bench lines)
layout = engine._classroom_layout
if layout:
    for desk in layout.desks:
        col = ROW_COLORS[desk.row % len(ROW_COLORS)]
        y   = int(desk.y_band)
        xl  = int(desk.x_left)
        xr  = int(desk.x_right)
        cv2.line(vis, (xl, y), (xr, y), col, 3)
        ov = vis.copy()
        cv2.rectangle(ov, (xl, y-4), (xr, y+4), col, -1)
        vis = cv2.addWeighted(vis, 0.80, ov, 0.20, 0)
        cv2.putText(vis, f"bench_{desk.row+1}  {desk.n_seats} seats",
                    (xl+4, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)

    rows_by_idx = {}
    for d in layout.desks:
        rows_by_idx.setdefault(d.row, []).append(d)
    for r_idx in range(layout.n_rows - 1):
        ra = rows_by_idx.get(r_idx, [])
        rb = rows_by_idx.get(r_idx+1, [])
        if ra and rb:
            sep = int((max(d.y_band for d in ra) + min(d.y_band for d in rb)) / 2)
            cv2.line(vis, (0, sep), (W, sep), (0, 220, 220), 2)
            cv2.putText(vis, "bench boundary", (8, sep-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,220,220), 1, cv2.LINE_AA)

# 2. Draw each tracked face
for face in ref_faces:
    sid   = face.get("id")
    ghost = face.get("ghost", False)
    if sid is None or ghost:
        continue                          # skip ghosts and unassigned

    gc    = face.get("gridCell", f"S{sid}")
    # Skip if gridCell is uninitialised / placeholder
    if not gc or "?" in str(gc):
        gc = f"S{sid}"

    attn  = face.get("attentionScore", 0)
    act   = face.get("reconciledActivity", face.get("activity", "neutral"))
    phone = face.get("phoneDetected", False)

    # Use head-crop bbox (tight), not person body bbox
    x1, y1, x2, y2 = face["bbox"]
    # Clamp to frame
    x1 = max(0, min(W-1, x1));  x2 = max(0, min(W-1, x2))
    y1 = max(0, min(H-1, y1));  y2 = max(0, min(H-1, y2))
    if x2 - x1 < 10 or y2 - y1 < 10:
        continue                          # degenerate bbox — skip

    bw = x2 - x1
    bh = y2 - y1

    # Row colour from gridCell
    row_num = 0
    if gc.startswith("R") and "C" in gc:
        try: row_num = int(gc.split("C")[0][1:]) - 1
        except: pass
    box_col = ROW_COLORS[row_num % len(ROW_COLORS)]
    a_col   = attn_color(attn)

    # Bounding box
    thickness = 1 if ghost else 2
    cv2.rectangle(vis, (x1, y1), (x2, y2), box_col, thickness)

    # Corner brackets
    br = max(8, bw // 6)
    for (px, py, sx, sy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(vis, (px, py), (px+sx*br, py), box_col, 2)
        cv2.line(vis, (px, py), (px, py+sy*br), box_col, 2)

    # Top badge: grid cell + seat id
    badge  = f"S{sid} {gc}"
    bw_px  = max(70, len(badge)*8 + 8)
    by1    = max(0, y1 - 20)
    by2    = max(20, y1 - 2)
    alpha_overlay(vis, x1, by1, min(W, x1+bw_px), by2, box_col, 0.80)
    cv2.putText(vis, badge, (x1+3, by2-4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, TEXT_BLACK, 1, cv2.LINE_AA)

    # Attention score (right of badge)
    a_str = f"{attn:.0f}%"
    cv2.putText(vis, a_str, (x1+3, y2-4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, a_col, 1, cv2.LINE_AA)

    # Attention bar (below bbox)
    bar_y = y2 + 3
    bar_w = bw
    cv2.rectangle(vis, (x1, bar_y), (x2, bar_y+4), (50,50,50), -1)
    fill = int(bar_w * min(100, attn) / 100)
    if fill > 0:
        cv2.rectangle(vis, (x1, bar_y), (x1+fill, bar_y+4), a_col, -1)

    # Activity label
    ACT_ICONS = {
        "studying":   "STUDY",   "attentive": "ATTN",
        "distracted": "DIST",    "drowsy":    "DRSY",
        "phone":      "PHONE",   "neutral":   "NTRL",
        "talking":    "TALK",    "fidgeting": "FIDG",
    }
    icon = ACT_ICONS.get(act, act.upper()[:6])
    cv2.putText(vis, icon, (x1+3, min(H-4, y2+16)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30,
                (80,220,80) if act in ("studying","attentive") else (180,180,100),
                1, cv2.LINE_AA)

    # Phone alert
    if phone:
        alpha_overlay(vis, x1, max(0,y1-38), x2, max(0,y1-22), (0,0,200), 0.85)
        cv2.putText(vis, "!! PHONE !!", (x1+3, max(10,y1-25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_WHITE, 2, cv2.LINE_AA)

# 3. Global HUD ─────────────────────────────────────────────────────────────
real_faces = [f for f in ref_faces if not f.get("ghost") and f.get("id") is not None]
attns   = [f.get("attentionScore", 0) for f in real_faces]
avg_cls = sum(attns)/len(attns) if attns else 0
n_study = sum(1 for f in real_faces
              if f.get("reconciledActivity","") in ("studying","attentive"))
n_dist  = sum(1 for f in real_faces
              if f.get("reconciledActivity","") in ("distracted","drowsy","phone"))
n_rows  = layout.n_rows if layout else "?"

panel_h = 70
alpha_overlay(vis, 0, 0, W, panel_h, (0,0,0), 0.72)

# Row 1 of HUD
cv2.putText(vis, "SmartClass AI  |  Spatial Perception V2", (10, 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,220,60), 1, cv2.LINE_AA)
cv2.putText(vis, f"[person_fiducial scaffold  |  NO STUDENT Y FOR TOPOLOGY]",
            (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160,160,160), 1, cv2.LINE_AA)

# Row 2 of HUD — stats
stats = (f"Students:{len(real_faces)}   "
         f"Rows:{n_rows}   "
         f"Seats:{len(engine.stabilizer.seats)}   "
         f"Class Avg Attn:{avg_cls:.0f}%   "
         f"Engaged:{n_study}   "
         f"Distracted:{n_dist}   "
         f"Locked from frame 1: YES")
cv2.putText(vis, stats, (10, 56),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT_WHITE, 1, cv2.LINE_AA)

# Attention colour legend (bottom-right)
legend_x = W - 200
legend_y = H - 55
for txt, col in [(">=80% Focused", ATTN_HIGH), ("60-79% Medium", ATTN_MED), ("<60% Low", ATTN_LOW)]:
    cv2.rectangle(vis, (legend_x, legend_y), (legend_x+12, legend_y+10), col, -1)
    cv2.putText(vis, txt, (legend_x+16, legend_y+10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, TEXT_WHITE, 1, cv2.LINE_AA)
    legend_y += 16

# ── Save ──────────────────────────────────────────────────────────────────────
cv2.imwrite(OUT, vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
print(f"\n✓ Saved: {OUT}")

# Print per-student summary
print("\nPer-student summary:")
print(f"  {'Seat':<8} {'Grid':<8} {'Attn':>6} {'Avg':>6} {'Activity':<20}")
print("  " + "-"*54)
for face in sorted(real_faces, key=lambda f: f.get("id",0)):
    sid  = face.get("id","?")
    gc   = face.get("gridCell","?")
    attn = face.get("attentionScore",0)
    avg  = face.get("avgAttention30",0)
    act  = face.get("reconciledActivity", face.get("activity","?"))
    print(f"  S{str(sid):<7} {gc:<8} {attn:>5.0f}%  {avg:>5.0f}%  {act}")

print(f"\n  Class avg attention: {avg_cls:.1f}%")
print(f"  Engaged   : {n_study}/{len(real_faces)}")
print(f"  Distracted: {n_dist}/{len(real_faces)}")
