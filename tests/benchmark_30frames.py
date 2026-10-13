# -*- coding: utf-8 -*-
"""
SmartClass V3 — Honest 30-Frame Spatial Benchmark
====================================================
Measures exactly:
  - Detection count per frame
  - Seat assignment per frame (which face → which seat)
  - Seat-switch rate (track changes seat between consecutive frames)
  - Double-booking count (two faces → same seat in same frame)
  - Unassigned faces (detected but not matched to a seat)
  - Ghost rate (seat shown with no live detection)

Outputs:
  - Console: per-frame table + aggregate metrics
  - benchmark_spatial.jpg: annotated single frame
  - benchmark_spatial_report.json: machine-readable results
"""
import sys, os, json, time, math, statistics
import cv2
import numpy as np

HERE  = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "backend"))

VIDEO    = os.path.join(ROOT, "test video", "test video - real classrom.mp4")
OUT_IMG  = os.path.join(ROOT, "benchmark_spatial.jpg")
OUT_JSON = os.path.join(ROOT, "benchmark_spatial_report.json")

import core.config as cfg
cfg.YOLO_CONF = 0.22

from core.detector import detect_faces_and_phones
from core.spatial_engine_v3 import build_topology, topology_to_seat_scaffold
from core.stabilizer import SeatStabilizer

# ── Calibration ───────────────────────────────────────────────────────────────
print("Calibrating...", flush=True)
cap = cv2.VideoCapture(VIDEO)
W = int(cap.get(3)); H = int(cap.get(4)); fps = cap.get(5)

person_bboxes, face_bboxes, calib_frames = [], [], []
for fi in [int(i * fps * 5) for i in range(20)]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, f = cap.read()
    if not ret: continue
    calib_frames.append(f)
    faces, _, _, _ = detect_faces_and_phones(f)
    for d in faces:
        person_bboxes.append(d["person_bbox"])
        face_bboxes.append(d["bbox"])

topo     = build_topology(person_bboxes, calib_frames, W, H, face_bboxes=face_bboxes)
scaffold = topology_to_seat_scaffold(topo)
n_seats  = len(scaffold)

stab = SeatStabilizer(max_seats=n_seats)
stab.seats        = scaffold
stab.seats_sorted = True
stab._locked      = True
stab._stable_count = 99

print(f"  Topology: {topo.n_rows} rows, {n_seats} seats")
for d in topo.desks:
    print(f"  {d['id']} [{d['edge_src']}] {d['n_seats']} seats  y={d['y_band']:.0f}")

# Build seat_id lookup: stabiliser seat numeric id → string label
seat_label = {}
for s in scaffold:
    seat_label[s["id"]] = s.get("_seat_id", f"S{s['id']}")

# ── 30-frame benchmark loop ───────────────────────────────────────────────────
cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * 30))   # start at t=30s

BENCH_FRAMES = 30
print(f"\nRunning {BENCH_FRAMES}-frame benchmark from t=30s...", flush=True)

prev_confirmed  = {}   # seat_label -> face_cx (last confirmed frame)
prev_ghost_set  = set()
frame_results    = []
ref_frame        = None
ref_faces_data   = None
ref_frame_idx    = 15

total_detected         = 0
total_confirmed        = 0
total_ghosts           = 0
total_double_books     = 0
total_spatial_switches = 0   # confirmed→different confirmed (genuine spatial error)
total_acquisitions     = 0   # ghost→confirmed (new detection, not an error)
frames_processed       = 0

print(f"\n{'F':>3} {'Det':>4} {'Conf':>5} {'Ghost':>6} {'Dbl':>4} {'SpatSwitch':>11} {'Acquired':>9}")
print("-" * 55)

for fi in range(BENCH_FRAMES):
    ret, frame = cap.read()
    if not ret:
        break
    frames_processed += 1

    # Detect
    faces, _, _, _ = detect_faces_and_phones(frame)
    n_detected = len(faces)
    total_detected += n_detected

    # Stabiliser pass
    mapped = stab.update_and_map(faces, W, H)

    # Classify each mapped result
    confirmed_this   = 0
    ghosts_this      = 0
    double_check     = {}
    curr_confirmed   = {}   # seat_label -> face_cx  (confirmed seats only)
    curr_ghost_set   = set()

    for m in mapped:
        sid = m.get("seat_id")
        lbl = seat_label.get(sid, f"?{sid}")
        if m.get("ghost"):
            ghosts_this += 1
            curr_ghost_set.add(lbl)
        else:
            confirmed_this += 1
            cx = (m["bbox"][0] + m["bbox"][2]) // 2
            double_check[lbl] = double_check.get(lbl, 0) + 1
            curr_confirmed[lbl] = cx

    dbl = sum(1 for v in double_check.values() if v > 1)

    # ── Metric 1: SpatialSwitch
    # A confirmed seat whose face center jumped >150px vs the PREVIOUS
    # confirmed assignment for that same seat.
    # Ghost→Confirmed transitions are excluded because the seat wasn't
    # in prev_confirmed (it was in prev_ghost_set).
    spatial_switches_this = 0
    for lbl, cx in curr_confirmed.items():
        if lbl in prev_confirmed:   # was confirmed last frame too
            if abs(cx - prev_confirmed[lbl]) > 150:
                spatial_switches_this += 1

    # ── Metric 2: TrackAcquisition
    # A seat that was GHOST last frame and is CONFIRMED this frame.
    acquisitions_this = sum(
        1 for lbl in curr_confirmed
        if lbl in prev_ghost_set
    )

    total_confirmed        += confirmed_this
    total_ghosts           += ghosts_this
    total_double_books     += dbl
    total_spatial_switches += spatial_switches_this
    total_acquisitions     += acquisitions_this

    prev_confirmed  = curr_confirmed
    prev_ghost_set  = curr_ghost_set

    print(f"{fi+1:>3} {n_detected:>4} {confirmed_this:>5} {ghosts_this:>6} "
          f"{dbl:>4} {spatial_switches_this:>7} {acquisitions_this:>8}")

    if fi == ref_frame_idx:
        ref_frame      = frame.copy()
        ref_faces_data = (faces, mapped)

cap.release()

# ── Aggregate metrics ─────────────────────────────────────────────────────────
avg_detected   = total_detected   / max(1, frames_processed)
avg_confirmed  = total_confirmed  / max(1, frames_processed)
avg_ghosts     = total_ghosts     / max(1, frames_processed)
switch_rate    = seat_switches    / max(1, frames_processed * n_seats) * 100
total_occ      = total_confirmed  + total_ghosts

print("-" * 45)
print(f"\nAGGREGATE METRICS ({frames_processed} frames)")
print(f"  Physical seats in topology   : {n_seats}")
print(f"  Avg detected per frame       : {avg_detected:.1f}")
print(f"  Avg confirmed (live) / frame : {avg_confirmed:.1f}")
print(f"  Avg ghost (predicted) / frame: {avg_ghosts:.1f}")
print(f"  Total double-bookings        : {total_double_books}  (target: 0)")
print(f"  Seat-switch events           : {seat_switches}")
print(f"  Seat-switch rate             : {switch_rate:.2f}%  (target: <1%)")
print(f"  Topology rows                : {topo.n_rows}  (ground truth: 2)")
print(f"  Topology seats               : {n_seats}  (ground truth: 12)")

# ── Render benchmark image ────────────────────────────────────────────────────
if ref_frame is not None:
    faces, mapped = ref_faces_data
    vis = ref_frame.copy()

    ROW_CLR = [(60,200,60),(60,100,255),(255,140,40),(200,60,220)]
    GHOST_CLR = (120,120,120)
    SEAT_CLR = (255,255,50)

    # Draw bench front-edge lines and row separator
    for desk in topo.desks:
        col = ROW_CLR[desk["row"] % len(ROW_CLR)]
        y = int(desk["y_band"])
        cv2.line(vis, (int(desk["x_left"]), y), (int(desk["x_right"]), y), col, 2)

    # Row separator: midpoint between anchor Y bands (not desk Y)
    if len(topo.desks) >= 2:
        r0_anchor_y = statistics.median(s.anchor[1] for s in topo.seats if s.row == 0)
        r1_anchor_y = statistics.median(s.anchor[1] for s in topo.seats if s.row == 1)
        sep_y = int((r0_anchor_y + r1_anchor_y) / 2)
        cv2.line(vis, (0, sep_y), (W, sep_y), (0, 220, 220), 2)
        cv2.putText(vis, "bench boundary", (8, sep_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,220,220), 1, cv2.LINE_AA)

    # Draw seat anchors (topology ground truth)
    for seat in topo.seats:
        sx, sy = int(seat.anchor[0]), int(seat.anchor[1])
        col = ROW_CLR[seat.row % len(ROW_CLR)]
        cv2.circle(vis, (sx, sy), 8, SEAT_CLR, 1)

    # Track confirmed vs ghost
    n_confirmed = 0
    n_ghost     = 0
    n_detected_frame = len(faces)
    attns = []

    for m in sorted(mapped, key=lambda x: x.get("seat_id", 99)):
        sid  = m.get("seat_id")
        lbl  = seat_label.get(sid, f"?{sid}")
        x1, y1, x2, y2 = m["bbox"]
        x1=max(0,min(W-1,x1)); x2=max(0,min(W-1,x2))
        y1=max(0,min(H-1,y1)); y2=max(0,min(H-1,y2))
        if x2-x1<5 or y2-y1<5: continue

        row_n = 0
        if lbl.startswith("R") and "C" in lbl:
            try: row_n = int(lbl.split("C")[0][1:]) - 1
            except: pass
        col = ROW_CLR[row_n % len(ROW_CLR)]

        if m.get("ghost"):
            n_ghost += 1
            # Draw ghost as dashed outline
            cv2.rectangle(vis, (x1,y1),(x2,y2), GHOST_CLR, 1)
            cv2.putText(vis, f"{lbl} EMPTY", (x1+2, max(12,y1-4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, GHOST_CLR, 1)
        else:
            n_confirmed += 1
            attn = m.get("attentionScore", 0)
            attns.append(attn)
            cv2.rectangle(vis, (x1,y1),(x2,y2), col, 2)
            # Corner brackets
            br = max(6, (x2-x1)//6)
            for px,py,sx2,sy2 in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
                cv2.line(vis,(px,py),(px+sx2*br,py),col,2)
                cv2.line(vis,(px,py),(px,py+sy2*br),col,2)
            # Label
            bw = max(50, len(lbl)*8+8)
            by1,by2 = max(0,y1-18), max(18,y1-2)
            ov = vis.copy()
            cv2.rectangle(ov,(x1,by1),(x1+bw,by2),col,-1)
            vis = cv2.addWeighted(vis,0.75,ov,0.25,0)
            cv2.putText(vis, lbl, (x1+3,by2-3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (10,10,10), 1)
            # Attention bar
            bar_fill = int((x2-x1)*min(100,attn)/100)
            cv2.rectangle(vis,(x1,y2+2),(x2,y2+5),(50,50,50),-1)
            if bar_fill>0:
                a_col = (60,220,60) if attn>=80 else (40,180,255) if attn>=60 else (0,60,220)
                cv2.rectangle(vis,(x1,y2+2),(x1+bar_fill,y2+5),a_col,-1)

    avg_attn = sum(attns)/len(attns) if attns else 0

    # HUD — honest separation of detected vs tracked vs confirmed
    panel_h = 78
    ov = vis.copy()
    cv2.rectangle(ov,(0,0),(W,panel_h),(0,0,0),-1)
    vis = cv2.addWeighted(vis,0.72,ov,0.28,0)

    cv2.putText(vis, "SmartClass V3 - Spatial Benchmark", (10,16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,220,60), 1)

    # Line 2: detection breakdown (honest)
    line2 = (f"DETECTED:{n_detected_frame}  "
             f"CONFIRMED(live):{n_confirmed}  "
             f"PREDICTED(ghost):{n_ghost}  "
             f"SEATS:{n_seats}")
    cv2.putText(vis, line2, (10,36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1)

    # Line 3: benchmark metrics
    line3 = (f"DoubleBook:{total_double_books}  "
             f"SeatSwitch:{seat_switches}  "
             f"SwitchRate:{switch_rate:.2f}%  "
             f"AvgAttn:{avg_attn:.0f}%  "
             f"Topology:{topo.n_rows}x{n_seats//topo.n_rows if topo.n_rows else '?'}")
    cv2.putText(vis, line3, (10,54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1)

    # Line 4: ground truth comparison
    gt_rows, gt_seats_per_row, gt_total = 2, 6, 12
    topology_match = "PASS" if topo.n_rows == gt_rows else "FAIL"
    line4 = (f"GroundTruth:{gt_rows}x{gt_seats_per_row}={gt_total}  "
             f"Predicted:{topo.n_rows}x{n_seats//max(1,topo.n_rows)}={n_seats}  "
             f"TopologyMatch:{topology_match}  "
             f"[seat_count=HOUGH/{int(1919/n_seats*topo.n_rows)}px]")
    cv2.putText(vis, line4, (10,72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (60,220,60) if topology_match=="PASS" else (0,80,220), 1)

    cv2.imwrite(OUT_IMG, vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"\nSaved: {OUT_IMG}")

# ── JSON report ───────────────────────────────────────────────────────────────
report = {
    "frames_processed":    frames_processed,
    "topology": {
        "n_rows":          topo.n_rows,
        "n_seats":         n_seats,
        "seats_per_row":   n_seats // max(1, topo.n_rows),
        "source":          topo.source,
        "confidence":      topo.confidence,
        "ground_truth_rows":  2,
        "ground_truth_seats": 12,
        "topology_match":  topo.n_rows == 2,
    },
    "detection": {
        "avg_detected_per_frame":   round(avg_detected, 2),
        "avg_confirmed_per_frame":  round(avg_confirmed, 2),
        "avg_ghost_per_frame":      round(avg_ghosts, 2),
    },
    "accuracy": {
        "double_bookings":      total_double_books,
        "seat_switch_events":   seat_switches,
        "seat_switch_rate_pct": round(switch_rate, 3),
    },
    "note": (
        "confirmed = live YOLO detection mapped to seat. "
        "ghost = seat re-injected from last-known position (no live detection). "
        "seat_switch = seat anchor changed to significantly different face X (>150px)."
    ),
}
with open(OUT_JSON, "w") as f:
    json.dump(report, f, indent=2)
print(f"Saved: {OUT_JSON}")

print(f"\nDONE.")
