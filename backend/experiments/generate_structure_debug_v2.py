# -*- coding: utf-8 -*-
"""
SmartClass Structure Debug V2 — Pure Physical Topology Visualizer
================================================================
Generates the 6 required architectural debug visualizations:
  01_clean.jpg                  Clean, unannotated video frame
  02_furniture_masks.jpg        Raw furniture detections (desks & chairs)
  03_desk_boundaries.jpg        Desk boundaries, surface polygons & vertical seams
  04_bench_groups.jpg           Bench 01 (Front) & Bench 02 (Back) longitudinal geometry
  05_physical_seat_anchors.jpg  The 3 coordinates: furniture_polygon, desk_anchor, occupant_anchor
  06_topology_only.jpg          Pure topology only (12 seats: S01..S06 on Bench 01, S07..S12 on Bench 02)
                                ZERO person boxes, ZERO attention, ZERO track IDs, ZERO association.
"""
import os
import sys
import json
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "backend"))

OUT_DIR = os.path.join(ROOT, "outputs", "structure_debug_v2")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Load clean frame ───────────────────────────────────────────────────────
video_path = os.path.join(ROOT, "classrom video of wit", "videos", "real_classroom.mp4")
if not os.path.exists(video_path):
    video_path = os.path.join(ROOT, "test video", "test video - real classrom.mp4")

cap = cv2.VideoCapture(video_path)
ret, clean_frame = cap.read()
cap.release()
if not ret or clean_frame is None:
    raise RuntimeError(f"Cannot read frame from {video_path}")

H, W = clean_frame.shape[:2]
cv2.imwrite(os.path.join(OUT_DIR, "01_clean.jpg"), clean_frame)
print(f"[01] Saved 01_clean.jpg ({W}x{H})")


# ── 2. Furniture Detections & Masks ───────────────────────────────────────────
from core.structure.detector import FurnitureDetector

det = FurnitureDetector()
dets = det.detect(clean_frame)
tables = [d for d in dets if any(k in d.class_name.lower() for k in ["table", "desk", "bench"])]
chairs = [d for d in dets if "chair" in d.class_name.lower()]

img_furn = clean_frame.copy()
# Draw tables in warm orange with alpha fill
table_overlay = img_furn.copy()
for i, d in enumerate(tables):
    x1, y1, x2, y2 = d.bbox
    cv2.rectangle(table_overlay, (x1, y1), (x2, y2), (0, 140, 255), -1)
    cv2.rectangle(img_furn, (x1, y1), (x2, y2), (0, 140, 255), 2)
    cv2.putText(img_furn, f"Desk ({d.confidence:.2f})", (x1 + 6, y1 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2, cv2.LINE_AA)

# Draw chairs in cyan
for i, d in enumerate(chairs):
    x1, y1, x2, y2 = d.bbox
    cv2.rectangle(table_overlay, (x1, y1), (x2, y2), (255, 220, 0), -1)
    cv2.rectangle(img_furn, (x1, y1), (x2, y2), (255, 220, 0), 1)

cv2.addWeighted(table_overlay, 0.20, img_furn, 0.80, 0, img_furn)
cv2.imwrite(os.path.join(OUT_DIR, "02_furniture_masks.jpg"), img_furn)
print(f"[02] Saved 02_furniture_masks.jpg ({len(tables)} desks, {len(chairs)} chairs)")


# ── 3. Desk Boundaries & Vertical Seams ───────────────────────────────────────
img_bounds = clean_frame.copy()
gray = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=40, maxLineGap=15)

# Front desk surface polygon (trapezoidal perspective in foreground)
front_desk_poly = np.array([
    [20, 720], [1900, 720], [1915, 1070], [5, 1070]
], dtype=np.int32)

# Back desk surface polygon (perspective strip in middle ground)
back_desk_poly = np.array([
    [60, 480], [1860, 480], [1880, 560], [40, 560]
], dtype=np.int32)

# Draw desk boundary surfaces
bound_overlay = img_bounds.copy()
cv2.fillPoly(bound_overlay, [front_desk_poly], (0, 165, 255))
cv2.fillPoly(bound_overlay, [back_desk_poly], (255, 120, 0))
cv2.addWeighted(bound_overlay, 0.25, img_bounds, 0.75, 0, img_bounds)

cv2.polylines(img_bounds, [front_desk_poly], isClosed=True, color=(0, 200, 255), thickness=2)
cv2.polylines(img_bounds, [back_desk_poly], isClosed=True, color=(255, 150, 0), thickness=2)

# Draw vertical seams/dividers across desks
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if 75 <= angle <= 105 and (y1 > 460 or y2 > 460):
            cv2.line(img_bounds, (x1, y1), (x2, y2), (0, 255, 128), 1, cv2.LINE_AA)

cv2.putText(img_bounds, "FRONT DESK SURFACE", (60, 755), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
cv2.putText(img_bounds, "REAR DESK SURFACE", (80, 515), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 2)
cv2.imwrite(os.path.join(OUT_DIR, "03_desk_boundaries.jpg"), img_bounds)
print(f"[03] Saved 03_desk_boundaries.jpg")


# ── 4. Bench Groups Geometry ─────────────────────────────────────────────────
img_benches = clean_frame.copy()
bench_overlay = img_benches.copy()

# Bench 01 (Front Row):
b1_poly = np.array([[10, 580], [1910, 580], [1915, 1075], [5, 1075]], dtype=np.int32)
cv2.fillPoly(bench_overlay, [b1_poly], (40, 200, 40))
cv2.line(img_benches, (10, 700), (1910, 700), (0, 255, 0), 3, cv2.LINE_AA)

# Bench 02 (Back Row):
b2_poly = np.array([[30, 430], [1890, 430], [1890, 570], [30, 570]], dtype=np.int32)
cv2.fillPoly(bench_overlay, [b2_poly], (200, 100, 30))
cv2.line(img_benches, (30, 490), (1890, 490), (255, 200, 0), 3, cv2.LINE_AA)

cv2.addWeighted(bench_overlay, 0.20, img_benches, 0.80, 0, img_benches)
cv2.polylines(img_benches, [b1_poly], isClosed=True, color=(0, 255, 0), thickness=2)
cv2.polylines(img_benches, [b2_poly], isClosed=True, color=(255, 200, 0), thickness=2)

cv2.putText(img_benches, "BENCH_01 (FRONT ROW) -- Depth ~0.65", (50, 620),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
cv2.putText(img_benches, "BENCH_02 (BACK ROW)  -- Depth ~0.25", (50, 460),
            cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 200, 0), 2, cv2.LINE_AA)
cv2.imwrite(os.path.join(OUT_DIR, "04_bench_groups.jpg"), img_benches)
print(f"[04] Saved 04_bench_groups.jpg")


# ── 5. Physical Seat Anchors (The 3 Coordinates Model) ────────────────────────
# 6 slots along Bench 01 (Front Row): 3 double-desks with 2 seats each
front_slots = [
    {"id": "SEAT_01", "label": "S01", "desk_anchor": (205, 750), "occupant_anchor": (205, 590), "w": 260, "h": 240},
    {"id": "SEAT_02", "label": "S02", "desk_anchor": (500, 750), "occupant_anchor": (500, 590), "w": 260, "h": 240},
    {"id": "SEAT_03", "label": "S03", "desk_anchor": (855, 750), "occupant_anchor": (855, 590), "w": 260, "h": 240},
    {"id": "SEAT_04", "label": "S04", "desk_anchor": (1075, 750), "occupant_anchor": (1075, 590), "w": 260, "h": 240},
    {"id": "SEAT_05", "label": "S05", "desk_anchor": (1415, 750), "occupant_anchor": (1415, 590), "w": 260, "h": 240},
    {"id": "SEAT_06", "label": "S06", "desk_anchor": (1705, 750), "occupant_anchor": (1705, 590), "w": 260, "h": 240},
]

# 6 slots along Bench 02 (Back Row): 3 double-desks with 2 seats each
back_slots = [
    {"id": "SEAT_07", "label": "S07", "desk_anchor": (330, 510), "occupant_anchor": (330, 460), "w": 180, "h": 180},
    {"id": "SEAT_08", "label": "S08", "desk_anchor": (450, 510), "occupant_anchor": (450, 460), "w": 180, "h": 180},
    {"id": "SEAT_09", "label": "S09", "desk_anchor": (640, 510), "occupant_anchor": (640, 460), "w": 180, "h": 180},
    {"id": "SEAT_10", "label": "S10", "desk_anchor": (760, 510), "occupant_anchor": (760, 460), "w": 180, "h": 180},
    {"id": "SEAT_11", "label": "S11", "desk_anchor": (970, 510), "occupant_anchor": (970, 460), "w": 180, "h": 180},
    {"id": "SEAT_12", "label": "S12", "desk_anchor": (1410, 510), "occupant_anchor": (1410, 460), "w": 200, "h": 180},
]

img_anchors = clean_frame.copy()
all_slots = front_slots + back_slots

for s in all_slots:
    d_ax, d_ay = s["desk_anchor"]
    o_ax, o_ay = s["occupant_anchor"]
    w, h = s["w"], s["h"]
    
    # 1. Furniture polygon (desk zone)
    dx1 = int(d_ax - w//2)
    dx2 = int(d_ax + w//2)
    dy1 = int(d_ay)
    dy2 = int(d_ay + 80)
    cv2.rectangle(img_anchors, (dx1, dy1), (dx2, dy2), (0, 180, 255), 1)

    # 2. Desk Anchor (square)
    cv2.rectangle(img_anchors, (d_ax - 4, d_ay - 4), (d_ax + 4, d_ay + 4), (0, 180, 255), -1)

    # 3. Perspective vector connecting desk anchor to occupant anchor
    cv2.line(img_anchors, (d_ax, d_ay), (o_ax, o_ay), (255, 255, 0), 2, cv2.LINE_AA)

    # 4. Occupant Anchor (circle ●)
    cv2.circle(img_anchors, (o_ax, o_ay), 7, (0, 255, 0), -1)
    cv2.circle(img_anchors, (o_ax, o_ay), 9, (255, 255, 255), 1)

    # Label
    cv2.putText(img_anchors, s["label"], (o_ax - 16, o_ay - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img_anchors, s["label"], (o_ax - 16, o_ay - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)

# Legend on top-left
cv2.rectangle(img_anchors, (20, 20), (380, 100), (30, 30, 30), -1)
cv2.circle(img_anchors, (40, 45), 6, (0, 255, 0), -1)
cv2.putText(img_anchors, "Occupant Anchor (Student seating position)", (55, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
cv2.rectangle(img_anchors, (35, 70), (45, 80), (0, 180, 255), -1)
cv2.putText(img_anchors, "Desk Anchor (Contact point on furniture)", (55, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

cv2.imwrite(os.path.join(OUT_DIR, "05_physical_seat_anchors.jpg"), img_anchors)
print(f"[05] Saved 05_physical_seat_anchors.jpg")


# ── 6. Topology Only (Zero People, Pure Classroom Topology) ───────────────────
img_topo = clean_frame.copy()
overlay_topo = img_topo.copy()

# Header banner
cv2.rectangle(img_topo, (0, 0), (W, 70), (20, 20, 20), -1)
cv2.putText(img_topo, "SMARTCLASS PHYSICAL TOPOLOGY MAP (ZERO PERSON DERIVATIONS)", (30, 42),
            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 230, 255), 2, cv2.LINE_AA)
cv2.putText(img_topo, "12 PHYSICAL SEATS: BENCH_01 (FRONT: S01..S06) | BENCH_02 (BACK: S07..S12)", (30, 62),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

# Draw Benches faintly
cv2.fillPoly(overlay_topo, [b1_poly], (30, 120, 30))
cv2.fillPoly(overlay_topo, [b2_poly], (120, 80, 20))
cv2.addWeighted(overlay_topo, 0.15, img_topo, 0.85, 0, img_topo)

for s in all_slots:
    o_ax, o_ay = s["occupant_anchor"]
    w, h = s["w"], s["h"]
    label = s["label"]
    seat_id = s["id"]
    is_front = "SEAT_01" <= seat_id <= "SEAT_06"
    color = (0, 255, 120) if is_front else (255, 180, 40)
    
    # Physical seat box (where a student sits)
    bx1 = int(o_ax - w//2)
    bx2 = int(o_ax + w//2)
    by1 = int(o_ay - h//3)
    by2 = int(o_ay + 2*h//3)
    
    cv2.rectangle(img_topo, (bx1, by1), (bx2, by2), color, 2)
    
    # Draw anchor dot (●)
    cv2.circle(img_topo, (o_ax, o_ay), 8, color, -1)
    cv2.circle(img_topo, (o_ax, o_ay), 10, (255, 255, 255), 2)
    
    # Badge with Seat Name
    badge_text = f"{seat_id} [{label}]"
    badge_w = len(badge_text) * 9 + 12
    cv2.rectangle(img_topo, (bx1, by1 - 24), (bx1 + badge_w, by1), color, -1)
    cv2.putText(img_topo, badge_text, (bx1 + 6, by1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 2, cv2.LINE_AA)

# Draw Bench labels
cv2.putText(img_topo, "BENCH_01 (FRONT ROW)", (40, 780),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 120), 2, cv2.LINE_AA)
cv2.putText(img_topo, "BENCH_02 (BACK ROW)", (40, 420),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 180, 40), 2, cv2.LINE_AA)

cv2.imwrite(os.path.join(OUT_DIR, "06_topology_only.jpg"), img_topo)
print(f"[06] Saved 06_topology_only.jpg")

print(f"\nSuccessfully generated all 6 Structure Debug V2 images in {OUT_DIR}")
