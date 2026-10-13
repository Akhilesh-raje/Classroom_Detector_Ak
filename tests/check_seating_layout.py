# -*- coding: utf-8 -*-
"""
Check seating layout detection - visualises where seat centers sit
relative to the actual people in the frame.
"""
import sys, os, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.dirname(HERE)
BACKEND_PATH = os.path.join(ROOT_PATH, "backend")
sys.path.insert(0, BACKEND_PATH)

VIDEO = os.path.join(ROOT_PATH, "test video", "test video - real classrom.mp4")

import core.config as cfg
cfg.YOLO_CONF = 0.30

from core.engine import ClassroomEngine
from core.stabilizer import _discover_rows

print("Seating Layout Checker")
print("=" * 60)

cap = cv2.VideoCapture(VIDEO)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# max_seats=12 matches actual count so the stabilizer locks and EMA converges
engine = ClassroomEngine(max_seats=12)

print("Processing frames to stabilize seat detection...")
last_frame = None
for i in range(30):
    ret, frame = cap.read()
    if not ret:
        break
    last_frame = frame
    engine.process_single_frame(frame, W, H, persist=True)
    if i % 5 == 0:
        print(f"  Frame {i+1}: {len(engine.stabilizer.seats)} seats")

cap.release()

stabilizer = engine.stabilizer
print(f"\nTotal seats : {len(stabilizer.seats)}")
print(f"Sorted      : {stabilizer.seats_sorted}")
print(f"Locked      : {stabilizer._locked}")
print(f"Row layout  : {stabilizer.row_layout}\n")

rows = _discover_rows(stabilizer.seats)
rows_sorted = sorted(rows, key=lambda r: sum(s["center"][1] for s in r) / len(r))

# Print table
for r_idx, row in enumerate(rows_sorted, 1):
    row_by_x = sorted(row, key=lambda s: s["center"][0])
    print(f"Row {r_idx} ({len(row)} students):")
    for c_idx, seat in enumerate(row_by_x, 1):
        cx, cy = seat["center"]
        x1, y1, x2, y2 = seat["last_bbox"]
        print(f"  S{seat['id']:2d}  R{r_idx}C{c_idx}  "
              f"head=({cx:.0f},{cy:.0f})  bbox=({x1},{y1},{x2},{y2})")
    print()

# ── Visualization ────────────────────────────────────────────────────────────
if last_frame is None:
    print("No frame captured"); exit()

vis = last_frame.copy()
COLORS = [
    (255,  80,  80),   # Row 1 — blue (BGR)
    ( 80, 220,  80),   # Row 2 — green
    ( 80,  80, 255),   # Row 3 — red
    (255, 255,  80),   # Row 4 — cyan
]

for r_idx, row in enumerate(rows_sorted):
    col = COLORS[r_idx % len(COLORS)]
    row_by_x = sorted(row, key=lambda s: s["center"][0])
    for c_idx, seat in enumerate(row_by_x, 1):
        cx, cy = int(seat["center"][0]), int(seat["center"][1])
        x1, y1, x2, y2 = seat["last_bbox"]

        # Draw the person bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)

        # Draw dot at the EMA seat center (head crop center)
        cv2.circle(vis, (cx, cy), 8, col, -1)
        cv2.circle(vis, (cx, cy), 8, (255, 255, 255), 1)  # white outline

        # Label: seat ID + grid cell (r_idx is 0-indexed, display as 1-indexed)
        label = f"S{seat['id']} R{r_idx+1}C{c_idx}"
        cv2.putText(vis, label, (x1, max(16, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)

# Draw row separator line between rows (using person bbox bottoms/tops, not centers)
for r_idx in range(len(rows_sorted) - 1):
    # Bottom of last person in upper row
    upper_bottom = max(s["last_bbox"][3] for s in rows_sorted[r_idx])
    # Top of first person in lower row
    lower_top    = min(s["last_bbox"][1] for s in rows_sorted[r_idx + 1])
    sep_y = (upper_bottom + lower_top) // 2
    cv2.line(vis, (0, sep_y), (W, sep_y), (0, 220, 220), 2)
    cv2.putText(vis, f"--- row boundary ---", (10, sep_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 220), 1, cv2.LINE_AA)

out = os.path.join(ROOT_PATH, "seating_layout_check.jpg")
cv2.imwrite(out, vis)
print(f"✓ Visualization saved: {out}")
