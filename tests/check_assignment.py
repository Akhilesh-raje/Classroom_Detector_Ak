"""Show exactly which face gets assigned to which seat using real engine."""
import sys, os, math, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend"))
import core.config as cfg; cfg.YOLO_CONF = 0.22
import cv2
from core.detector import detect_faces_and_phones
from core.spatial_engine_v3 import build_topology, topology_to_seat_scaffold
from core.stabilizer import SeatStabilizer

ROOT  = os.path.dirname(os.path.dirname(__file__))
VIDEO = os.path.join(ROOT, "test video", "test video - real classrom.mp4")

cap = cv2.VideoCapture(VIDEO); W=int(cap.get(3)); H=int(cap.get(4)); fps=cap.get(5)

# Collect calibration data (same as run_v3.py)
person_bboxes, face_bboxes, calib_frames = [], [], []
for fi in [int(i*fps*5) for i in range(20)]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi); ret, f = cap.read()
    if not ret: continue
    calib_frames.append(f)
    faces, _, _, _ = detect_faces_and_phones(f)
    for d in faces:
        person_bboxes.append(d["person_bbox"])
        face_bboxes.append(d["bbox"])

cap.set(cv2.CAP_PROP_POS_FRAMES, 5); ret, frame = cap.read(); cap.release()

# Build topology
topo = build_topology(person_bboxes, calib_frames, W, H, face_bboxes=face_bboxes)
scaffold = topology_to_seat_scaffold(topo)

print(f"Scaffold: {len(scaffold)} seats")
for s in scaffold:
    print(f"  id={s['id']} _seat_id={s['_seat_id']} _row={s['_row']} center=({s['center'][0]:.0f},{s['center'][1]:.0f})")

# Build stabiliser and run one frame
stab = SeatStabilizer(max_seats=len(scaffold))
stab.seats = scaffold
stab.seats_sorted = True
stab._locked = True
stab._stable_count = 99

runtime_faces, _, _, _ = detect_faces_and_phones(frame)
mapped = stab.update_and_map(runtime_faces, W, H)

# Show Y boundaries used
row_ys = {}
for s in scaffold:
    r = s.get("_row")
    if r is not None:
        row_ys.setdefault(r, []).append(s["center"][1])
row_means = sorted([(r, sum(v)/len(v)) for r,v in row_ys.items()], key=lambda x:x[1])
print("\nY-band boundaries:")
for i,(ridx,ry) in enumerate(row_means):
    y_lo = row_means[i-1][1]+(ry-row_means[i-1][1])/2 if i>0 else 0
    y_hi = ry+(row_means[i+1][1]-ry)/2 if i<len(row_means)-1 else float("inf")
    print(f"  Row {ridx}: anchor_y={ry:.0f}  band=[{y_lo:.0f}, {y_hi:.0f}]")

print("\nFinal assignments (non-ghost):")
problems = 0
for m in sorted([x for x in mapped if not x.get("ghost")], key=lambda x: x.get("seat_id",99)):
    sid = m.get("seat_id")
    seat = next((s for s in scaffold if s["id"]==sid), None)
    if not seat: continue
    cx=(m["bbox"][0]+m["bbox"][2])//2; cy=(m["bbox"][1]+m["bbox"][3])//2
    row_label = seat["_seat_id"]
    expected_row = "R1" if cy < 518 else "R2"
    actual_row = row_label[:2]
    ok = "✓" if expected_row == actual_row else "✗ WRONG"
    if expected_row != actual_row: problems += 1
    print(f"  {row_label:<8} face_cx={cx:4d}  face_cy={cy:3d}  {ok}")

print(f"\n{problems} wrong assignments")
