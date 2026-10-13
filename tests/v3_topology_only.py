# Just builds topology and saves images — no live frame processing
import sys, os, json
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "backend"))

import core.config as cfg; cfg.YOLO_CONF = 0.22
from core.detector import detect_faces_and_phones
from core.spatial_engine_v3 import build_topology, topology_to_dict, render_topology

VIDEO = os.path.join(ROOT, "test video", "test video - real classrom.mp4")

cap = cv2.VideoCapture(VIDEO)
W = int(cap.get(3)); H = int(cap.get(4)); fps = cap.get(5)

import time
calib_frames, face_bboxes, person_bboxes = [], [], []
for fi in [int(i * fps * 5) for i in range(20)]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, f = cap.read()
    if not ret: continue
    calib_frames.append(f)
    faces, _, _, _ = detect_faces_and_phones(f)
    for d in faces:
        face_bboxes.append(d["bbox"])        # head-crop bbox — matches runtime
        person_bboxes.append(d["person_bbox"])  # full body — for desk Y estimation

cap.release()
print(f"Frames:{len(calib_frames)}  face_bboxes:{len(face_bboxes)}  person_bboxes:{len(person_bboxes)}")

t0 = time.perf_counter()
# Pass person_bboxes for desk-Y and row clustering (body position)
# but face_bboxes so head_y = actual face center
topo = build_topology(person_bboxes, calib_frames, W, H, face_bboxes=face_bboxes)
print(f"Topology built in {(time.perf_counter()-t0)*1000:.0f}ms")
print(f"Rows:{topo.n_rows}  Seats:{len(topo.seats)}  conf:{topo.confidence:.2f}")

for d in topo.desks:
    print(f"  {d['id']}  y={d['y_band']:.0f}  "
          f"len={d['length_px']:.0f}px/{145}={d['n_seats']}seats  "
          f"n_persons={d['n_persons']}  [HOUGH {d['n_lines']}lines]")

for r in range(topo.n_rows):
    ids = [s.seat_id for s in topo.seats if s.row == r]
    print(f"  Row {r+1}: {' '.join(ids)}")

# Save images
vis = render_topology(calib_frames[0], topo)
cv2.imwrite(os.path.join(ROOT, "v3_topology.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
print("Saved v3_topology.jpg")

with open(os.path.join(ROOT, "v3_layout.json"), "w") as f:
    json.dump(topology_to_dict(topo), f, indent=2)
print("Saved v3_layout.json")
print(f"\nKEY RESULT: {topo.n_rows} rows, {len(topo.seats)} seats from HOUGH geometry")
print(f"Algorithm was told row/seat count: NO")
