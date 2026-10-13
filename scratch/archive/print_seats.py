import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from core.engine import ClassroomEngine
import cv2

video_path = os.path.join(ROOT, "test video", "test video - real classrom.mp4")
cap = cv2.VideoCapture(video_path)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

calib_frames = []
for _ in range(60):
    ret, cf = cap.read()
    if not ret:
        break
    calib_frames.append(cf)
cap.release()

engine = ClassroomEngine()
engine.calibrate(calib_frames, W, H)

topo = engine.topology
print(f"Topology: {topo.rows} rows, seats per row: {topo.cols_per_row}")
for r in topo.row_geometries:
    print(f"  Row {r.label} ({r.semantic}): y_median={r.y_median:.1f}, y_band=({r.y_min:.1f}, {r.y_max:.1f})")
print("\nSeats:")
for s in topo.seats:
    print(f"  {s.label} ({s.row},{s.col}): anchor=({s.anchor[0]:.1f}, {s.anchor[1]:.1f}) col_bounds=({s.column_boundary_left:.1f}, {s.column_boundary_right:.1f})")
