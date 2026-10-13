import os
import sys
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from core.engine import ClassroomEngine
from core.topology import TopologyDiscovery, TopologyConfig

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
engine.calibrate(calib_frames[:30], W, H)

topo = engine.topology
print(f"\nDiscovered {len(topo.seats)} seats across {topo.rows} rows (30 frames):")
for s in topo.seats:
    print(f"  Seat {s.label}: row={s.row}, col={s.col}, anchor=({s.anchor[0]:.1f}, {s.anchor[1]:.1f}), bbox={[round(x,1) for x in s.bbox]}")

# Also test with 60 calibration frames
engine60 = ClassroomEngine()
engine60.calibrate(calib_frames, W, H)
topo60 = engine60.topology
print(f"\nDiscovered {len(topo60.seats)} seats across {topo60.rows} rows (60 frames):")
for s in topo60.seats:
    print(f"  Seat {s.label}: row={s.row}, col={s.col}, anchor=({s.anchor[0]:.1f}, {s.anchor[1]:.1f}), bbox={[round(x,1) for x in s.bbox]}")
