import os
import sys
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "backend"))

from core.engine import ClassroomEngine
from benchmark_v2_2 import compute_v2_2_metrics, print_v2_2_report

video_path = os.path.join(ROOT, "test video", "test video - real classrom.mp4")
cap = cv2.VideoCapture(video_path)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Sampling 60 calibration frames...")
calib_frames = []
for _ in range(60):
    ret, cf = cap.read()
    if not ret:
        break
    calib_frames.append(cf)

engine = ClassroomEngine()
engine.calibrate(calib_frames, W, H)

topo = engine.topology
print(f"Discovered {len(topo.seats)} seats across {topo.rows} rows (seats per row: {topo.cols_per_row}):")
for s in topo.seats:
    print(f"  {s.label}: anchor=({s.anchor[0]:.1f}, {s.anchor[1]:.1f})")

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
num_test_frames = 300
print(f"\nProcessing {num_test_frames} frames to evaluate association stability...")

for f_idx in range(num_test_frames):
    ret, frame = cap.read()
    if not ret:
        break
    faces = engine.process_single_frame(frame, W, H)
    if (f_idx + 1) % 50 == 0:
        print(f"  Frame {f_idx + 1}/{num_test_frames} done (faces: {len(faces)})")

cap.release()

csv_out = os.path.join(HERE, "benchmark_12seats_300f.csv")
engine.export_benchmark_csv(csv_out)

metrics = compute_v2_2_metrics(csv_out)
print_v2_2_report(metrics)
