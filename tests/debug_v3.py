import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend"))
import core.config as cfg; cfg.YOLO_CONF = 0.22
import cv2
from core.detector import track_persons
from core.spatial_engine_v3 import _find_bench_bands, _validate_bands

ROOT  = os.path.dirname(os.path.dirname(__file__))
VIDEO = os.path.join(ROOT, "test video", "test video - real classrom.mp4")

cap = cv2.VideoCapture(VIDEO)
W = int(cap.get(3)); H = int(cap.get(4))
cap.set(cv2.CAP_PROP_POS_FRAMES, 5)
ret, frame = cap.read()
cap.release()

dets = track_persons(frame, persist=False)
bboxes = [d["bbox"] for d in dets]
bottoms = sorted([b[3] for b in bboxes])
print(f"Person bbox bottoms ({len(dets)}): {bottoms}")

bands = _find_bench_bands(frame, bboxes, W, H)
print(f"\n{len(bands)} bands:")
for b in bands:
    nearby = [bot for bot in bottoms if abs(bot - b["y"]) < 80]
    print(f"  y={b['y']:.0f}  len={b['length_px']:.0f}  n_lines={b['n_lines']}  nearby_bottoms={nearby}")

print("\nValidated (window band_y-40 to band_y+20):")
v = _validate_bands(bands, bboxes)
for b in v:
    print(f"  y={b['y']:.0f}  n_persons={b['n_persons_above']}")
