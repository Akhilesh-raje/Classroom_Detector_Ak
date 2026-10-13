import os
import sys
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from core.engine import ClassroomEngine
from core.topology import TopologyDiscovery

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

discovery = TopologyDiscovery(frame_size=(W, H))
for f_idx, frame in enumerate(calib_frames):
    from core.detector import detect_faces_and_phones, YOLO_CONF
    raw_dets, _, _, _ = detect_faces_and_phones(frame, min_conf=YOLO_CONF, persist=True)
    discovery.accumulate_raw(raw_dets, frame_idx=f_idx)

stable_tracks = {}
for tid, data in discovery._tracks.items():
    if data["count"] >= 10:
        stable_tracks[tid] = {
            "tid": tid,
            "count": data["count"],
            "head_x": float(np.median(data["head_x"])),
            "head_y": float(np.median(data["head_y"])),
            "bot_y": float(np.median(data["bot_y"])),
            "bbox_h": float(np.median(data["bbox_h"])),
        }

print(f"{'TID':<5} {'Count':<6} {'Head X':>8} {'Head Y':>8} {'Bot Y':>8} {'Height':>8}")
print("-" * 50)
for tid, t in sorted(stable_tracks.items(), key=lambda x: x[1]["head_x"]):
    print(f"{t['tid']:<5} {t['count']:<6} {t['head_x']:>8.1f} {t['head_y']:>8.1f} {t['bot_y']:>8.1f} {t['bbox_h']:>8.1f}")
