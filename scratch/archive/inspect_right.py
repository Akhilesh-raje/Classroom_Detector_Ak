import os
import sys
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from core.detector import detect_faces_and_phones, YOLO_CONF

video_path = os.path.join(ROOT, "test video", "test video - real classrom.mp4")
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

raw_dets, _, _, _ = detect_faces_and_phones(frame, min_conf=YOLO_CONF, persist=False)
print(f"Total detections on frame 0: {len(raw_dets)}")
for i, d in enumerate(raw_dets):
    pb = d["person_bbox"]
    fb = d["bbox"]
    cx = (fb[0] + fb[2]) / 2.0
    cy = (fb[1] + fb[3]) / 2.0
    print(f"Det {i+1}: head=({cx:.1f}, {cy:.1f}), person_bbox={[int(x) for x in pb]}")
