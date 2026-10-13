import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend"))
import core.config as cfg; cfg.YOLO_CONF = 0.22
import cv2
from core.detector import detect_faces_and_phones

ROOT  = os.path.dirname(os.path.dirname(__file__))
d = json.load(open(os.path.join(ROOT, "v3_layout.json")))

print("Scaffold anchors:")
for s in d["seats"]:
    print(f"  {s['seat_id']}  x={s['anchor'][0]:.0f}  y={s['anchor'][1]:.0f}")

print()
cap = cv2.VideoCapture(os.path.join(ROOT, "test video", "test video - real classrom.mp4"))
cap.set(cv2.CAP_PROP_POS_FRAMES, 5); ret, frame = cap.read(); cap.release()
W = int(cap.get(3))

faces, _, _, _ = detect_faces_and_phones(frame)
sorted_faces = sorted(faces, key=lambda f: (f["bbox"][1]+f["bbox"][3])/2)

# Group by person_bottom split at 685
back  = [f for f in sorted_faces if f["person_bbox"][3] <= 685]
front = [f for f in sorted_faces if f["person_bbox"][3] >  685]

print("Runtime back-row faces (sorted by cx):")
for f in sorted(back, key=lambda f:(f["bbox"][0]+f["bbox"][2])/2):
    cx=(f["bbox"][0]+f["bbox"][2])//2; cy=(f["bbox"][1]+f["bbox"][3])//2
    print(f"  cx={cx}  cy={cy}  person_bottom={f['person_bbox'][3]}")

print("Runtime front-row faces (sorted by cx):")
for f in sorted(front, key=lambda f:(f["bbox"][0]+f["bbox"][2])/2):
    cx=(f["bbox"][0]+f["bbox"][2])//2; cy=(f["bbox"][1]+f["bbox"][3])//2
    print(f"  cx={cx}  cy={cy}  person_bottom={f['person_bbox'][3]}")
