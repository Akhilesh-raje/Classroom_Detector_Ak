import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend"))
import core.config as cfg; cfg.YOLO_CONF = 0.22
import cv2
from core.detector import detect_faces_and_phones

ROOT  = os.path.dirname(os.path.dirname(__file__))
VIDEO = os.path.join(ROOT, "test video", "test video - real classrom.mp4")

cap = cv2.VideoCapture(VIDEO)
W = int(cap.get(3)); H = int(cap.get(4))
cap.set(cv2.CAP_PROP_POS_FRAMES, 5)
ret, frame = cap.read(); cap.release()

faces, phones, laptops, earphones = detect_faces_and_phones(frame)
print(f"{len(faces)} faces detected at runtime:")
for f in sorted(faces, key=lambda x: (x['bbox'][1]+x['bbox'][3])/2):
    x1,y1,x2,y2 = f['bbox']
    cx=(x1+x2)//2; cy=(y1+y2)//2
    px1,py1,px2,py2 = f['person_bbox']
    print(f"  face_cy={cy}  face_h={y2-y1}  person_bottom={py2}  cx={cx}")
