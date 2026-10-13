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

raw_dets, _, _, _ = detect_faces_and_phones(frame, min_conf=0.25, persist=False)
out_dir = os.path.join(HERE, "crops_check")
os.makedirs(out_dir, exist_ok=True)

debug_img = frame.copy()
for i, d in enumerate(raw_dets):
    pb = [int(v) for v in d["person_bbox"]]
    fb = [int(v) for v in d["bbox"]]
    cv2.rectangle(debug_img, (pb[0], pb[1]), (pb[2], pb[3]), (0, 255, 0), 2)
    cv2.rectangle(debug_img, (fb[0], fb[1]), (fb[2], fb[3]), (0, 0, 255), 2)
    cv2.putText(debug_img, f"#{i+1}", (fb[0], fb[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Save face crop
    fc = frame[max(0,fb[1]):max(0,fb[3]), max(0,fb[0]):max(0,fb[2])]
    if fc.size > 0:
        cv2.imwrite(os.path.join(out_dir, f"det_{i+1}_face.jpg"), fc)

cv2.imwrite(os.path.join(HERE, "all_dets_frame0.jpg"), debug_img)
print(f"Saved {len(raw_dets)} detections and visual image to {os.path.join(HERE, 'all_dets_frame0.jpg')}")
