import cv2, os, sys
sys.path.insert(0, 'backend')
from core.detector import detect_faces_and_phones

cap = cv2.VideoCapture(os.path.join('test video', 'test video - real classrom.mp4'))
cap.set(cv2.CAP_PROP_POS_FRAMES, 106)
ret, frame = cap.read()
cap.release()

dets, _, _, _ = detect_faces_and_phones(frame, min_conf=0.25, persist=True)
for i, d in enumerate(sorted(dets, key=lambda x: x['bbox'][0])):
    fb = d['bbox']
    cx = (fb[0] + fb[2]) / 2.0
    cy = (fb[1] + fb[3]) / 2.0
    pb = d.get('person_bbox', [0,0,0,0])
    tid = d.get('id')
    print(f"Det {i+1}: tid={tid} head=({cx:.1f}, {cy:.1f}) pb_bot={pb[3]}")
