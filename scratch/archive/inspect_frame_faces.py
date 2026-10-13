import cv2, os, sys
sys.path.insert(0, 'backend')
from core.detector import detect_faces_and_phones, YOLO_CONF

video_path = os.path.join('test video', 'test video - real classrom.mp4')
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 840)
ret, frame = cap.read()
cap.release()

if ret:
    dets, _, _, _ = detect_faces_and_phones(frame, min_conf=0.25, persist=False)
    print(f"Total detections at frame 840: {len(dets)}")
    for i, d in enumerate(sorted(dets, key=lambda x: x['bbox'][0])):
        fb = d['bbox']
        cx = (fb[0] + fb[2]) / 2.0
        cy = (fb[1] + fb[3]) / 2.0
        pb = d.get('person_bbox', [0, 0, 0, 0])
        print(f"Det {i+1:2d}: head=({cx:6.1f}, {cy:6.1f}) face_w={fb[2]-fb[0]:3d} face_h={fb[3]-fb[1]:3d} person_bot={pb[3]:4d}")
