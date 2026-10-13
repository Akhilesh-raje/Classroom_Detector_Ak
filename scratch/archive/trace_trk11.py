import cv2, os, sys
sys.path.insert(0, 'backend')
from core.detector import detect_faces_and_phones, YOLO_CONF

video_path = os.path.join('test video', 'test video - real classrom.mp4')
cap = cv2.VideoCapture(video_path)

# Let's inspect frames 0, 100, 200, 300, 400, 500, 600, 700, 800, 840
print("Tracking trk11 across video...")
for f_idx in range(0, 850, 30):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
    ret, frame = cap.read()
    if not ret: break
    dets, _, _, _ = detect_faces_and_phones(frame, min_conf=0.25, persist=True)
    for d in dets:
        if d.get("id") == 11 or d.get("track_id") == 11:
            fb = d["bbox"]
            cx = (fb[0] + fb[2]) / 2.0
            cy = (fb[1] + fb[3]) / 2.0
            pb = d.get("person_bbox", [0,0,0,0])
            print(f"Frame {f_idx:4d}: trk11 head=({cx:6.1f}, {cy:6.1f}) pb_bot={pb[3]}")
cap.release()
