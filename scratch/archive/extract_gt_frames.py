import os
import sys
import json
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "backend"))

from core.engine import ClassroomEngine

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

engine = ClassroomEngine()
engine.calibrate(calib_frames, W, H)

# Sample frames at 30, 90, 150, 210, 270
sample_indices = [30, 90, 150, 210, 270]
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

gt_data = {
    "video": os.path.basename(video_path),
    "resolution": [W, H],
    "total_physical_seats": 12,
    "rows": {
        "R1": {"semantic": "FRONT", "count": 5, "seats": ["R1C1", "R1C2", "R1C3", "R1C4", "R1C5"]},
        "R2": {"semantic": "BACK", "count": 7, "seats": ["R2C1", "R2C2", "R2C3", "R2C4", "R2C5", "R2C6", "R2C7"]},
    },
    "frames": {}
}

current_f = 0
while current_f <= max(sample_indices):
    ret, frame = cap.read()
    if not ret:
        break
    faces = engine.process_single_frame(frame, W, H)
    if current_f in sample_indices:
        frame_preds = []
        for face in faces:
            frame_preds.append({
                "assigned_seat": face.get("seat_label"),
                "status": face.get("status"),
                "bbox": face.get("bbox"),
                "geometric_match": round(face.get("geometric_match", 1.0), 3),
            })
        gt_data["frames"][str(current_f)] = frame_preds
    current_f += 1

cap.release()

gt_path = os.path.join(ROOT, "backend", "tests", "ground_truth_classroom.json")
os.makedirs(os.path.dirname(gt_path), exist_ok=True)
with open(gt_path, "w", encoding="utf-8") as f:
    json.dump(gt_data, f, indent=2)

print(f"Exported ground truth reference with {len(sample_indices)} frames to {gt_path}")
