"""
Single-frame ground truth:
  - What face centers does runtime detection produce?
  - What anchor centers does calibration produce?
  - Are they in the right row bands?
"""
import sys, os, cv2, statistics
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend"))
import core.config as cfg; cfg.YOLO_CONF = 0.22

ROOT  = os.path.dirname(os.path.dirname(__file__))
VIDEO = os.path.join(ROOT, "test video", "test video - real classrom.mp4")

cap = cv2.VideoCapture(VIDEO)
W = int(cap.get(3)); H = int(cap.get(4))
cap.set(cv2.CAP_PROP_POS_FRAMES, 5)
ret, frame = cap.read(); cap.release()

from core.detector import detect_faces_and_phones, track_persons

# --- Runtime face-crop centers ---
faces, _, _, _ = detect_faces_and_phones(frame)
print("RUNTIME face-crop centers (sorted by cy):")
face_data = sorted(
    [{"cx": (f["bbox"][0]+f["bbox"][2])//2,
      "cy": (f["bbox"][1]+f["bbox"][3])//2,
      "pbottom": f["person_bbox"][3]} for f in faces],
    key=lambda x: x["cy"]
)
for fd in face_data:
    print(f"  cy={fd['cy']:4d}  cx={fd['cx']:4d}  person_bottom={fd['pbottom']}")

cys = [fd["cy"] for fd in face_data]
pbots = [fd["pbottom"] for fd in face_data]

# --- Person bbox bottoms for row split ---
print("\nPerson bbox bottoms (sorted):", sorted(pbots))
gaps = sorted([abs(pbots[i+1]-pbots[i]) for i in range(len(pbots)-1)], reverse=True)
print(f"Top-3 gaps in person bottoms: {gaps[:3]}")

# --- What calibration produces as head_y ---
persons = track_persons(frame, persist=False)
persons_sorted = sorted(persons, key=lambda d: (d["bbox"][1]+d["bbox"][3])/2)
print("\nCalibration person bboxes -> head_y at 0.20 fraction:")
for p in persons_sorted:
    y1,y2 = p["bbox"][1], p["bbox"][3]
    head_y_est = y1 + (y2-y1)*0.20
    print(f"  person_bottom={y2:4d}  head_y@0.20={head_y_est:5.0f}")

print(f"\nSUMMARY:")
print(f"  Actual back-row face cy range:  {min(cys[:8])}-{max(cys[:8])}")
print(f"  Actual front-row face cy range: {min(cys[8:])}-{max(cys[8:])}")
back_head = statistics.median(cys[:8])
front_head = statistics.median(cys[8:])
print(f"  Back-row median face cy:  {back_head:.0f}")
print(f"  Front-row median face cy: {front_head:.0f}")
print(f"  Gap between row anchors needed: {front_head - back_head:.0f}px")
print(f"  Y-band half-gap: {(front_head - back_head)/2:.0f}px")
