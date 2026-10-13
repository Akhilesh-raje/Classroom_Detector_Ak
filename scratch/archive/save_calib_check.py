import os
import sys
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
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

# Draw on first frame
ref_frame = calib_frames[0].copy()
topo = engine.topology

colors = {0: (0, 255, 100), 1: (255, 180, 0)}  # R1 green, R2 cyan/blue

for s in topo.seats:
    x1, y1, x2, y2 = [int(v) for v in s.bbox]
    ax, ay = int(s.anchor[0]), int(s.anchor[1])
    col = colors.get(s.row, (255, 255, 255))
    
    cv2.rectangle(ref_frame, (x1, y1), (x2, y2), col, 2)
    cv2.circle(ref_frame, (ax, ay), 5, (0, 0, 255), -1)
    
    badge = f"{s.label}"
    cv2.putText(ref_frame, badge, (x1 + 4, y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
    cv2.putText(ref_frame, badge, (x1 + 4, y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)

# Draw row division line
if len(topo.row_geometries) >= 2:
    mid_y = int((topo.row_geometries[0].y_median + topo.row_geometries[1].y_median) / 2)
    cv2.line(ref_frame, (0, mid_y), (W, mid_y), (0, 0, 255), 2)
    cv2.putText(ref_frame, f"Row Split Y={mid_y}", (20, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

out_path = os.path.join(HERE, "calib_12_seats.jpg")
cv2.imwrite(out_path, ref_frame)
print(f"Saved visual verification to: {out_path}")
cap.release()
