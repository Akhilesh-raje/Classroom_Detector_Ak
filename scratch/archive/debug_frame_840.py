import cv2, os, sys
sys.path.insert(0, 'backend')
from core.engine import ClassroomEngine
from core.topology import TopologyConfig

video_path = os.path.join('test video', 'test video - real classrom.mp4')
cap = cv2.VideoCapture(video_path)

# Calibrate with 5 frames
calib_frames = []
for _ in range(5):
    ret, cf = cap.read()
    if not ret: break
    calib_frames.append(cf)

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
engine = ClassroomEngine(max_seats=15)
engine.calibrate(calib_frames, W, H, config=TopologyConfig(mode="structure_ai"))

# Now read frame 840
cap.set(cv2.CAP_PROP_POS_FRAMES, 840)
ret, frame = cap.read()
cap.release()

faces = engine.process_single_frame(frame, W, H)

print("\n--- RESULTS AT FRAME 840 ---")
for f in sorted(faces, key=lambda x: x.get('bbox', [0])[0]):
    fb = f['bbox']
    cx = (fb[0] + fb[2]) / 2.0
    cy = (fb[1] + fb[3]) / 2.0
    pb = f.get('person_bbox', [0,0,0,0])
    lbl = f.get('seat_label')
    tid = f.get('track_id')
    status = f.get('status')
    conf = f.get('seat_confidence')
    print(f"tid={tid:<4} head=({cx:6.1f}, {cy:6.1f}) pb_y2={pb[3]:4d} -> seat={lbl:<6} status={status} conf={conf}")
