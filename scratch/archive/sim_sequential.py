import cv2, os, sys
sys.path.insert(0, 'backend')
from core.engine import ClassroomEngine
from core.topology import TopologyConfig

video_path = os.path.join('test video', 'test video - real classrom.mp4')
cap = cv2.VideoCapture(video_path)

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Calibrate with 5 frames
calib_frames = []
for _ in range(5):
    ret, cf = cap.read()
    if not ret: break
    calib_frames.append(cf)

engine = ClassroomEngine(max_seats=15)
engine.calibrate(calib_frames, W, H, config=TopologyConfig(mode="structure_ai"))

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
f_idx = 0

print("Simulating test_real_classroom.py sequentially up to frame 850...")
while f_idx <= 850:
    ret, frame = cap.read()
    if not ret: break
    if f_idx % 2 == 0:  # AI_EVERY = 2
        faces = engine.process_single_frame(frame, W_px=W, H_px=H)
        if f_idx in (0, 100, 200, 300, 400, 500, 600, 700, 800, 840):
            # Print assignments for seats R1C6, R2C5, and track 11
            t11 = next((f for f in faces if f.get("track_id") == 11), None)
            s_r1c6 = next((f for f in faces if f.get("seat_label") == "R1C6"), None)
            s_r2c5 = next((f for f in faces if f.get("seat_label") == "R2C5"), None)
            
            t11_str = f"tid=11 -> {t11.get('seat_label')} (status={t11.get('status')})" if t11 else "tid=11 not found"
            r1c6_str = f"R1C6: tid={s_r1c6.get('track_id')} head={s_r1c6.get('bbox')[:2]}" if s_r1c6 else "R1C6: vacant"
            r2c5_str = f"R2C5: tid={s_r2c5.get('track_id')} head={s_r2c5.get('bbox')[:2]}" if s_r2c5 else "R2C5: vacant"
            print(f"Frame {f_idx:4d} | {t11_str:<30} | {r1c6_str:<32} | {r2c5_str}")
    f_idx += 1

cap.release()
