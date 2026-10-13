import cv2, os, sys
sys.path.insert(0, 'backend')
from core.engine import ClassroomEngine
from core.topology import TopologyConfig

video_path = os.path.join('test video', 'test video - real classrom.mp4')
cap = cv2.VideoCapture(video_path)

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

calib_frames = []
for _ in range(5):
    ret, cf = cap.read()
    if not ret: break
    calib_frames.append(cf)

engine = ClassroomEngine(max_seats=15)
engine.calibrate(calib_frames, W, H, config=TopologyConfig(mode="structure_ai"))

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
f_idx = 0

print("Scanning for frames where trk11 is assigned R1C6...")
while f_idx < 900:
    ret, frame = cap.read()
    if not ret: break
    if f_idx % 2 == 0:
        faces = engine.process_single_frame(frame, W_px=W, H_px=H)
        for f in faces:
            if f.get("track_id") == 11 and f.get("seat_label") == "R1C6":
                print(f"Found at frame {f_idx} (t={f_idx/30:.1f}s): trk11 assigned R1C6, bbox={f.get('bbox')}")
                # Print all faces at this frame
                for ff in sorted(faces, key=lambda x: x.get('bbox', [0])[0]):
                    print(f"   tid={ff.get('track_id')} seat={ff.get('seat_label')} bbox={ff.get('bbox')} status={ff.get('status')}")
                cap.release()
                sys.exit(0)
    f_idx += 1

cap.release()
print("Not found in first 900 frames")
