"""Check exactly what head_y values build_topology computes vs ground truth."""
import sys, os, statistics
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend"))
import core.config as cfg; cfg.YOLO_CONF = 0.22
import cv2
from core.detector import detect_faces_and_phones

ROOT  = os.path.dirname(os.path.dirname(__file__))
VIDEO = os.path.join(ROOT, "test video", "test video - real classrom.mp4")

cap = cv2.VideoCapture(VIDEO); W=int(cap.get(3)); H=int(cap.get(4)); fps=cap.get(5)

# Same 20 frames as calibration
person_bboxes, face_bboxes = [], []
for fi in [int(i*fps*5) for i in range(20)]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi); ret,f=cap.read()
    if not ret: continue
    faces,_,_,_ = detect_faces_and_phones(f)
    for d in faces:
        person_bboxes.append(d["person_bbox"])
        face_bboxes.append(d["bbox"])
cap.release()

# Dedup
dedup_p, dedup_f = [], []
for i,bb in enumerate(person_bboxes):
    cx=(bb[0]+bb[2])/2; cy=(bb[1]+bb[3])/2
    if not any(abs(cx-(e[0]+e[2])/2)<W*0.05 and abs(cy-(e[1]+e[3])/2)<H*0.05 for e in dedup_p):
        dedup_p.append(bb); dedup_f.append(face_bboxes[i])

print(f"After dedup: {len(dedup_p)} persons")

# Split by largest person_bottom gap
sorted_idx = sorted(range(len(dedup_p)), key=lambda i: dedup_p[i][3])
bots = [dedup_p[i][3] for i in sorted_idx]
gaps = [bots[i+1]-bots[i] for i in range(len(bots)-1)]
best_gi = max(range(len(gaps)), key=lambda i: gaps[i])
print(f"Largest gap: {gaps[best_gi]}px at idx={best_gi} (between bottom={bots[best_gi]:.0f} and {bots[best_gi+1]:.0f})")

row0_idx = [sorted_idx[i] for i in range(0, best_gi+1)]
row1_idx = [sorted_idx[i] for i in range(best_gi+1, len(sorted_idx))]

print(f"\nRow 0 ({len(row0_idx)} persons):")
print(f"  person_bottoms: {[dedup_p[i][3] for i in row0_idx]}")
face_cys_0 = [(dedup_f[i][1]+dedup_f[i][3])/2 for i in row0_idx]
print(f"  face_cys:       {[round(c) for c in face_cys_0]}")
print(f"  head_y (median): {statistics.median(face_cys_0):.1f}")

print(f"\nRow 1 ({len(row1_idx)} persons):")
print(f"  person_bottoms: {[dedup_p[i][3] for i in row1_idx]}")
face_cys_1 = [(dedup_f[i][1]+dedup_f[i][3])/2 for i in row1_idx]
print(f"  face_cys:       {[round(c) for c in face_cys_1]}")
print(f"  head_y (median): {statistics.median(face_cys_1):.1f}")

print(f"\nGround truth (single frame):")
print(f"  Back-row face cy:  470-508, median=480")
print(f"  Front-row face cy: 513-598, median=555")
