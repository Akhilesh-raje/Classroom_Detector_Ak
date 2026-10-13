import sys, os, statistics
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend"))
import core.config as cfg; cfg.YOLO_CONF = 0.22
import cv2
from core.detector import detect_faces_and_phones

ROOT  = os.path.dirname(os.path.dirname(__file__))
VIDEO = os.path.join(ROOT, "test video", "test video - real classrom.mp4")

cap = cv2.VideoCapture(VIDEO); W=int(cap.get(3)); H=int(cap.get(4)); fps=cap.get(5)
person_bboxes, face_bboxes = [], []
for fi in [int(i*fps*5) for i in range(20)]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi); ret,f=cap.read()
    if not ret: continue
    faces,_,_,_ = detect_faces_and_phones(f)
    for d in faces:
        person_bboxes.append(d["person_bbox"])
        face_bboxes.append(d["bbox"])
cap.release()

dedup_p, dedup_f = [], []
for i,bb in enumerate(person_bboxes):
    cx=(bb[0]+bb[2])/2; cy=(bb[1]+bb[3])/2
    if not any(abs(cx-(e[0]+e[2])/2)<W*0.05 and abs(cy-(e[1]+e[3])/2)<H*0.05 for e in dedup_p):
        dedup_p.append(bb); dedup_f.append(face_bboxes[i])

sorted_idx = sorted(range(len(dedup_p)), key=lambda i: dedup_p[i][3])
bots = [dedup_p[i][3] for i in sorted_idx]
gaps = [bots[i+1]-bots[i] for i in range(len(bots)-1)]
MIN_PER_ROW = max(2, len(dedup_p)//4)
print(f"N={len(dedup_p)}  MIN_PER_ROW={MIN_PER_ROW}")
print(f"\nAll gaps (top 10):")
for gi in sorted(range(len(gaps)), key=lambda i: gaps[i], reverse=True)[:10]:
    l, r = gi+1, len(bots)-(gi+1)
    ok = "OK" if l>=MIN_PER_ROW and r>=MIN_PER_ROW else f"SKIP(l={l},r={r})"
    print(f"  gap[{gi}]={gaps[gi]:.0f}px  bot={bots[gi]:.0f}->{bots[gi+1]:.0f}  split:{l}+{r}  {ok}")

for gi in sorted(range(len(gaps)), key=lambda i: gaps[i], reverse=True):
    l,r=gi+1,len(bots)-(gi+1)
    if l>=MIN_PER_ROW and r>=MIN_PER_ROW:
        print(f"\nChosen split: gap[{gi}]={gaps[gi]:.0f}px  bot={bots[gi]:.0f}->{bots[gi+1]:.0f}")
        r0=[sorted_idx[i] for i in range(0,gi+1)]
        r1=[sorted_idx[i] for i in range(gi+1,len(sorted_idx))]
        fc0=[(dedup_f[i][1]+dedup_f[i][3])/2 for i in r0]
        fc1=[(dedup_f[i][1]+dedup_f[i][3])/2 for i in r1]
        print(f"Row 0: {len(r0)} persons  face_cy median={statistics.median(fc0):.0f}")
        print(f"Row 1: {len(r1)} persons  face_cy median={statistics.median(fc1):.0f}")
        break
