import os
import sys
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from core.detector import detect_faces_and_phones

video_path = os.path.join(ROOT, "test video", "Student Sitting in the Class  Royalty Free Video  No Copyright Video_1080p.mp4")
cap = cv2.VideoCapture(video_path)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

tracks_acc = {}

for f_idx in range(100):
    ret, frame = cap.read()
    if not ret:
        break
    raw_dets, _, _, _ = detect_faces_and_phones(frame, min_conf=0.25, persist=True)
    for d in raw_dets:
        tid = d.get("id")
        if tid is None:
            continue
        pb = d.get("person_bbox", d.get("bbox"))
        fb = d.get("bbox", pb)
        hx = (fb[0] + fb[2]) / 2.0
        hy = (fb[1] + fb[3]) / 2.0
        by = float(pb[3])
        if tid not in tracks_acc:
            tracks_acc[tid] = {"count": 0, "hx": [], "hy": [], "by": []}
        tracks_acc[tid]["count"] += 1
        tracks_acc[tid]["hx"].append(hx)
        tracks_acc[tid]["hy"].append(hy)
        tracks_acc[tid]["by"].append(by)

cap.release()

print(f"Total raw tracks: {len(tracks_acc)}")
stable = []
for tid, d in tracks_acc.items():
    if d["count"] >= 10:
        stable.append({
            "tid": tid,
            "count": d["count"],
            "hx": np.median(d["hx"]),
            "hy": np.median(d["hy"]),
            "by": np.median(d["by"]),
            "y_feat": 0.35 * np.median(d["hy"]) + 0.65 * np.median(d["by"])
        })

stable.sort(key=lambda x: x["hx"])
print(f"Stable tracks (count >= 10): {len(stable)}")
for s in stable:
    print(f"  TID {s['tid']}: count={s['count']}, head=({s['hx']:.1f}, {s['hy']:.1f}), bot_y={s['by']:.1f}, y_feat={s['y_feat']:.1f}")

# Pairwise distances
print("\nPairwise distances between stable tracks:")
for i in range(len(stable)):
    for j in range(i + 1, len(stable)):
        t1, t2 = stable[i], stable[j]
        d_head = np.hypot(t1["hx"] - t2["hx"], t1["hy"] - t2["hy"])
        d_yfeat = abs(t1["y_feat"] - t2["y_feat"])
        print(f"  TID {t1['tid']} vs TID {t2['tid']}: d_head={d_head:.1f}px, d_yfeat={d_yfeat:.1f}px, dx={abs(t1['hx']-t2['hx']):.1f}px")
