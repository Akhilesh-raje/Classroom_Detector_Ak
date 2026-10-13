import cv2, sys, os
sys.path.insert(0, 'backend')
from core.topology import TopologyDiscovery, TopologyConfig
from core.detector import detect_faces_and_phones
import numpy as np

cap = cv2.VideoCapture('test video/test video - real classrom.mp4')
discovery = TopologyDiscovery(frame_size=(1920, 1080))
for f in range(30):
    ret, frame = cap.read()
    if not ret: break
    raw_dets, _, _, _ = detect_faces_and_phones(frame, min_conf=0.25, persist=True)
    discovery.accumulate_raw(raw_dets, frame_idx=f)

cap.release()

stable = []
for tid, d in discovery._tracks.items():
    if d['count'] >= 5:
        hy = float(np.median(d['head_y']))
        by = float(np.median(d['bot_y']))
        hx = float(np.median(d['head_x']))
        y_feat = 0.4*hy + 0.6*by
        stable.append({"tid": tid, "hx": hx, "hy": hy, "by": by, "y_feat": y_feat, "count": d["count"]})

stable.sort(key=lambda x: x["y_feat"])
print(f"Total stable tracks: {len(stable)}")
for i, s in enumerate(stable):
    print(f"[{i:2d}] TID {s['tid']:2d}: count={s['count']:2d} hx={s['hx']:6.1f} hy={s['hy']:6.1f} by={s['by']:6.1f} y_feat={s['y_feat']:6.1f}")

sorted_y = [s["y_feat"] for s in stable]
gaps = [sorted_y[i+1] - sorted_y[i] for i in range(len(sorted_y)-1)]
print("\nGaps between consecutive sorted y_feat:")
for i, g in enumerate(gaps):
    print(f"  Gap {i} (between {stable[i]['tid']} and {stable[i+1]['tid']}): {g:6.2f} px")

print(f"Max gap: {max(gaps):.2f}, Mean gap: {np.mean(gaps):.2f}, Ratio: {max(gaps)/np.mean(gaps):.2f}")
