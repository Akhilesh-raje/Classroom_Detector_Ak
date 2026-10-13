import cv2, sys
sys.path.insert(0, 'backend')
from core.detector import detect_faces_and_phones

cap = cv2.VideoCapture('test video/test video - real classrom.mp4')
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video size: {W}x{H}")

frames_tracks = []
for f_idx in range(60):
    ret, frame = cap.read()
    if not ret: break
    raw_dets, _, _, _ = detect_faces_and_phones(frame, min_conf=0.25, persist=True)
    frames_tracks.append((f_idx, raw_dets))

cap.release()

# Look at all tracks seen in first 60 frames
track_stats = {}
for f_idx, dets in frames_tracks:
    for d in dets:
        tid = d.get("id")
        if tid is None: continue
        pb = d["person_bbox"]
        fb = d["bbox"]
        cx_p = (pb[0] + pb[2]) / 2.0
        cy_p = (pb[1] + pb[3]) / 2.0
        cx_h = (fb[0] + fb[2]) / 2.0
        cy_h = (fb[1] + fb[3]) / 2.0
        bot_y = float(pb[3])
        if tid not in track_stats:
            track_stats[tid] = {"frames": 0, "cx_h": [], "cy_h": [], "bot_y": [], "pb_h": []}
        track_stats[tid]["frames"] += 1
        track_stats[tid]["cx_h"].append(cx_h)
        track_stats[tid]["cy_h"].append(cy_h)
        track_stats[tid]["bot_y"].append(bot_y)
        track_stats[tid]["pb_h"].append(pb[3] - pb[1])

print(f"\nTotal unique tracks in first 60 frames: {len(track_stats)}")
# Filter tracks seen at least 20 frames
stable_tracks = {tid: s for tid, s in track_stats.items() if s["frames"] >= 20}
print(f"Stable tracks (seen >= 20 frames): {len(stable_tracks)}")

import numpy as np
for tid, s in sorted(stable_tracks.items(), key=lambda x: np.median(x[1]["cy_h"])):
    med_cx = np.median(s["cx_h"])
    med_cy = np.median(s["cy_h"])
    med_bot = np.median(s["bot_y"])
    med_h = np.median(s["pb_h"])
    print(f"Track {tid:2d}: frames={s['frames']:2d} head_cx={med_cx:6.1f} head_cy={med_cy:6.1f} bot_y={med_bot:6.1f} bbox_h={med_h:6.1f}")
