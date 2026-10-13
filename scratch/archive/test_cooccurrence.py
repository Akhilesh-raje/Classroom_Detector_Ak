import os
import sys
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from core.detector import detect_faces_and_phones

def test_video(vname):
    video_path = os.path.join(ROOT, "test video", vname)
    cap = cv2.VideoCapture(video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracks_acc = {}
    frame_tids = []

    for f_idx in range(100):
        ret, frame = cap.read()
        if not ret:
            break
        raw_dets, _, _, _ = detect_faces_and_phones(frame, min_conf=0.25, persist=True)
        tids_in_frame = set()
        for d in raw_dets:
            tid = d.get("id")
            if tid is None:
                continue
            tids_in_frame.add(tid)
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
        frame_tids.append(tids_in_frame)

    cap.release()

    stable = {}
    for tid, d in tracks_acc.items():
        if d["count"] >= 10:
            stable[tid] = {
                "tid": tid,
                "count": d["count"],
                "hx": np.median(d["hx"]),
                "hy": np.median(d["hy"]),
                "by": np.median(d["by"]),
                "y_feat": 0.35 * np.median(d["hy"]) + 0.65 * np.median(d["by"])
            }

    # Co-occurrence
    stable_tids = list(stable.keys())
    cooccur = {t1: {t2: 0 for t2 in stable_tids} for t1 in stable_tids}
    for t_set in frame_tids:
        for t1 in t_set:
            if t1 in stable:
                for t2 in t_set:
                    if t2 in stable:
                        cooccur[t1][t2] += 1

    print(f"\n==================== {vname} ====================")
    print(f"Stable tracks: {len(stable)}")
    for tid, s in stable.items():
        print(f"  TID {tid}: count={s['count']}, head=({s['hx']:.1f}, {s['hy']:.1f}), bot_y={s['by']:.1f}, y_feat={s['y_feat']:.1f}")

    print("\nCo-occurrences and distances:")
    for i in range(len(stable_tids)):
        for j in range(i + 1, len(stable_tids)):
            t1 = stable_tids[i]
            t2 = stable_tids[j]
            s1, s2 = stable[t1], stable[t2]
            co = cooccur[t1][t2]
            d_head = np.hypot(s1["hx"] - s2["hx"], s1["hy"] - s2["hy"])
            d_yfeat = abs(s1["y_feat"] - s2["y_feat"])
            print(f"  TID {t1} & TID {t2}: co-occur={co} frames, d_head={d_head:.1f}px, d_yfeat={d_yfeat:.1f}px")

test_video("Student Sitting in the Class  Royalty Free Video  No Copyright Video_1080p.mp4")
test_video("test video - real classrom.mp4")
