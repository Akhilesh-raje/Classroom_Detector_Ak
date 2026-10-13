import os
import sys
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from core.detector import detect_faces_and_phones

def analyze(vname, n_frames=60):
    video_path = os.path.join(ROOT, "test video", vname)
    cap = cv2.VideoCapture(video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracks_acc = {}
    frame_tids = []

    for f_idx in range(n_frames):
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
            pw = pb[2] - pb[0]
            ph = pb[3] - pb[1]
            hx = (fb[0] + fb[2]) / 2.0
            hy = (fb[1] + fb[3]) / 2.0
            by = float(pb[3])
            if tid not in tracks_acc:
                tracks_acc[tid] = {"count": 0, "hx": [], "hy": [], "by": [], "pw": [], "ph": []}
            tracks_acc[tid]["count"] += 1
            tracks_acc[tid]["hx"].append(hx)
            tracks_acc[tid]["hy"].append(hy)
            tracks_acc[tid]["by"].append(by)
            tracks_acc[tid]["pw"].append(pw)
            tracks_acc[tid]["ph"].append(ph)
        frame_tids.append(tids_in_frame)

    cap.release()

    min_obs = max(3, n_frames // 8)
    stable = []
    for tid, d in tracks_acc.items():
        if d["count"] >= min_obs:
            stable.append({
                "tid": tid,
                "count": d["count"],
                "hx": float(np.median(d["hx"])),
                "hy": float(np.median(d["hy"])),
                "by": float(np.median(d["by"])),
                "pw": float(np.median(d["pw"])),
                "ph": float(np.median(d["ph"])),
                "y_feat": float(0.35 * np.median(d["hy"]) + 0.65 * np.median(d["by"]))
            })

    # Co-occurrence based dedup
    # If two tracks co-occur in >= 2 frames, they CANNOT be merged.
    # If they NEVER co-occur and d_head < 50px, merge them.
    stable.sort(key=lambda x: x["count"], reverse=True)
    merged = []
    for t in stable:
        is_dup = False
        for m in merged:
            # Check co-occurrence
            co_count = sum(1 for f_set in frame_tids if t["tid"] in f_set and m["tid"] in f_set)
            d_head = np.hypot(t["hx"] - m["hx"], t["hy"] - m["hy"])
            d_y = abs(t["y_feat"] - m["y_feat"])
            if co_count == 0 and d_head < 60.0 and d_y < 50.0:
                is_dup = True
                break
        if not is_dup:
            merged.append(t)

    stable = merged
    print(f"\n========================================\nVideo: {vname}")
    print(f"Total stable distinct occupants: {len(stable)}")
    for s in sorted(stable, key=lambda x: x["hx"]):
        print(f"  TID {s['tid']}: count={s['count']}, head=({s['hx']:.1f}, {s['hy']:.1f}), bot_y={s['by']:.1f}, y_feat={s['y_feat']:.1f}")

    # Evaluate row clustering (1D k-means / Jenks on y_feat)
    y_vals = sorted(s["y_feat"] for s in stable)
    n = len(y_vals)
    print(f"Sorted y_feat: {[round(y, 1) for y in y_vals]}")

    # Test k=1, 2, 3 clustering
    def get_clusters_and_var(k):
        if k == 1:
            return [stable], np.var(y_vals) if n > 0 else 0
        if k == 2:
            best_v_between = -1
            best_split = 1
            tot_var = np.var(y_vals)
            tot_mean = np.mean(y_vals)
            for i in range(1, n):
                g1, g2 = y_vals[:i], y_vals[i:]
                v_between = (len(g1)*(np.mean(g1)-tot_mean)**2 + len(g2)*(np.mean(g2)-tot_mean)**2)/n
                if v_between > best_v_between:
                    best_v_between = v_between
                    best_split = i
            sorted_s = sorted(stable, key=lambda x: x["y_feat"])
            return [sorted_s[:best_split], sorted_s[best_split:]], best_v_between / max(1e-6, tot_var)
        if k == 3:
            if n < 3:
                return [stable], 0
            best_v_between = -1
            best_splits = (1, 2)
            tot_var = np.var(y_vals)
            tot_mean = np.mean(y_vals)
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    g1, g2, g3 = y_vals[:i], y_vals[i:j], y_vals[j:]
                    v_between = (len(g1)*(np.mean(g1)-tot_mean)**2 + len(g2)*(np.mean(g2)-tot_mean)**2 + len(g3)*(np.mean(g3)-tot_mean)**2)/n
                    if v_between > best_v_between:
                        best_v_between = v_between
                        best_splits = (i, j)
            sorted_s = sorted(stable, key=lambda x: x["y_feat"])
            i, j = best_splits
            return [sorted_s[:i], sorted_s[i:j], sorted_s[j:]], best_v_between / max(1e-6, tot_var)

    _, r2 = get_clusters_and_var(2)
    _, r3 = get_clusters_and_var(3)
    print(f"Variance explained: 2 rows = {r2:.3f}, 3 rows = {r3:.3f}")

    # Optimal row count decision:
    # If 3 rows explains > 85% variance AND improves by >= 15% over 2 rows -> 3 rows
    # If 2 rows explains > 55% variance -> 2 rows
    # Otherwise -> 1 row
    if n >= 6 and r3 >= 0.85 and (r3 - r2) >= 0.15:
        chosen_k = 3
    elif r2 >= 0.55:
        chosen_k = 2
    else:
        chosen_k = 1

    rows_grouped, _ = get_clusters_and_var(chosen_k)
    # Order rows: Row 0 = Front (largest Y), Row N-1 = Back (smallest Y)
    rows_grouped.sort(key=lambda r: np.median([t["y_feat"] for t in r]), reverse=True)

    print(f"\nChosen Row Count: {chosen_k}")
    for r_idx, r_tracks in enumerate(rows_grouped):
        r_tracks.sort(key=lambda t: t["hx"])
        r_label = f"R{r_idx + 1}"
        med_y = np.median([t["y_feat"] for t in r_tracks])
        print(f"  Row {r_label} (y_median={med_y:.1f}, {len(r_tracks)} seats):")
        for c_idx, t in enumerate(r_tracks):
            print(f"    Seat {r_label}C{c_idx + 1} (TID {t['tid']}): head=({t['hx']:.1f}, {t['hy']:.1f}), bot_y={t['by']:.1f}")

analyze("Student Sitting in the Class  Royalty Free Video  No Copyright Video_1080p.mp4", 60)
analyze("test video - real classrom.mp4", 60)
