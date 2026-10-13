import os
import sys
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from core.detector import detect_faces_and_phones

def test_feature(vname, w_head=0.15, w_bot=0.85):
    video_path = os.path.join(ROOT, "test video", vname)
    cap = cv2.VideoCapture(video_path)
    tracks_acc = {}
    frame_tids = []

    for f_idx in range(60):
        ret, frame = cap.read()
        if not ret:
            break
        raw_dets, _, _, _ = detect_faces_and_phones(frame, min_conf=0.25, persist=True)
        tids = set()
        for d in raw_dets:
            tid = d.get("id")
            if tid is None:
                continue
            tids.add(tid)
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
        frame_tids.append(tids)
    cap.release()

    stable = []
    for tid, d in tracks_acc.items():
        if d["count"] >= 15:
            hy = float(np.median(d["hy"]))
            by = float(np.median(d["by"]))
            stable.append({
                "tid": tid,
                "count": d["count"],
                "hx": float(np.median(d["hx"])),
                "hy": hy,
                "by": by,
                "y_feat": w_head * hy + w_bot * by
            })

    # Dedup
    stable.sort(key=lambda x: x["count"], reverse=True)
    merged = []
    for t in stable:
        is_dup = False
        for m in merged:
            co = sum(1 for f_set in frame_tids if t["tid"] in f_set and m["tid"] in f_set)
            d_head = np.hypot(t["hx"] - m["hx"], t["hy"] - m["hy"])
            d_y = abs(t["y_feat"] - m["y_feat"])
            if co == 0 and d_head < 60.0 and d_y < 50.0:
                is_dup = True
                break
        if not is_dup:
            merged.append(t)

    stable = merged
    y_vals = sorted(s["y_feat"] for s in stable)
    n = len(y_vals)

    def evaluate_k(k):
        tot_var = np.var(y_vals)
        tot_mean = np.mean(y_vals)
        if k == 1:
            return [stable], 0.0
        if k == 2:
            best_v, best_s = -1, 1
            for i in range(1, n):
                g1, g2 = y_vals[:i], y_vals[i:]
                v = (len(g1)*(np.mean(g1)-tot_mean)**2 + len(g2)*(np.mean(g2)-tot_mean)**2)/n
                if v > best_v:
                    best_v, best_s = v, i
            sorted_s = sorted(stable, key=lambda x: x["y_feat"])
            return [sorted_s[:best_s], sorted_s[best_s:]], best_v / max(1e-6, tot_var)
        if k == 3:
            if n < 3:
                return [stable], 0.0
            best_v, best_splits = -1, (1, 2)
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    g1, g2, g3 = y_vals[:i], y_vals[i:j], y_vals[j:]
                    v = (len(g1)*(np.mean(g1)-tot_mean)**2 + len(g2)*(np.mean(g2)-tot_mean)**2 + len(g3)*(np.mean(g3)-tot_mean)**2)/n
                    if v > best_v:
                        best_v, best_splits = v, (i, j)
            sorted_s = sorted(stable, key=lambda x: x["y_feat"])
            i, j = best_splits
            return [sorted_s[:i], sorted_s[i:j], sorted_s[j:]], best_v / max(1e-6, tot_var)

    _, r2 = evaluate_k(2)
    _, r3 = evaluate_k(3)

    # Universal selection:
    # 3 rows if r3 >= 0.90 and (r3 - r2) >= 0.08, or if r3 >= 0.95
    if n >= 6 and (r3 >= 0.92 and (r3 - r2) >= 0.08):
        k = 3
    elif r2 >= 0.55:
        k = 2
    else:
        k = 1

    clusters, r = evaluate_k(k)
    clusters.sort(key=lambda cl: np.median([t["y_feat"] for t in cl]), reverse=True)

    print(f"\n[{vname}] -> {k} rows (r2={r2:.3f}, r3={r3:.3f}), total seats = {len(stable)}")
    for r_idx, cl in enumerate(clusters):
        cl.sort(key=lambda t: t["hx"])
        row_label = f"R{r_idx + 1}"
        med_y = np.median([t["y_feat"] for t in cl])
        seats_str = ", ".join([f"C{c_idx+1}:TID{t['tid']} (x={t['hx']:.0f})" for c_idx, t in enumerate(cl)])
        print(f"  Row {row_label} (y_med={med_y:.1f}, count={len(cl)}): {seats_str}")

test_feature("Student Sitting in the Class  Royalty Free Video  No Copyright Video_1080p.mp4", 0.15, 0.85)
test_feature("test video - real classrom.mp4", 0.15, 0.85)
