import os
import sys
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from core.detector import detect_faces_and_phones

def extract_tracks(vname, n_frames=60):
    video_path = os.path.join(ROOT, "test video", vname)
    cap = cv2.VideoCapture(video_path)
    tracks_acc = {}
    frame_tids = []
    for _ in range(n_frames):
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
            pw = pb[2] - pb[0]
            ph = pb[3] - pb[1]
            if tid not in tracks_acc:
                tracks_acc[tid] = {"count": 0, "hx": [], "hy": [], "by": [], "pw": [], "ph": []}
            tracks_acc[tid]["count"] += 1
            tracks_acc[tid]["hx"].append(hx)
            tracks_acc[tid]["hy"].append(hy)
            tracks_acc[tid]["by"].append(by)
            tracks_acc[tid]["pw"].append(pw)
            tracks_acc[tid]["ph"].append(ph)
        frame_tids.append(tids)
    cap.release()

    stable = []
    for tid, d in tracks_acc.items():
        if d["count"] >= 15:
            hx = float(np.median(d["hx"]))
            hy = float(np.median(d["hy"]))
            by = float(np.median(d["by"]))
            pw = float(np.median(d["pw"]))
            ph = float(np.median(d["ph"]))
            stable.append({
                "tid": tid,
                "count": d["count"],
                "hx": hx,
                "hy": hy,
                "by": by,
                "pw": pw,
                "ph": ph,
                "y_feat": 0.20 * hy + 0.80 * by
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
    return merged

def test_model_selection(vname):
    tracks = extract_tracks(vname, 60)
    print(f"\n=======================================================")
    print(f"Video: {vname}")
    print(f"Detected stable occupants: {len(tracks)}")
    
    # Analyze optimal slant and clustering
    thetas = np.linspace(-np.radians(15), np.radians(15), 31)
    
    for k in [1, 2, 3]:
        best_score = -1e9
        best_theta = 0
        best_wss = 1e9
        best_bic = 1e9
        best_groups = []
        
        n = len(tracks)
        for theta in thetas:
            proj_vals = [t["y_feat"] * np.cos(theta) - t["hx"] * np.sin(theta) for t in tracks]
            s_idx = np.argsort(proj_vals)
            s_proj = [proj_vals[i] for i in s_idx]
            sorted_tracks = [tracks[i] for i in s_idx]
            tot_var = float(np.var(s_proj))
            tot_mean = float(np.mean(s_proj))
            
            if k == 1:
                wss = tot_var * n
                groups = [sorted_tracks]
            elif k == 2:
                best_v = -1
                best_split = 1
                for i in range(1, n):
                    g1, g2 = s_proj[:i], s_proj[i:]
                    v = (len(g1)*(float(np.mean(g1))-tot_mean)**2 + len(g2)*(float(np.mean(g2))-tot_mean)**2)/n
                    if v > best_v:
                        best_v = v
                        best_split = i
                wss = (tot_var - best_v) * n
                groups = [sorted_tracks[:best_split], sorted_tracks[best_split:]]
            elif k == 3:
                best_v = -1
                best_splits = (1, 2)
                for i in range(1, n - 1):
                    for j in range(i + 1, n):
                        g1, g2, g3 = s_proj[:i], s_proj[i:j], s_proj[j:]
                        v = (len(g1)*(float(np.mean(g1))-tot_mean)**2 + len(g2)*(float(np.mean(g2))-tot_mean)**2 + len(g3)*(float(np.mean(g3))-tot_mean)**2)/n
                        if v > best_v:
                            best_v = v
                            best_splits = (i, j)
                wss = (tot_var - best_v) * n
                i, j = best_splits
                groups = [sorted_tracks[:i], sorted_tracks[i:j], sorted_tracks[j:]]
            
            # BIC: n * ln(WSS/n) + k * ln(n) * penalty
            variance = max(1e-4, wss / n)
            bic = n * np.log(variance) + (k * 2) * np.log(n)
            
            # Distance between row centroids in pixels
            row_meds = [np.median([t["y_feat"] for t in g]) for g in groups]
            row_meds.sort()
            min_row_dist = min([row_meds[idx+1] - row_meds[idx] for idx in range(len(row_meds)-1)]) if len(row_meds) > 1 else 999.0
            
            if -bic > best_score:
                best_score = -bic
                best_theta = theta
                best_wss = wss
                best_bic = bic
                best_groups = groups
                best_min_dist = min_row_dist
        
        print(f"  K={k}: WSS={best_wss:.1f}, BIC={best_bic:.2f}, min_row_gap={best_min_dist:.1f}px, groups={[len(g) for g in best_groups]}")

test_model_selection("Student Sitting in the Class  Royalty Free Video  No Copyright Video_1080p.mp4")
test_model_selection("test video - real classrom.mp4")
