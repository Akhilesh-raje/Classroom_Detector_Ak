import os
import sys
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "backend"))

from test_model_selection import extract_tracks

def synthesize_classroom_topology(vname):
    tracks = extract_tracks(vname, 60)
    n = len(tracks)
    thetas = np.linspace(-np.radians(15), np.radians(15), 31)

    # 1. Determine optimal row count using min_row_gap constraint (physical classroom rows > 65px apart)
    candidates = {}
    for k in [1, 2, 3]:
        best_bic = 1e9
        best_theta = 0
        best_groups = []
        best_min_dist = 0
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

            variance = max(1e-4, wss / n)
            bic = n * np.log(variance) + (k * 2) * np.log(n)
            row_meds = sorted([np.median([t["y_feat"] for t in g]) for g in groups])
            min_gap = min([row_meds[idx+1] - row_meds[idx] for idx in range(len(row_meds)-1)]) if len(row_meds) > 1 else 999.0

            if bic < best_bic:
                best_bic = bic
                best_theta = theta
                best_groups = groups
                best_min_dist = min_gap
        candidates[k] = (best_bic, best_theta, best_groups, best_min_dist)

    # Choose K:
    # A candidate is valid only if min_row_gap >= 65px (real physical row distance)
    # Among valid candidates, pick the one with lowest BIC
    valid_ks = [k for k in [1, 2, 3] if candidates[k][3] >= 65.0]
    chosen_k = min(valid_ks, key=lambda k: candidates[k][0]) if valid_ks else 1
    _, opt_theta, rows_grouped, _ = candidates[chosen_k]

    # Order rows Front (largest Y) to Back
    rows_grouped.sort(key=lambda g: np.median([t["y_feat"] for t in g]), reverse=True)

    # Within each row, merge any tracks that represent the exact same seat occupant (dx < 90px)
    deduped_rows = []
    for g in rows_grouped:
        g.sort(key=lambda t: t["hx"])
        merged_g = []
        for t in g:
            if not merged_g or abs(t["hx"] - merged_g[-1]["hx"]) >= 90.0:
                merged_g.append(t)
            else:
                # Merge into previous occupant
                if t["count"] > merged_g[-1]["count"]:
                    merged_g[-1] = t
        deduped_rows.append(merged_g)
    rows_grouped = deduped_rows

    # 2. Bench & Seat Grid Synthesis
    # Determine maximum columns across rows
    max_observed_cols = max(len(g) for g in rows_grouped)
    target_cols = max_observed_cols
    all_xs = [t["hx"] for t in tracks]
    room_x_min = min(all_xs)
    room_x_max = max(all_xs)

    print(f"\n=======================================================")
    print(f"[{vname}]")
    print(f"Optimal Rows: {chosen_k} (Slant theta = {np.degrees(opt_theta):.1f} deg)")
    print(f"Observed counts per row: {[len(g) for g in rows_grouped]}")
    print(f"Target physical seats per row: {target_cols}")

    # Generate physical seat positions per row
    synthesized_seats = []
    for r_idx, g in enumerate(rows_grouped):
        row_label = f"R{r_idx + 1}"
        med_y = np.median([t["y_feat"] for t in g])
        g_xs = sorted(t["hx"] for t in g)

        if len(g) == target_cols:
            # All seats occupied
            seat_xs = g_xs
        else:
            # Some seats unoccupied on this bench! Synthesize full physical bench grid across room span
            seat_xs = np.linspace(room_x_min, room_x_max, target_cols).tolist()

        print(f"  Row {row_label} ({len(seat_xs)} physical seats, y={med_y:.1f}):")
        for c_idx, sx in enumerate(seat_xs):
            lbl = f"{row_label}C{c_idx + 1}"
            print(f"    Seat {lbl}: anchor=({sx:.1f}, {med_y:.1f})")

synthesize_classroom_topology("test video - real classrom.mp4")
synthesize_classroom_topology("Student Sitting in the Class  Royalty Free Video  No Copyright Video_1080p.mp4")
