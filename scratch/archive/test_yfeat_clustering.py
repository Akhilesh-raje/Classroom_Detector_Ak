import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "backend"))

# The 12 tracks we extracted:
tracks = [
    {"tid": 6,  "head_x": 298.5,  "head_y": 496.2, "bot_y": 665.5, "bbox_w": 256, "bbox_h": 200},
    {"tid": 5,  "head_x": 421.0,  "head_y": 513.5, "bot_y": 698.0, "bbox_w": 273, "bbox_h": 224},
    {"tid": 10, "head_x": 599.0,  "head_y": 483.0, "bot_y": 591.0, "bbox_w": 103, "bbox_h": 124},
    {"tid": 8,  "head_x": 609.0,  "head_y": 557.5, "bot_y": 736.5, "bbox_w": 293, "bbox_h": 225},
    {"tid": 9,  "head_x": 729.0,  "head_y": 501.0, "bot_y": 631.5, "bbox_w": 152, "bbox_h": 157},
    {"tid": 4,  "head_x": 837.5,  "head_y": 588.0, "bot_y": 787.0, "bbox_w": 394, "bbox_h": 239},
    {"tid": 19, "head_x": 950.0,  "head_y": 518.0, "bot_y": 651.0, "bbox_w": 135, "bbox_h": 162},
    {"tid": 3,  "head_x": 1069.5, "head_y": 576.2, "bot_y": 808.5, "bbox_w": 422, "bbox_h": 283},
    {"tid": 7,  "head_x": 1184.0, "head_y": 524.0, "bot_y": 672.5, "bbox_w": 141, "bbox_h": 177},
    {"tid": 2,  "head_x": 1391.0, "head_y": 574.2, "bot_y": 862.0, "bbox_w": 399, "bbox_h": 353},
    {"tid": 11, "head_x": 1415.0, "head_y": 490.0, "bot_y": 670.0, "bbox_w": 245, "bbox_h": 219},
    {"tid": 1,  "head_x": 1681.5, "head_y": 533.5, "bot_y": 741.0, "bbox_w": 363, "bbox_h": 251},
]

for t in tracks:
    t["y_feat"] = 0.35 * t["head_y"] + 0.65 * t["bot_y"]

sorted_tracks = sorted(tracks, key=lambda t: t["y_feat"])
s_vals = [t["y_feat"] for t in sorted_tracks]
n = len(sorted_tracks)
total_var = float(np.var(s_vals))
best_ratio = 0.0
best_k = 1

for k in range(1, n):
    g1 = s_vals[:k]
    g2 = s_vals[k:]
    v_between = (len(g1) * len(g2) * (float(np.mean(g1)) - float(np.mean(g2)))**2) / (n * n)
    ratio = v_between / max(1e-6, total_var)
    if ratio > best_ratio:
        best_ratio = ratio
        best_k = k

print(f"Best split index: k={best_k}, variance explained ratio={best_ratio:.3f}")
cluster1 = sorted_tracks[:best_k]
cluster2 = sorted_tracks[best_k:]

print(f"\nCluster 1 (Back row, smaller Y, count={len(cluster1)}):")
for t in cluster1:
    print(f"  TID {t['tid']:<3} head_x={t['head_x']:<6.1f} head_y={t['head_y']:<6.1f} bot_y={t['bot_y']:<6.1f} y_feat={t['y_feat']:<6.1f}")

print(f"\nCluster 2 (Front row, larger Y, count={len(cluster2)}):")
for t in cluster2:
    print(f"  TID {t['tid']:<3} head_x={t['head_x']:<6.1f} head_y={t['head_y']:<6.1f} bot_y={t['bot_y']:<6.1f} y_feat={t['y_feat']:<6.1f}")
