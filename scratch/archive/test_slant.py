import os
import sys
import numpy as np

# Let's test perspective-aware 2D clustering
# Given points (x, y_feat):
# We can estimate the classroom desk slant/slope m = dy/dx across rows, or use 2D k-means / GMM on (normalized x, normalized y)
# Or fit 1 or 2 parallel lines y = m*x + c_i

points_v2 = [
    {"tid": 14, "x": 297.5, "y": 607.2},
    {"tid": 13, "x": 421.0, "y": 633.8},
    {"tid": 17, "x": 607.0, "y": 665.7},
    {"tid": 16, "x": 729.0, "y": 587.7},
    {"tid": 12, "x": 837.0, "y": 716.7},
    {"tid": 11, "x": 1068.5, "y": 727.2},
    {"tid": 15, "x": 1184.0, "y": 621.0},
    {"tid": 10, "x": 1390.5, "y": 763.3},
    {"tid": 18, "x": 1412.5, "y": 609.1},
    {"tid": 9,  "x": 1680.5, "y": 668.5},
]

# In Video 1 (straight frontal view):
points_v1 = [
    {"tid": 3, "x": 392.0, "y": 873.1},
    {"tid": 4, "x": 589.0, "y": 479.9},
    {"tid": 5, "x": 736.0, "y": 658.3},
    {"tid": 2, "x": 1268.0, "y": 402.9},
    {"tid": 7, "x": 1326.5, "y": 504.8},
    {"tid": 1, "x": 1584.0, "y": 772.0},
]

def cluster_with_slant(points):
    # Find optimal slant angle theta in [-30 deg, +30 deg]
    # Projected coordinate: y_proj = y * cos(theta) - x * sin(theta)
    # The true slant maximizes between-row separation (variance explained of 1D projected y)
    best_theta = 0.0
    best_var_exp = 0.0
    best_k = 1
    best_groups = [points]

    thetas = np.linspace(-np.radians(25), np.radians(25), 51)
    
    for theta in thetas:
        proj_vals = [p["y"] * np.cos(theta) - p["x"] * np.sin(theta) for p in points]
        s_idx = np.argsort(proj_vals)
        s_proj = [proj_vals[i] for i in s_idx]
        sorted_p = [points[i] for i in s_idx]
        n = len(s_proj)
        tot_var = np.var(s_proj)
        tot_mean = np.mean(s_proj)

        # Test k=2
        for i in range(1, n):
            g1, g2 = s_proj[:i], s_proj[i:]
            v = (len(g1)*(np.mean(g1)-tot_mean)**2 + len(g2)*(np.mean(g2)-tot_mean)**2)/n
            ve2 = v / max(1e-6, tot_var)
            if ve2 > best_var_exp:
                best_var_exp = ve2
                best_theta = theta
                best_k = 2
                best_groups = [sorted_p[:i], sorted_p[i:]]

        # Test k=3 if n >= 6
        if n >= 6:
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    g1, g2, g3 = s_proj[:i], s_proj[i:j], s_proj[j:]
                    v = (len(g1)*(np.mean(g1)-tot_mean)**2 + len(g2)*(np.mean(g2)-tot_mean)**2 + len(g3)*(np.mean(g3)-tot_mean)**2)/n
                    ve3 = v / max(1e-6, tot_var)
                    if ve3 > best_var_exp and (ve3 - 0.75) > 0.10:
                        best_var_exp = ve3
                        best_theta = theta
                        best_k = 3
                        best_groups = [sorted_p[:i], sorted_p[i:j], sorted_p[j:]]

    print(f"Optimal angle = {np.degrees(best_theta):.1f} deg, Best K = {best_k}, VarExplained = {best_var_exp:.3f}")
    best_groups.sort(key=lambda g: np.median([p["y"] for p in g]), reverse=True)
    for r_idx, g in enumerate(best_groups):
        g.sort(key=lambda p: p["x"])
        tids = [p["tid"] for p in g]
        print(f"  Row R{r_idx+1} ({len(g)} seats): TIDs = {tids}")

print("=== Video 2 (Real classroom) ===")
cluster_with_slant(points_v2)

print("\n=== Video 1 (Student Sitting) ===")
cluster_with_slant(points_v1)
