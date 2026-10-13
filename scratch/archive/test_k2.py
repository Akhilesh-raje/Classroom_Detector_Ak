import numpy as np

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

thetas = np.linspace(-np.radians(25), np.radians(25), 51)
best_ve = 0
best_theta = 0
best_split = None

for theta in thetas:
    proj_vals = [p["y"] * np.cos(theta) - p["x"] * np.sin(theta) for p in points_v2]
    s_idx = np.argsort(proj_vals)
    s_proj = [proj_vals[i] for i in s_idx]
    sorted_p = [points_v2[i] for i in s_idx]
    n = len(s_proj)
    tot_var = np.var(s_proj)
    tot_mean = np.mean(s_proj)

    for i in range(1, n):
        g1, g2 = s_proj[:i], s_proj[i:]
        v = (len(g1)*(np.mean(g1)-tot_mean)**2 + len(g2)*(np.mean(g2)-tot_mean)**2)/n
        ve = v / max(1e-6, tot_var)
        if ve > best_ve:
            best_ve = ve
            best_theta = theta
            best_split = (sorted_p[:i], sorted_p[i:])

print(f"Angle: {np.degrees(best_theta):.1f} deg, VarExplained: {best_ve:.3f}")
r1, r2 = best_split
if np.median([p["y"] for p in r1]) < np.median([p["y"] for p in r2]):
    back, front = r1, r2
else:
    front, back = r1, r2

front.sort(key=lambda p: p["x"])
back.sort(key=lambda p: p["x"])

print(f"Front Row ({len(front)}): TIDs = {[p['tid'] for p in front]}")
print(f"Back Row ({len(back)}): TIDs = {[p['tid'] for p in back]}")
