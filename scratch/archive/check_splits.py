import numpy as np

y_vals = [613.0, 641.0, 645.0, 650.0, 670.0, 706.0, 709.0, 756.0, 773.0, 821.0]
n = len(y_vals)
tot_wss = np.var(y_vals) * n

for i in range(1, n):
    g1, g2 = y_vals[:i], y_vals[i:]
    wss = np.var(g1) * len(g1) + np.var(g2) * len(g2)
    var_exp = 1.0 - (wss / tot_wss)
    print(f"Split at i={i}: WSS={wss:.1f}, VarExplained={var_exp:.4f}")
