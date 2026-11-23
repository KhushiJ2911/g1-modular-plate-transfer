"""Print a per-dataset overview: episodes, length, slots, grasp/release timing, base motion.

Usage: data_overview.py
"""
import collections
import glob

import numpy as np

for name in ["human60", "teacher_v1", "teacher_v1_dart0", "dagger", "dagger2"]:
    files = [f for f in sorted(glob.glob(f"/home/keyu/advaitic/data/{name}/ep_*.npz")) if ".priv." not in f]
    L, close, open_, slots, succ, dx, dy = [], [], [], collections.Counter(), 0, [], []
    for f in files:
        z = np.load(f)
        a, s = z["action"], z["state"]
        g = a[:, 18] > 0
        L.append(len(a))
        if g.any():
            close.append(int(np.argmax(g)))
            after = np.where(~g[close[-1]:])[0]
            if len(after):
                open_.append(close[-1] + int(after[0]))
        slots[int(z["slot"])] += 1
        succ += bool(z["success"])
        dx.append(s[-1, 18] - s[0, 18]); dy.append(s[-1, 19] - s[0, 19])
    L = np.array(L)
    med = lambda v: f"{int(np.median(v))}" if len(v) else "-"
    print(f"{name:17s} n={len(files):3d} success={succ:3d}  len median {int(np.median(L))} [{L.min()}-{L.max()}]  "
          f"close@{med(close)}  open@{med(open_)} (opened in {len(open_)})  "
          f"base dx {np.mean(dx):+.2f} dy {np.mean(dy):+.2f} m  slots {dict(sorted(slots.items()))}")
