"""Offline probe: does the BC gripper prediction just copy the observed gripper state?

For demo frames, predict gripper (dim 18, first step of chunk) with the real state and
with the gripper state (44, 45) and finger joints (7, 8) swapped to the other phase's values.
Analysis only. Usage: copycat_probe.py CKPT EP_NPZ...
"""
import sys
import numpy as np
import torch

sys.path.insert(0, "/home/keyu/advaitic/wt-bc")
from submission.policy import BCPolicy  # noqa: E402

pol = BCPolicy(sys.argv[1])
GDIMS = [7, 8, 44, 45]
for path in sys.argv[2:]:
    z = np.load(path)
    rgb, st, act, stride = z["rgb"], z["state"], z["action"], int(z["rgb_stride"])
    closed = act[:, 18] > 0
    on, off = np.where(closed)[0], np.where(~closed)[0]
    ref_closed = st[on[len(on) // 2], GDIMS]  # gripper/fingers while holding
    ref_open = st[off[0], GDIMS]
    agree_real = agree_flip = n = 0
    for i in range(0, len(rgb)):
        t = i * stride
        for variant in ("real", "flip"):
            s = st[t].copy()
            if variant == "flip":
                s[GDIMS] = ref_open if closed[t] else ref_closed
            pol.reset()
            g = pol({"rgb": rgb[i], "state": s})[0, 18] > 0
            if variant == "real":
                agree_real += g == closed[t]
            else:
                agree_flip += g == closed[t]
        n += 1
    print(f"{path.split('/')[-1]}: frames={n} label-agreement real-state={agree_real/n:.2f} "
          f"flipped-gripper-state={agree_flip/n:.2f}")
