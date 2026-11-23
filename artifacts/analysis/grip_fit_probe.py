"""Raw gripper predictions (dim 18, chunk step 0) on DAgger frames by oracle label.

Analysis only. Usage: grip_fit_probe.py CKPT DAGGER_DIR [N_EPISODES]
"""
import glob, sys
import numpy as np
import torch

sys.path.insert(0, "/home/keyu/advaitic/wt-bc")
from submission.policy import BCPolicy  # noqa: E402
from submission.model import ACTIVE_DIMS  # noqa: E402

pol = BCPolicy(sys.argv[1])
gi = list(ACTIVE_DIMS).index(18)
files = sorted(glob.glob(sys.argv[2] + "/ep_*.npz"))[: int(sys.argv[3]) if len(sys.argv) > 3 else 60]
out = {"open": [], "closed": []}
for f in files:
    z = np.load(f)
    m, a, st, rgb = z["label_mask"][:, 18], z["action"][:, 18], z["state"], z["rgb"]
    for i in range(len(rgb)):
        t = 2 * i
        if t >= len(m) or not m[t]:
            continue
        p = (st[t, list(pol.proprio_dims)] - pol.mean) / pol.std
        with torch.inference_mode():
            c = pol.model(torch.from_numpy(rgb[i].copy())[None].to(pol.device),
                          torch.from_numpy(p.astype(np.float32))[None].to(pol.device))[0].float().cpu().numpy()
        out["open" if a[t] < 0 else "closed"].append(c[0, gi])
for k, v in out.items():
    v = np.array(v)
    print(f"{k}: n={len(v)} mean={v.mean():+.2f} frac_pred_open={(v < 0).mean():.2f} quantiles={np.quantile(v, [.1, .5, .9]).round(2)}")
