"""Success-filtered self-training data: keep a policy's own successful rollouts, label = executed actions.
python flywheel_filter.py SRC_ROOT DST  (SRC_ROOT holds shard_*/ep_*.npz from method.dagger.collect)"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=False)
n = k = bad = 0
for f in sorted(glob.glob(str(src / "shard_*" / "ep_*.npz"))):
    n += 1
    try:
        z = np.load(f, allow_pickle=False)
        if not bool(z["success"]):
            continue
        a = {key: z[key] for key in z.files}
    except Exception as exc:  # truncated download: skip, count, never guess
        bad += 1
        print("unreadable", f, type(exc).__name__)
        continue
    a["action"] = np.clip(a["action_executed"], -1, 1).astype(np.float32)
    a["label_mask"] = np.ones_like(a["label_mask"], dtype=bool)
    np.savez_compressed(dst / Path(f).name, **a)
    k += 1
(dst / "flywheel_config.json").write_text(json.dumps({"src": str(src), "episodes": n, "kept_successes": k, "unreadable": bad}))
print("episodes", n, "kept", k, "unreadable", bad)
