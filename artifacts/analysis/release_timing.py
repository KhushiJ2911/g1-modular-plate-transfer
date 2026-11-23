"""Analysis only: plate state at the moment the left gripper is commanded open.

Usage: release_timing.py RESULT_DIR EP[,EP...] OUT.jsonl
Replays saved actions camera-free, same physics; never modifies the env.
"""
import csv, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/keyu/advaitic/wt-teacher")
from method.teacher.diagnostics import PrivilegedSensors  # noqa: E402
import onnxruntime  # noqa: E402
onnxruntime.disable_telemetry_events()
from bigym.loco import make  # noqa: E402

GRIP = 18  # left gripper command: +1 closed, -1 open
res, eps, out = Path(sys.argv[1]), [int(e) for e in sys.argv[2].split(",")], Path(sys.argv[3])
rows = {int(r["episode"]): r for r in csv.DictReader(open(res / "episodes.csv"))}
env = make("move_plate", camera_keys=())


def brief(s):
    return dict(xy=round(s["site_xy_distance"], 3),
                dz=round(s["pose"][2] - s["site_xyz"][2], 3),
                tilt=round(s["tilt_deg"], 1), held=s["held"],
                rack=s["contacts"]["target_rack"], table=s["contacts"]["table"])


for ep in eps:
    row = rows[ep]
    acts = np.load(res / f"episode_{ep:03d}.actions.npz")["actions"]
    env.reset(seed=int(row["seed"]))
    sens = PrivilegedSensors(env.inner_env)
    z0 = sens.snapshot(0)["pose"][2]
    hist, flips, lifted_once = [], 0, False
    release = None
    for t, a in enumerate(acts):
        ts = env.step(a)
        s = sens.snapshot(t + 1)
        hist.append(s)
        lifted = s["pose"][2] - z0 > 0.05
        lifted_once |= lifted
        if t and np.sign(acts[t, GRIP]) != np.sign(acts[t - 1, GRIP]):
            flips += 1
            # first open command after the plate has been carried
            if release is None and lifted_once and acts[t, GRIP] < 0:
                release = t
        if ts.last():
            break
    rec = dict(episode=ep, slot=int(row["source_slot"]), success=row["success"] == "True",
               grip_flips=flips, release_step=release, steps=len(hist))
    if release is not None:
        rec["at_open"] = brief(hist[release])
        rec["before_20"] = brief(hist[max(0, release - 20)])
        rec["after_25"] = brief(hist[min(len(hist) - 1, release + 25)])
        rec["after_100"] = brief(hist[min(len(hist) - 1, release + 100)])
        # did the gripper re-close after the first open?
        rec["reclosed_after_open"] = bool((acts[release:len(hist), GRIP] > 0).any())
    print(json.dumps(rec), flush=True)
    with out.open("a") as f:
        f.write(json.dumps(rec) + "\n")
