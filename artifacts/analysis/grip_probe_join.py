"""Analysis only: join a probe run's per-step gripper-open probability with the privileged oracle label.

Usage: grip_probe_join.py RESULT_DIR PROBE_JSONL OUT_JSONL
Replays saved actions camera-free (deterministic), computes the DAgger oracle label
(method/dagger/diagnostics.gripper_label) at each pre-action state, and writes per-episode stats.
"""
import csv, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/keyu/advaitic/wt-vla")
from method.dagger.diagnostics import PrivilegedSensors, gripper_label  # noqa: E402
import onnxruntime  # noqa: E402
onnxruntime.disable_telemetry_events()
from bigym.loco import make  # noqa: E402

res, probe, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
# split probe log into episodes (t resets to 0 at each episode start, in eval order)
eps, cur = [], None
for line in probe.read_text().splitlines():
    r = json.loads(line)
    if r["t"] == 0:
        cur = []
        eps.append(cur)
    cur.extend(r["p"])
rows = list(csv.DictReader(open(res / "episodes.csv")))
assert len(eps) == len(rows), (len(eps), len(rows))
env = make("move_plate", camera_keys=())
with out.open("w") as f:
    for row, p in zip(rows, eps):
        ep = int(row["episode"])
        acts = np.load(res / f"episode_{ep:03d}.actions.npz")["actions"]
        env.reset(seed=int(row["seed"]))
        sens = PrivilegedSensors(env.inner_env)
        z0 = sens.snapshot(0)["pose"][2]
        labels = []
        s = sens.snapshot(0)
        for t, a in enumerate(acts):
            labels.append(gripper_label(s, z0))
            ts = env.step(a)
            s = sens.snapshot(t + 1)
            if ts.last():
                break
        n = min(len(labels), len(p))
        P = np.array(p[:n]); L = labels[:n]
        open_idx = [i for i, l in enumerate(L) if l == -1.0]
        closed_idx = [i for i, l in enumerate(L) if l == 1.0]
        executed_open = [i for i in range(n) if acts[i, 18] < 0]
        rec = dict(episode=ep, seed=int(row["seed"]), slot=int(row["source_slot"]),
                   success=row["success"] == "True", steps=n,
                   n_oracle_open=len(open_idx), first_oracle_open=open_idx[0] if open_idx else None,
                   p_open_when_oracle_open=dict(
                       mean=round(float(P[open_idx].mean()), 3) if open_idx else None,
                       max=round(float(P[open_idx].max()), 3) if open_idx else None,
                       frac_over_05=round(float((P[open_idx] > 0.5).mean()), 3) if open_idx else None,
                       frac_over_03=round(float((P[open_idx] > 0.3).mean()), 3) if open_idx else None),
                   p_open_when_oracle_closed_max=round(float(P[closed_idx].max()), 3) if closed_idx else None,
                   first_executed_open=executed_open[0] if executed_open else None)
        f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
