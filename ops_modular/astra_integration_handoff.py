"""Quantify the stability of the plate's hand-frame pose on recorded public-policy trajectories."""

import json
import subprocess
from pathlib import Path

import modal

from ops_modular.modular_sim import image

app = modal.App("modular-astra-handoff")
data = modal.Volume.from_name("modular-data")
if modal.is_local():
    image = image.add_local_file(Path(__file__), "/root/code/handoff_source.py")


def analyze():
    import numpy as np
    from method.modular.kinematics import G1Kinematics
    from method.modular.orientation_feedback import LEFT_GRASP_NORMAL

    kin = G1Kinematics()
    rows = []

    def pose(i):
        kin.set_state(state[i])
        pos, R = kin.hand_pose("left")
        n = np.cross(truth[i, 8:11], truth[i, 11:14])
        return R.T @ n, R.T @ (truth[i, :3] - pos)

    def deg(x, y):
        return float(
            np.rad2deg(np.arccos(np.clip(x @ y / np.linalg.norm(x) / np.linalg.norm(y), -1, 1)))
        )

    for seed in range(3110000, 3110300):
        with np.load(f"/data/astra-dagger/sol/{seed}.npz") as f:
            ph, truth, state, steps = f["phase"], f["truth"], f["state"], f["step"]
            lower = np.flatnonzero((ph == "LOWER") & (truth[:, 7] > 0.5))
            if not len(lower):
                continue
            # Evaluate at the last held LOWER observation before the first open.
            loc = steps[lower[0]]
            opens = np.flatnonzero(f["actions"][loc:, 18] < 0)
            if len(opens):
                lower = lower[steps[lower] <= loc + opens[0]]
            if not len(lower):
                continue
            idx = lower[-1]
            target, center = pose(idx)
            r = dict(
                seed=seed,
                opened=bool(len(opens)),
                success=json.loads(str(f["meta"]))["success"],
                prior_deg=deg(LEFT_GRASP_NORMAL, target),
            )
            # Use the final attempted pickup before this placement rather than
            # attributing earlier failed pinches to the carried grasp.
            for phase, position in (("GRASP", -1), ("LIFT", -1), ("CARRY", 0), ("ALIGN", 0)):
                opts = np.flatnonzero((ph == phase) & (truth[:, 7] > 0.5) & (steps < steps[idx]))
                if not len(opts):
                    continue
                if position == 0:
                    # Select the start of the latest visit if a placement retried.
                    opts = (
                        opts[np.r_[np.diff(opts) > 1, True].nonzero()[0][-2] + 1 :]
                        if (np.diff(opts) > 1).any()
                        else opts
                    )
                n, p = pose(opts[position])
                r[phase] = dict(
                    oracle_deg=deg(n, target),
                    center_drift_cm=float(100 * np.linalg.norm(p - center)),
                )
            rows.append(r)
    report = {}
    for key, group in (
        ("all", rows),
        ("opened_success", [r for r in rows if r["opened"] and r["success"]]),
        ("opened_failure", [r for r in rows if r["opened"] and not r["success"]]),
    ):

        def summary(vals):
            return (
                dict(n=len(vals), median=float(np.median(vals)), p95=float(np.percentile(vals, 95)))
                if vals
                else dict(n=0)
            )

        report[key] = dict(prior_deg=summary([r["prior_deg"] for r in group]))
        for phase in ("GRASP", "LIFT", "CARRY", "ALIGN"):
            report[key][phase] = {
                k: summary([r[phase][k] for r in group if phase in r])
                for k in ("oracle_deg", "center_drift_cm")
            }
    return report


@app.function(image=image, cpu=1, memory=2048, timeout=900, volumes={"/data": data})
def report():
    # The policy interpreter has pinned MuJoCo and the standalone kinematics.
    # Execute the analysis body without importing Modal into that interpreter.
    import inspect

    body = inspect.getsource(analyze)
    cmd = [
        "/root/policy/bin/python",
        "-c",
        "import json\n" + body + "\nprint(json.dumps(analyze()))",
    ]
    p = subprocess.run(cmd, cwd="/root/code", capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr[-1800:])
    out = json.loads(p.stdout.splitlines()[-1])
    (Path("/data/astra-integration") / "handoff.json").write_text(json.dumps(out, indent=2))
    data.commit()
    return out
