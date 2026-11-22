"""Release and recontact measurements on fresh rendered DAgger rollouts and their physical labels."""

import json
from pathlib import Path

import modal

app = modal.App("modular-astra-release-audit")
data = modal.Volume.from_name("modular-data")
image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.2.6")


@app.function(image=image, cpu=1, memory=768, timeout=600, volumes={"/data": data})
def audit():
    import numpy as np

    data.reload()
    root = Path("/data/astra-dagger/sol")
    rows = []

    def normal(z):
        return np.cross(z[:, 8:11], z[:, 11:14])

    def delta_degrees(n, m):
        return float(np.rad2deg(np.arccos(np.clip(n @ m, -1, 1))))

    for seed in range(3110000, 3110300):
        with np.load(root / f"{seed}.npz") as f:
            actions, phase, steps, truth, state, base = [
                f[k] for k in ("actions", "phase", "step", "truth", "state", "baseline")
            ]
            outcome = json.loads(str(f["meta"]))
            row = dict(seed=seed, success=outcome["success"])
            entered = np.flatnonzero(phase == "LOWER")
            if not len(entered):
                rows.append(row)
                continue
            lo = steps[entered[0]]
            opened = np.flatnonzero(actions[lo:, 18] < 0)
            if not len(opened):
                rows.append(row)
                continue
            step = lo + opened[0]
            # This is the last observed instant before or at the first open command.
            i = steps.searchsorted(step, side="right") - 1
            j = max(entered[0], steps.searchsorted(step - 12, side="left"))
            # Exit and HOLD frames can be sparse. Require an actual sampled
            # observation within the stated window when computing each event.
            if step - steps[i] > 6:
                rows.append(row)
                continue
            z = truth[i]
            physical = z[:3] - z[3:6]
            yaw = state[i, 21]
            c, s = np.cos(yaw), np.sin(yaw)
            world = np.array(
                [c * physical[0] - s * physical[1], s * physical[0] + c * physical[1], physical[2]]
            )
            normals = normal(truth)
            # The robot yaw can change. Rotate each normal to world before a comparison.
            yaw = state[:, 21]
            c, s = np.cos(yaw), np.sin(yaw)
            x, y = normals[:, 0].copy(), normals[:, 1].copy()
            normals[:, 0] = c * x - s * y
            normals[:, 1] = s * x + c * y
            row.update(
                open=int(step),
                pre_dt=int(steps[i] - steps[j]),
                pre_normal_drift=delta_degrees(normals[i], normals[j]),
                xy_cm=float(100 * np.linalg.norm(world[:2])),
                x_cm=float(100 * world[0]),
                y_cm=float(100 * world[1]),
                depth_cm=float(100 * world[2]),
                tilt=float(z[6]),
                rack=bool(z[15]),
                held=bool(z[7]),
                visual_tilt=float(base[i, 6]),
            )
            window = (steps >= step) & (steps <= step + 75)
            held = truth[window, 7] > 0.5
            free = np.flatnonzero(~held)
            row["physical_release_observed"] = bool(len(free))
            row["recontact_75"] = bool(len(free) and held[free[0] :].any())
            for lag in (12, 30, 60):
                idx = steps.searchsorted(step + lag)
                if idx < len(steps) and steps[idx] - (step + lag) <= 6:
                    row[f"tilt_{lag}"] = float(truth[idx, 6])
                    row[f"rocking_{lag}"] = delta_degrees(normals[idx], normals[i])
                    row[f"rack_{lag}"] = bool(truth[idx, 15])
                    row[f"open_width_{lag}_mm"] = float(1000 * state[idx, 7:9].mean())
            rows.append(row)
    report = {}
    for split in ("train", "holdout", "all"):
        split_rows = [
            r for r in rows if split == "all" or (r["seed"] % 10 == 0) == (split == "holdout")
        ]
        for success in (True, False):
            allrows = [r for r in split_rows if r["success"] == success]
            sub = [r for r in allrows if "open" in r]
            summary = {"episodes": len(allrows), "opened": len(sub)}
            for k in (
                "xy_cm",
                "x_cm",
                "y_cm",
                "depth_cm",
                "tilt",
                "visual_tilt",
                "pre_normal_drift",
                "tilt_12",
                "tilt_30",
                "tilt_60",
                "rocking_12",
                "rocking_30",
                "rocking_60",
                "open_width_12_mm",
                "open_width_60_mm",
            ):
                vals = [r[k] for r in sub if k in r]
                if vals:
                    summary[k] = np.percentile(vals, [10, 50, 90]).tolist()
            for k in (
                "rack",
                "held",
                "physical_release_observed",
                "recontact_75",
                "rack_12",
                "rack_60",
            ):
                vals = [r[k] for r in sub if k in r]
                summary[k] = [sum(vals), len(vals)]
            report[f"{split}_{'success' if success else 'failure'}"] = summary
    dest = Path("/data/astra-integration/release_audit.json")
    dest.write_text(json.dumps(dict(report=report, episodes=rows), indent=2))
    data.commit()
    return report
