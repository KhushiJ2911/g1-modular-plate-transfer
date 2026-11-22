"""Test whether observed support and arm deflection warn of failure before opening."""

import json
import time
from pathlib import Path

import modal

from ops_modular.astra_integration_release_audit import data, image

if modal.is_local():
    layout = Path(__file__).resolve().parents[1] / "src/advaitic_eval/action_layout.json"
    image = image.add_local_file(layout, "/root/action_layout.json")
app = modal.App("modular-astra-preopen-audit-v2")


@app.function(image=image, volumes={"/data": data}, cpu=1, memory=768, timeout=900)
def audit(smoke=False, seed_for_smoke=3110003):
    import numpy as np

    start = time.monotonic()
    limits = json.loads(Path("/root/action_layout.json").read_text())
    low, high = (np.asarray(limits[k])[4:11] for k in ("action_low", "action_high"))
    rows = []
    if not 3110000 <= seed_for_smoke < 3110300:
        raise ValueError("Generation seed required")
    seeds = (seed_for_smoke,) if smoke else range(3110000, 3110300)
    for seed in seeds:
        with np.load(f"/data/astra-dagger/sol/{seed}.npz", allow_pickle=False) as f:
            actions, phases, steps = (f[k] for k in ("actions", "phase", "step"))
            lower = steps[phases == "LOWER"]
            if not len(lower):
                continue
            opening = np.flatnonzero(actions[lower[0] :, 18] < 0)
            if not len(opening):
                continue
            step = int(lower[0] + opening[0])
            eligible = np.flatnonzero((steps < step) & (steps >= step - 15))
            if len(eligible) < 2:
                continue
            state, pred = f["state"][eligible], f["baseline"][eligible]
            gap = pred[:, :3] - pred[:, 3:6]
            dt = steps[eligible[-1]] - steps[eligible[0]]
            xyz_speed = (gap[-1] - gap[0]) / dt
            # Commands from the PRECEDING observation are known before issuing an opening.
            cmd = actions[steps[eligible] - 1, 4:11]
            tracking = (low + (cmd + 1) / 2 * (high - low)) - state[:, :7]
            geom = np.r_[
                gap[-1], xyz_speed, np.ptp(gap, axis=0), pred[-1, 6], pred[-1, 6] - pred[0, 6]
            ]
            mech = np.r_[
                tracking.mean(0),
                tracking[-1] - tracking[0],
                state[:, 22:29].mean(0),
                state[:, 7:9].mean(0),
                state[-1, 7:9] - state[0, 7:9],
                state[:, 29:31].mean(0),
            ]
            rows.append((seed, not json.loads(str(f["meta"]))["success"], np.r_[geom, mech]))
    if smoke:
        return dict(seed=seed_for_smoke, checked=1, opened=len(rows), seconds=time.monotonic() - start)
    nums, labels, x = zip(*rows, strict=True)
    seed, y, x = np.array(nums), np.array(labels), np.stack(x)
    fit, valid, test = seed % 5 >= 2, seed % 5 == 1, seed % 5 == 0

    def metric(pred, mask):
        return dict(
            positives=int(y[mask].sum()),
            negatives=int((~y[mask]).sum()),
            hits=int((pred[mask] & y[mask]).sum()),
            false_alarms=int((pred[mask] & ~y[mask]).sum()),
        )

    out = {}
    for label, cols in [
        ("visual_tilt", None),
        ("geometry", slice(0, 11)),
        ("geometry_and_tracking", slice(None)),
    ]:
        if cols is None:
            scores = x[:, 9]
        else:
            v = x[:, cols]
            z = np.clip((v - v[fit].mean(0)) / np.maximum(v[fit].std(0), 1e-6), -8, 8)
            # Fixed ridge strength on standardized features before looking at either held-out split.
            a = z[fit]
            coef = np.linalg.solve(
                a.T @ a + 0.1 * fit.sum() * np.eye(a.shape[1]), a.T @ (y[fit] - y[fit].mean())
            )
            scores = z @ coef + y[fit].mean()
        cuts = [float("inf"), *np.unique(scores[valid])]
        candidates = []
        for cut in cuts:
            m = metric(scores >= cut, valid)
            if m["false_alarms"] <= 0.10 * m["negatives"]:
                candidates.append((m["hits"], -m["false_alarms"], cut))
        threshold = max(candidates)[2]
        out[label] = {
            k: metric(scores >= threshold, mask)
            for k, mask in [("fit", fit), ("validation", valid), ("test", test)]
        }
        out[label]["threshold"] = float(threshold) if np.isfinite(threshold) else None
    return dict(
        episodes=len(y),
        target="eventual failed transfer given an opening",
        metrics=out,
        seconds=time.monotonic() - start,
    )
