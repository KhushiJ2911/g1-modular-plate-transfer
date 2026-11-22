"""Evaluate open-finger recontact detection on disjoint fresh collection episodes."""

import json
from pathlib import Path

import modal

from ops_modular.astra_integration_release_audit import data, image

app = modal.App("modular-astra-recontact")
if modal.is_local():
    root = Path(__file__).resolve().parents[1]
    image = image.add_local_file(root / "method/modular/recontact.py", "/root/recontact.py")


@app.function(image=image, cpu=2, memory=1024, timeout=800, volumes={"/data": data})
def fit():
    import numpy as np
    from recontact import features, score

    data.reload()
    rows = []
    for s in range(3110000, 3110300):
        with np.load(f"/data/astra-dagger/sol/{s}.npz") as f:
            phase = f["phase"]
            lower = np.flatnonzero(phase == "LOWER")
            if not len(lower):
                continue
            steps = f["step"]
            lo = steps[lower[0]]
            opened = np.flatnonzero(f["actions"][lo:, 18] < 0)
            if not len(opened):
                continue
            op = lo + opened[0]
            mask = (steps >= op + 4) & (steps <= op + 75) & (f["state"][:, 44] <= 0.08)
            if not mask.any():
                continue
            split = {0: "test", 1: "tune"}.get(s % 5, "train")
            base = f["baseline"][mask]
            truth = f["truth"][mask]
            rows.append(
                dict(
                    s=s,
                    split=split,
                    state=f["state"][mask],
                    vis=f["features"][mask],
                    base=base,
                    held=truth[:, 7] > 0.5,
                    steps=steps[mask],
                    success=json.loads(str(f["meta"]))["success"],
                )
            )
    results = {}
    dest = Path("/data/astra-integration")

    def metrics(r, p):
        y = np.concatenate([z["held"] for z in r])
        pred = np.concatenate(p)

        def sustained(x, steps):
            return bool(np.any(x[:-1] & x[1:] & ((steps[1:] - steps[:-1]) <= 6)))

        positive = [sustained(z["held"], z["steps"]) for z in r]
        alarms = [sustained(a, z["steps"]) for z, a in zip(r, p, strict=True)]
        return dict(
            episodes=len(r),
            frames=len(y),
            held_frames=int(y.sum()),
            tp=int((pred & y).sum()),
            fp=int((pred & ~y).sum()),
            fn=int((~pred & y).sum()),
            tn=int((~pred & ~y).sum()),
            hit_episodes=sum(a and b for a, b in zip(positive, alarms, strict=True)),
            positive_episodes=sum(positive),
            false_alarm_episodes=sum(a and not b for a, b in zip(alarms, positive, strict=True)),
            false_alarm_success=sum(a and z["success"] for a, z in zip(alarms, r, strict=True)),
            successes=sum(z["success"] for z in r),
        )

    def evaluation(arm, subset, weights=None, t=0.5):
        rr = [z for z in rows if z["split"] == subset]
        preds = [(z["base"][:, 7] if arm == "v1b" else score(weights, z["x"])) >= t for z in rr]
        return metrics(rr, preds)

    for arm in ("v1b", "proprio", "vision"):
        if arm == "v1b":
            results[arm] = evaluation(arm, "test")
            continue
        for z in rows:
            z["x"] = features(
                z["state"], z["vis"], z["base"][:, :3], z["base"][:, 3:6], arm == "vision"
            )
        tr = [z for z in rows if z["split"] == "train"]
        X = np.concatenate([z["x"] for z in tr]).astype(float)
        y = np.concatenate([z["held"] for z in tr]).astype(float)
        w = np.concatenate([np.ones(len(z["x"])) / len(z["x"]) for z in tr])
        # Balance the two classes while preserving episode weights within each.
        w = np.where(y > 0, w / (2 * w[y > 0].sum()), w / (2 * w[y == 0].sum()))
        mean = (X * w[:, None]).sum(0)
        scale = np.sqrt(((X - mean) ** 2 * w[:, None]).sum(0)).clip(0.001)
        x = ((X - mean) / scale).clip(-10, 10)
        bias = float(w @ y)
        best = -np.inf
        for ridge in (0.001, 0.02, 0.25):
            coef = np.linalg.solve(
                x.T @ (x * w[:, None]) + ridge * np.eye(x.shape[1]), x.T @ ((y - bias) * w)
            )
            m = dict(mean=mean, scale=scale, bias=bias, coef=coef)
            for t in (0.35, 0.45, 0.55, 0.65, 0.75):
                mtr = evaluation(arm, "tune", m, t)
                # Reward episode detection while penalizing false mechanical intervention.
                utility = (
                    mtr["hit_episodes"] / max(1, mtr["positive_episodes"])
                    - 2
                    * mtr["false_alarm_episodes"]
                    / max(1, mtr["episodes"] - mtr["positive_episodes"])
                    + 0.1 * mtr["tp"] / max(1, mtr["held_frames"])
                )
                if utility > best:
                    best = utility
                    weights = m
                    threshold = t
                    penalty = ridge
        results[arm] = dict(
            test=evaluation(arm, "test", weights, threshold),
            tune=evaluation(arm, "tune", weights, threshold),
            threshold=threshold,
            ridge=penalty,
        )
        np.savez_compressed(dest / (arm + "_recontact.npz"), **weights, threshold=threshold)
    results["seeds"] = {
        split: [z["s"] for z in rows if z["split"] == split] for split in ("train", "tune", "test")
    }
    (dest / "recontact.json").write_text(json.dumps(results, indent=2))
    data.commit()
    return results
