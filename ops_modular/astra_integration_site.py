"""Fit continuous residuals on whole generation episodes and measure on a separate split."""

import hashlib
import json
import time
from pathlib import Path

import modal

from ops_modular.astra_v2_train import ROOT, data, image

app = modal.App("modular-astra-integration-site")
if modal.is_local():
    image = image.add_local_file(
        ROOT / "method/modular/site_estimator.py", "/root/code/method/modular/site_estimator.py"
    )


@app.function(image=image, cpu=2, volumes={"/data": data}, timeout=1500)
def fit():
    import sys

    sys.path.insert(0, "/root/code")
    import numpy as np
    import torch

    from method.modular.site_estimator import PLACE, plate_features, predict, site_features

    start = time.monotonic()
    data.reload()
    root = Path("/data/astra-dagger/sol")
    dest = Path("/data/astra-integration")
    dest.mkdir(exist_ok=True)
    files = sorted(root.glob("*.npz"))
    if len(files) < 250 or not all(3110000 <= int(p.stem) < 3110300 for p in files):
        raise ValueError("Need the reserved generation collection")
    out = {"train": [], "test": [], "tune": []}
    columns = {g: {s: [] for s in out} for g in ("site", "plate")}
    for p in files:
        s = int(p.stem)
        split = "test" if s % 10 == 0 else "tune" if s % 10 == 1 else "train"
        out[split].append(s)
        with np.load(p) as f:
            raw, truth, state, v, ph = (
                f["baseline"],
                f["truth"],
                f["state"],
                f["features"],
                f["phase"],
            )
            columns["site"][split].append(
                (site_features(v[:1], raw[:1, 3:6]), truth[:1, 3:6] - raw[:1, 3:6], ph[:1])
            )
            mask = np.isin(ph, PLACE)
            if mask.any():
                columns["plate"][split].append(
                    (
                        plate_features(v[mask], state[mask], raw[mask, :3], raw[mask, 3:6]),
                        truth[mask, :3] - raw[mask, :3],
                        ph[mask],
                    )
                )
    weights = {}

    def tensor(x):
        return torch.as_tensor(x, device="cpu", dtype=torch.float64)

    for group, splits in columns.items():

        def arrays(split, splits=splits):
            return tuple(map(np.concatenate, zip(*splits[split], strict=True)))

        x, y, _ = arrays("train")
        mean = x.mean(0)
        std = x.std(0).clip(0.001)
        offset = y.mean(0)
        X = tensor(((x - mean) / std).clip(-8, 8))
        Y = tensor(y - offset)
        gram = X.T @ X
        weights.update({group + "_mean": mean, group + "_scale": std, group + "_offset": offset})
        tx, ty, _ = arrays("tune")
        best = float("inf")
        for penalty in (0.1, 1.0, 10.0):
            coef = (
                torch.linalg.solve(
                    gram + len(x) * penalty * torch.eye(len(mean), device="cpu"), X.T @ Y
                )
                .cpu()
                .numpy()
            )
            weights[group + "_coef"] = coef
            err = np.linalg.norm(predict(weights, group, tx) - ty, axis=1).mean()
            if err < best:
                best = err
                selected = coef.copy()
                selected_penalty = penalty
        weights[group + "_coef"] = selected
        x, y, phase = arrays("test")
        after = predict(weights, group, x) - y
        stats = {}
        for key in ("all", *np.unique(phase)):
            m = np.ones(len(y), bool) if key == "all" else phase == key
            stats[key] = {}
            for arm, e in (("base", -y[m] * 100), ("corrected", after[m] * 100)):
                norm = np.linalg.norm(e, axis=1)
                stats[key][arm] = dict(
                    mae=float(norm.mean()),
                    p95=float(np.percentile(norm, 95)),
                    signed_xyz=e.mean(0).tolist(),
                    axis_mae=np.abs(e).mean(0).tolist(),
                    frames=len(e),
                )
        out[group] = dict(penalty=selected_penalty, metrics=stats)
    ck = dest / "place_v1.npz"
    np.savez_compressed(ck, **weights)
    out.update(sha256=hashlib.sha256(ck.read_bytes()).hexdigest(), seconds=time.monotonic() - start)
    (dest / "place_v1.json").write_text(json.dumps(out, indent=2))
    data.commit()
    return out
