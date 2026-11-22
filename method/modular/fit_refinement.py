"""Fit continuous normal and contact heads with a disjoint split of whole episodes."""

import argparse
import json
from pathlib import Path

import numpy as np

from .refined import GROUP_PHASES, inputs


def examples(path, group):
    with np.load(path) as record:
        info = json.loads(str(record["info"]))
        if info["driver"] not in ("truth", "estimate") or not 3_000_000 <= info["seed"] < 3_100_000:
            raise ValueError("Only fresh privileged-development samples can train the head")
        mask = np.isin(record["phase"], GROUP_PHASES[group])
        pred, truth = record["pred"][mask], record["truth"][mask]
        x = inputs(
            record["features"][mask].astype(float),
            record["state"][mask],
            record["hand"][mask],
            pred,
            info["first_plate"],
            info["first_site"],
        )
        target = truth[:, 6:9] - pred[:, 6:9] if group == "normal" else truth[:, 12:13]
        return x, target, pred, truth, record["phase"][mask]


def metric(pred, truth, output, group):
    if group == "contact":
        target = truth[:, 12]

        def score(p):
            return dict(
                error=float(np.mean(p != target)),
                tp=int(np.sum(p & (target == 1))),
                fn=int(np.sum(~p & (target == 1))),
                fp=int(np.sum(p & (target == 0))),
            )

        return dict(v1b=score(pred[:, 12].astype(bool)), head=score(output[:, 0] >= 0.5))
    normal = pred[:, 6:9] + output
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-8)

    def score(n, tilt):
        error = np.degrees(np.arccos(np.clip((n * truth[:, 6:9]).sum(1), -1, 1)))
        tilt_err = abs(tilt - truth[:, 9])
        return dict(
            normal_mae_deg=float(error.mean()),
            normal_p95_deg=float(np.percentile(error, 95)),
            tilt_mae_deg=float(tilt_err.mean()),
            tilt_p95_deg=float(np.percentile(tilt_err, 95)),
        )

    return dict(
        v1b=score(pred[:, 6:9], pred[:, 9]),
        head=score(normal, np.degrees(np.arccos(np.clip(-normal[:, 1], -1, 1)))),
    )


def fit(train, valid, groups, penalty, out):
    weights = {"groups": ",".join(groups)}
    report = dict(
        train=[p.stem for p in train], validation=[p.stem for p in valid], penalty=penalty
    )
    for group in groups:

        def batch(paths, group=group):
            return [
                np.concatenate(values)
                for values in zip(*(examples(p, group) for p in paths), strict=True)
            ]

        x, y, *_ = batch(train)
        mean = x.mean(0)
        scale = np.maximum(x.std(0), 0.005)
        norm = (x - mean) / scale
        bias = y.mean(0)
        gram = norm.T @ norm
        gram.flat[:: gram.shape[0] + 1] += penalty
        coef = np.linalg.solve(gram, norm.T @ (y - bias))
        weights.update(
            {
                group + "_" + k: v
                for k, v in dict(mean=mean, scale=scale, bias=bias, coef=coef).items()
            }
        )
        val, _, pred, truth, phases = batch(valid)
        estimate = (val - mean) / scale @ coef + bias
        report[group] = dict(
            train_frames=len(x), valid_frames=len(val), overall=metric(pred, truth, estimate, group)
        )
        for phase in np.unique(phases):
            keep = phases == phase
            report[group][phase] = metric(pred[keep], truth[keep], estimate[keep], group)
    np.savez_compressed(out, **weights)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("group", choices=["normal", "contact", "normal,contact"])
    p.add_argument("output", type=Path)
    p.add_argument("input", type=Path, nargs="+")
    p.add_argument("--penalty", type=float, default=60.0)
    args = p.parse_args()
    paths = sorted(p for folder in args.input for p in folder.glob("*.npz"))
    train = [p for p in paths if int(p.stem.split("-")[-1]) % 5]
    valid = [p for p in paths if p not in train]
    print(json.dumps(fit(train, valid, args.group.split(","), args.penalty, args.output), indent=2))
