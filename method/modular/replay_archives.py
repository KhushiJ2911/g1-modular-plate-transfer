"""Replay every archived failure without rendering and measure physical lift/seating."""

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .perception_audit import replay_truth


def run(job):
    folder, rec = job
    seed = int(rec["seed"])
    with np.load(folder / f"episode_{int(rec['episode']):03}.actions.npz") as f:
        actions = f["actions"]
    truth, state, outcome = replay_truth(seed, actions)
    expected = dict(
        steps=int(rec["steps"]),
        success=rec["success"] == "True",
        fell=rec["fell"] == "True",
        termination=rec["termination"],
    )
    if not outcome["terminal"] or any(outcome[k] != v for k, v in expected.items()):
        raise RuntimeError(f"Replayed terminal mismatch {seed}: {outcome} vs {expected}")
    closed = actions[:, 18] > 0
    changes = np.diff(np.r_[False, closed, False].astype(int))
    attempts = []
    for begin, end in zip(np.where(changes == 1)[0], np.where(changes == -1)[0], strict=True):
        t = truth[begin:end]
        attempts.append(
            dict(
                close=int(begin),
                end=int(end),
                held_frames=int(t[:, 10].sum()),
                rise_cm=round(float(np.max(t[:, 2] - truth[0, 2]) * 100), 2),
            )
        )
    return dict(
        seed=seed,
        episode=int(rec["episode"]),
        block=int(folder.name),
        source_slot=int(rec["source_slot"]),
        **outcome,
        attempts=attempts,
        peak_lift_cm=round(float(np.max(truth[:, 2] - truth[0, 2]) * 100), 2),
        target_contact_frames=int(truth[:, 12].sum()),
        held_frames=int(truth[:, 10].sum()),
        final_site_distance_cm=round(
            float(np.linalg.norm(truth[-1, :3] - truth[-1, 3:6]) * 100), 2
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    jobs = []
    for path in sorted(args.root.glob("*/episodes.csv")):
        with path.open() as f:
            jobs.extend(
                (path.parent, row) for row in csv.DictReader(f) if row["success"] == "False"
            )
    with ProcessPoolExecutor(args.workers) as pool, args.output.open("w") as stream:
        for result in pool.map(run, jobs):
            print(json.dumps(result), file=stream, flush=True)
