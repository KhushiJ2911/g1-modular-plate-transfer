"""Rendered privileged rollouts to audit the observation-only estimator, pre-action."""

import argparse
import gzip
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

PHASE_GROUPS = {
    "APPROACH": "approach",
    "PREGRASP": "pregrasp",
    "GRASP": "grasp",
    "LIFT": "lift",
    "CARRY": "carry",
    "ALIGN": "place",
    "LOWER": "place",
    "RELEASE": "release",
    "SETTLE": "settle",
    "RETREAT": "retreat",
    "HOLD": "hold",
}


def run(seed, out, checkpoint):
    from advaitic_eval.environment import make_environment, observation
    from submission.estimator import Estimator

    from .controller import ModularController
    from .estimated import EstimatedTaskState
    from .kinematics import G1Kinematics
    from .privileged import PrivilegedTaskState

    if not (3_000_000 <= seed < 2**32) or 7_310_000 <= seed < 7_320_000:
        raise ValueError("Use an unreserved privileged development seed")
    estimator = EstimatedTaskState(Estimator(checkpoint), G1Kinematics())
    controller = ModularController(G1Kinematics())
    env = make_environment()
    hashes = {
        p: hashlib.sha256(Path(__file__).with_name(p).read_bytes()).hexdigest()
        for p in ("controller.py", "phases.py", "privileged.py", "estimated.py", "kinematics.py")
    }
    try:
        ts = env.reset(seed=seed)
        truth_source = PrivilegedTaskState(env)
        shape = list(observation(ts)["rgb"].shape)
        if shape != [3, 3, 84, 84]:
            raise ValueError(f"Unexpected evaluator RGB shape {shape}")
        steps = 0
        with gzip.open(out, "xt") as stream:
            while not ts.last():
                obs = observation(ts)
                truth = truth_source(obs["state"])
                pred = estimator.update(obs)
                row = {
                    "seed": seed,
                    "step": steps,
                    "phase": controller.phase.value,
                    "held_probability": estimator.held_probability,
                }
                for label, values in (("truth", truth), ("estimate", pred)):
                    row[label] = {
                        k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in values.items()
                    }
                stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
                ts = env.step(controller.act(obs["state"], truth))
                steps += 1
        return dict(
            seed=seed,
            steps=steps,
            success=env.episode_succeeded(),
            fell=env.episode_fell(),
            termination=env.episode_termination(),
            rgb_shape=shape,
            source_sha256=hashes,
        )
    finally:
        env.close()


def summarize(directory):
    groups = defaultdict(list)
    for path in sorted(directory.glob("*.jsonl.gz")):
        with gzip.open(path, "rt") as stream:
            for line in stream:
                row = json.loads(line)
                est, truth = row["estimate"], row["truth"]
                plate = np.abs(np.subtract(est["plate_rel"], truth["plate_rel"])) * 100
                site = np.abs(np.subtract(est["target_site_rel"], truth["target_site_rel"])) * 100
                metrics = dict(zip(("x_cm", "y_cm", "z_cm"), plate.tolist(), strict=True))
                metrics.update(
                    plate_cm=np.linalg.norm(plate),
                    site_cm=np.linalg.norm(site),
                    tilt_deg=abs(est["tilt_deg"] - truth["tilt_deg"]),
                    held_prob=abs(row["held_probability"] - truth["held"]),
                    held_wrong=float(est["held"] != truth["held"]),
                )
                for key in ("source_rack", "target_rack", "table", "floor"):
                    metrics[key + "_wrong"] = float(est["contacts"][key] != truth["contacts"][key])
                for key in ("plate_rot", "site_rot"):
                    delta = np.array(est[key]) @ np.array(truth[key]).T
                    metrics[key + "_deg"] = float(
                        np.rad2deg(np.arccos(np.clip((np.trace(delta) - 1) / 2, -1, 1)))
                    )
                groups[PHASE_GROUPS[row["phase"]]].append((row["seed"], metrics))
    summary = {}
    for phase in dict.fromkeys(PHASE_GROUPS.values()):
        rows = groups[phase]
        record = {"steps": len(rows), "episodes": len({s for s, _ in rows})}
        if rows:
            for k in rows[0][1]:
                vals = [m[k] for _, m in rows]
                record[k] = {"mean": float(np.mean(vals)), "p95": float(np.percentile(vals, 95))}
        summary[phase] = record
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    cmd = parser.add_subparsers(dest="command", required=True)
    episode = cmd.add_parser("run")
    episode.add_argument("seed", type=int)
    episode.add_argument("out", type=Path)
    stats = cmd.add_parser("summarize")
    stats.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        result = run(args.seed, args.out, os.environ["ADV_ESTIMATOR"])
    else:
        result = summarize(args.directory)
    print(json.dumps(result, indent=2))
