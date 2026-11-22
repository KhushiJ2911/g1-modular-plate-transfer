"""Phase-wise localization and sensing errors on recorded observation-driven episodes."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .phase_errors import PHASE_GROUPS


def summarize(paths):
    groups = defaultdict(list)
    for path in paths:
        with np.load(path) as r:
            yaw = r["state"][:, 21]
            c, s = np.cos(yaw), np.sin(yaw)
            p, t = r["pred"], r["truth"]
            delta = p[:, :3] - t[:, :3]
            pelvis = np.column_stack(
                (c * delta[:, 0] + s * delta[:, 1], -s * delta[:, 0] + c * delta[:, 1], delta[:, 2])
            )
            for phase in np.unique(r["phase"]):
                m = r["phase"] == phase
                groups[phase].append((pelvis[m] * 100, p[m], t[m]))
    out = {}
    for phase in PHASE_GROUPS:
        if phase not in groups:
            continue
        episode = groups[phase]
        xyz, p, t = (np.concatenate(v) for v in zip(*episode, strict=True))

        def pair(values):
            return np.round([np.mean(values), np.percentile(values, 95)], 3).tolist()

        out[phase] = dict(
            episodes=len(episode),
            frames=len(p),
            plate_cm=pair(np.linalg.norm(xyz, axis=1)),
            bias_xyz_cm=xyz.mean(0).round(3).tolist(),
            episode_bias_cm=np.mean([np.linalg.norm(e.mean(0)) for e, _, _ in episode]),
            within_episode_noise_cm=np.mean([np.linalg.norm(e.std(0)) for e, _, _ in episode]),
            nearest_site_cm=pair(np.linalg.norm(p[:, 3:6] - t[:, 3:6], axis=1) * 100),
            tilt_deg=pair(np.abs(p[:, 9] - t[:, 9])),
            normal_deg=pair(np.degrees(np.arccos(np.clip((p[:, 6:9] * t[:, 6:9]).sum(1), -1, 1)))),
            held_wrong=int(np.sum(p[:, 10] != t[:, 10])),
            rack_fn=int(np.sum((p[:, 12] == 0) & (t[:, 12] == 1))),
            rack_positive=int(np.sum(t[:, 12] == 1)),
            rack_fp=int(np.sum((p[:, 12] == 1) & (t[:, 12] == 0))),
        )
    return out


if __name__ == "__main__":
    cmd = argparse.ArgumentParser(description=__doc__)
    cmd.add_argument("dirs", type=Path, nargs="+")
    args = cmd.parse_args()
    print(
        json.dumps(summarize(sorted(path for d in args.dirs for path in d.glob("*.npz"))), indent=2)
    )
