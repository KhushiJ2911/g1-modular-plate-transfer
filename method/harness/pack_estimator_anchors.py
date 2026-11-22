"""Subsample original estimator training rows for rehearsal during fine-tuning."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from method.harness.data import LABEL_KEYS, check_seed
from method.harness.estimator_audit import distance_cm


def select_rows(mask, limit):
    rows = np.flatnonzero(mask)
    return rows[np.linspace(0, len(rows) - 1, min(limit, len(rows)), dtype=int)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    original_train = {Path(p).name for p in checkpoint["train_episodes"]}
    original_dev = {Path(p).name for p in checkpoint["val_episodes"]}
    fields = {key: [] for key in ("seed", "rgb", "state", "target")}
    for file in sorted(args.source.glob("ep_*.npz")):
        if file.name not in original_train:
            continue
        with np.load(file, allow_pickle=False) as episode:
            seed = check_seed(episode["seed"])
            if (
                file.name in original_dev
                or seed != int(file.stem[3:])
                or not 2_500_000 <= seed < 2_501_000
                or int(episode["rgb_stride"]) != 2
            ):
                raise ValueError(f"Invalid DAgger3 anchor source {file}")
            target = np.column_stack([episode[key][::4] for key in LABEL_KEYS]).astype(np.float32)
            dist, tilt, held = distance_cm(target), target[:, 6], target[:, 7].astype(bool)
            near = dist <= 10
            high = near & (tilt > 20) & held
            slanted = near & (tilt > 20) & (tilt <= 40) & ~held
            rows = np.unique(
                np.concatenate(
                    [
                        select_rows(high, 32),
                        select_rows(slanted, 12),
                        select_rows(near & ~(high | slanted), 8),
                        select_rows(~near, 8),
                    ]
                )
            )
            fields["seed"].append(np.full(len(rows), seed, np.int32))
            fields["rgb"].append(episode["rgb"][2 * rows])
            fields["state"].append(episode["state"][4 * rows])
            fields["target"].append(target[rows])
    packed = {key: np.concatenate(chunks) for key, chunks in fields.items()}
    with args.out.open("xb") as file:
        np.savez_compressed(file, **packed)
    target = packed["target"]
    high_held = (distance_cm(target) <= 10) & (target[:, 6] > 20) & target[:, 7].astype(bool)
    print(
        json.dumps(
            {
                "rows": len(target),
                "episodes": len(np.unique(packed["seed"])),
                "near_high_held": int(high_held.sum()),
                "near_high_held_episodes": len(np.unique(packed["seed"][high_held])),
            }
        )
    )


if __name__ == "__main__":
    main()
