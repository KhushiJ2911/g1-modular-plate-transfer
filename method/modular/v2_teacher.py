"""Select synchronized RGB/labels from the disclosed, already recorded teacher archive."""

import argparse
import json
from pathlib import Path

import numpy as np


def convert(src, dest, count):
    # The 2200000 teacher block was already used for v1b training.
    dest.mkdir(parents=True, exist_ok=True)
    for file in sorted(src.glob("ep_*.npz"))[:count]:
        with np.load(file) as f:
            seed = int(f["seed"])
            if not 2_200_000 <= seed < 2_201_000:
                raise ValueError(f"Unexpected training seed {seed}")
            stride = int(f["rgb_stride"])
            step = np.arange(0, len(f["state"]), stride * 4)
            y = np.full((len(step), 18), np.nan, np.float32)
            y[:, :3] = f["priv_plate_rel"][step]
            y[:, 6] = f["priv_plate_tilt_deg"][step]
            y[:, 7] = f["priv_held"][step]
            # Historical site labels are nearest-site, not J's chosen site.
            # Missing target/rotation/contact labels never enter the loss.
            np.savez_compressed(
                dest / f"{seed}.npz",
                rgb=f["rgb"][step // stride],
                state=f["state"][step],
                truth=y,
                step=step,
                meta=json.dumps(dict(seed=seed, source=src.name, frames=len(step))),
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("destination", type=Path)
    p.add_argument("--count", type=int, default=60)
    a = p.parse_args()
    convert(a.source, a.destination, a.count)
