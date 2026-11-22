"""Package the independently measured grasp calibration with the original contact head."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def package(calibration, contact, output):
    report = json.loads(calibration.read_text())
    fit, check = map(set, (report["fit_seeds"], report["check_seeds"]))
    if not fit or not check or fit & check or not all(3056500 <= s < 3056600 for s in fit | check):
        raise ValueError("Expected the predeclared handoff calibration and disjoint check")
    if (
        hashlib.sha256(contact.read_bytes()).hexdigest()
        != "51257d3363a1d8eaae21ba863c36c166299e4e5bf2e0842a69ae81bbd8b7c86e"
    ):
        raise ValueError("Expected the frozen contact-v1 weights")
    normal = np.asarray(report["fitted"], dtype=np.float64)
    if (
        normal.shape != (3,)
        or not np.isfinite(normal).all()
        or abs(np.linalg.norm(normal) - 1) > 0.001
    ):
        raise ValueError("Expected a measured unit normal")
    with np.load(contact, allow_pickle=False) as f:
        weights = dict(f)
    import io

    stream = io.BytesIO()
    np.savez_compressed(stream, **weights, loaded_normal=normal)
    payload = stream.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    dest = output / (digest + ".npz")
    dest.write_bytes(payload)
    return dest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("calibration", type=Path)
    p.add_argument("--contact", type=Path, default=Path("checkpoints/astra/contact_v1.npz"))
    p.add_argument("--output", type=Path, default=Path("checkpoints/astra"))
    a = p.parse_args()
    print(package(a.calibration, a.contact, a.output))
