"""Package the independently measured grasp calibration with the original contact head."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def package(calibration, contact, output, mask=None, plate=None, pinch=None):
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
    if pinch is not None:
        p = json.loads(pinch.read_text())
        training = {e["seed"] for e in p["episodes"] if e["split"] == "leave_one_out"}
        checks = {e["seed"] for e in p["episodes"] if e["split"] == "check"}
        bias = np.asarray(p["biases"]["pinch"], dtype=np.float64)
        if (
            training != fit
            or checks != check
            or p["training_choice"] != "pinch_calibrated"
            or bias.shape != (3,)
            or not np.isfinite(bias).all()
        ):
            raise ValueError("Unexpected common pinch calibration")
        weights["pinch_center_bias"] = bias
    if plate is not None:
        if hashlib.sha256(plate.read_bytes()).hexdigest() != (
            "0f4053833b6aac4f187d95bb17f0daa3d3a81d2c4e461ee320b3f6b3f17ffcad"
        ):
            raise ValueError("Expected the DAgger placement weights")
        with np.load(plate, allow_pickle=False) as f:
            weights.update({k: f[k] for k in f.files if k.startswith("plate_")})
    if mask is not None:
        if hashlib.sha256(mask.read_bytes()).hexdigest() != (
            "409e9b7f694de4cfe2ab07e60eb06435ff16394be04ddd0e7e95f8c8ee4b5cfa"
        ):
            raise ValueError("Expected the frozen two-camera segmenter archive")
        with np.load(mask, allow_pickle=False) as f:
            weights.update({k: f[k] for k in f.files if k.startswith("mask/")})
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
    p.add_argument("--mask", type=Path, help="Include the frozen RGB segmenter for lift capture")
    p.add_argument("--plate", type=Path, help="Include the frozen continuous plate XYZ correction")
    p.add_argument("--pinch", type=Path, help="Include the 12-grasp pinch center calibration")
    a = p.parse_args()
    print(package(a.calibration, a.contact, a.output, a.mask, a.plate, a.pinch))
