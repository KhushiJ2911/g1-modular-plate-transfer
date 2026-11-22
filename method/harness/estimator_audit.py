"""Measure estimator error and release geometry on frozen, seed-disjoint frames."""

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import torch

from method.harness.data import ESTIMATOR_PHASES, dagger2_challenge_seed, flywheel2_split
from submission.estimator import Estimator
from submission.harness import DEFAULT_CONFIG
from submission.model import PROPRIO_DIMS


def partition_files(directory, split):
    files = sorted(Path(directory).glob("ep_*.npz"))
    if not files:
        raise ValueError(f"No {split} episodes in {directory}")
    for file in files:
        with np.load(file, allow_pickle=False) as episode:
            seed = int(episode["seed"])
        if split == "challenge":
            dagger2_challenge_seed(seed)
        if seed != int(file.stem[3:]) or (split != "challenge" and flywheel2_split(seed) != split):
            raise ValueError(f"Unexpected seed {seed} in {split}: {file}")
    return files


def audit_disjoint(checkpoint, files):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    touched = {Path(file).stem for key in ("train_episodes", "val_episodes") for file in saved[key]}
    overlap = touched.intersection(file.stem for file in files)
    if overlap:
        raise ValueError(f"Audit episodes used by checkpoint: {sorted(overlap)}")


def distance_cm(physical):
    return np.linalg.norm(physical[:, :3] - physical[:, 3:6], axis=1) * 100


@torch.inference_mode()
def predict(estimator, rgb, states, batch_size=32):
    stats = estimator.stats
    proprio = (states[:, PROPRIO_DIMS] - stats["proprio_mean"]) / stats["proprio_std"]
    outputs = []
    for start in range(0, len(rgb), batch_size):
        stop = start + batch_size
        outputs.append(
            estimator.model(
                torch.from_numpy(rgb[start:stop]).to(estimator.device),
                torch.from_numpy(proprio[start:stop]).to(estimator.device),
            )
            .float()
            .cpu()
            .numpy()
        )
    output = np.concatenate(outputs)
    output[:, :7] = output[:, :7] * stats["target_std"] + stats["target_mean"]
    if not np.isfinite(output).all():
        raise ValueError("Nonfinite prediction")
    return output


def mean(value):
    return float(value.mean()) if value.size else None


def metrics(true, pred, seeds, threshold=5.0):
    dist = distance_cm(true)
    pred_dist = distance_cm(pred)
    distance_error = pred_dist - dist
    tilt_error = pred[:, 6] - true[:, 6]
    gate = (dist <= threshold) & (true[:, 6] <= 12)
    pred_gate = (pred_dist <= threshold) & (pred[:, 6] <= 12)
    harness_gate = pred_gate & (pred[:, 6] >= 0)
    return {
        "n": len(dist),
        "episodes": len(np.unique(seeds)),
        "distance_mae_cm": mean(np.abs(distance_error)),
        "distance_bias_cm": mean(distance_error),
        "tilt_mae_deg": mean(np.abs(tilt_error)),
        "tilt_bias_deg": mean(tilt_error),
        "tilt_error_gt_5": mean(np.abs(tilt_error) > 5),
        "gate_accuracy": mean(gate == pred_gate),
        "gate_fp": int((~gate & pred_gate).sum()),
        "gate_fp_distance_only": int(((dist > threshold) & (true[:, 6] <= 12) & pred_gate).sum()),
        "gate_fp_tilt_only": int(((dist <= threshold) & (true[:, 6] > 12) & pred_gate).sum()),
        "gate_fp_both": int(((dist > threshold) & (true[:, 6] > 12) & pred_gate).sum()),
        "gate_fn": int((gate & ~pred_gate).sum()),
        "gate_pos": int(gate.sum()),
        "gate_neg": int((~gate).sum()),
        "gate_fpr": mean(pred_gate[~gate]),
        "gate_fnr": mean(~pred_gate[gate]),
        "geometry_accuracy_with_nonnegative_tilt": mean(gate == harness_gate),
        "negative_predicted_tilt": int((pred[:, 6] < 0).sum()),
        "held_accuracy": mean((pred[:, 7] >= 0) == true[:, 7].astype(bool)),
    }


def group_masks(rows):
    true = rows["target"]
    dist, tilt = distance_cm(true), true[:, 6]
    phase, contact = rows["phase"], rows["contact"]
    hidden, exposed = rows["occlusion"].transpose(2, 0, 1)
    measurable_view = exposed >= 10
    blocked_view = measurable_view & (hidden >= 3) & (hidden >= 0.10 * exposed)
    measured = measurable_view.any(axis=1)
    occluded = measured & ~(measurable_view & ~blocked_view).any(axis=1)
    place = phase == ESTIMATOR_PHASES.index("place")
    held = true[:, 7].astype(bool)
    distance_bins = {
        "Distance 0–3 cm": dist < 3,
        "Distance 3–6 cm": (dist >= 3) & (dist < 6),
        "Distance 6–10 cm": dist >= 6,
    }
    tilt_bins = {
        "Tilt <10°": tilt < 10,
        "Tilt 10–20°": (tilt >= 10) & (tilt <= 20),
        "Tilt >20°": tilt > 20,
    }
    phase_bins = {name.capitalize(): phase == i for i, name in enumerate(ESTIMATOR_PHASES)}
    occlusion_bins = {
        "Gripper occluded in all measurable views": occluded,
        "At least one view clear of the gripper": measured & ~occluded,
        "Plate not measurable in any camera": ~measured,
    }
    near = dist <= 10
    groups = {
        "All near rack": near,
        **distance_bins,
        **tilt_bins,
        **phase_bins,
        **occlusion_bins,
        "Head view occluded by gripper": blocked_view[:, 0],
        "Head view clear of gripper": measurable_view[:, 0] & ~blocked_view[:, 0],
        "Rack contact": contact,
        "Rack contact, tilt >20°": contact & (tilt > 20),
        "Rack contact, tilt 23–25°": contact & (tilt >= 23) & (tilt <= 25),
        "Hover 5–7 cm, no rack contact": (dist > 5) & (dist <= 7) & ~contact,
        "Held near rack": held,
        "Held near rack, tilt >20°": held & (tilt > 20),
        "Held near rack, tilt 20–40°": held & (tilt > 20) & (tilt <= 40),
        "Place, held and quiet": place & held & rows["quiet"],
        "Place, within 5 cm, tilt >20°": place & (dist <= 5) & (tilt > 20),
        "Released, no rack contact": phase_bins["Released"] & ~contact,
        "Released, not quiet": phase_bins["Released"] & ~rows["quiet"],
        "Released, no longer held": phase_bins["Released"] & ~held,
        "Successful episode": rows["success"],
        "Failed episode": ~rows["success"],
    }
    joint_groups = {}
    for bins in product(
        distance_bins.items(), tilt_bins.items(), phase_bins.items(), occlusion_bins.items()
    ):
        names, masks = zip(*bins, strict=True)
        joint_groups[" / ".join(names)] = np.logical_and.reduce(masks)
    return (
        {name: mask & near for name, mask in groups.items()},
        {name: mask & near for name, mask in joint_groups.items()},
    )


def group_metrics(rows, pred):
    true = rows["target"]
    groups, joint_groups = group_masks(rows)
    result = {
        key: metrics(true[mask], pred[mask], rows["seed"][mask]) for key, mask in groups.items()
    }
    near = groups["All near rack"]
    result["All near rack at 3.5 cm gate"] = metrics(
        true[near], pred[near], rows["seed"][near], threshold=3.5
    )
    return result, {
        key: metrics(true[mask], pred[mask], rows["seed"][mask])
        for key, mask in joint_groups.items()
    }


def audit(checkpoint, files, device):
    audit_disjoint(checkpoint, files)
    estimator = Estimator(checkpoint, device)
    fields = ("target", "phase", "contact", "occlusion", "step")
    collected = {k: [] for k in (*fields, "seed", "success", "quiet", "prediction")}
    for file in files:
        with np.load(file, allow_pickle=False) as e:
            n = len(e["step"])
            if not n:
                continue
            rgb, state = e["rgb"], e["state"]
            pred = predict(estimator, rgb, state)
            if not collected["prediction"]:
                single = estimator({"rgb": rgb[0], "state": state[0]})
                np.testing.assert_allclose(
                    pred[0, :7],
                    np.r_[single["plate_rel"], single["target_site_rel"], single["tilt_deg"]],
                    atol=1e-4,
                )
            collected["prediction"].append(pred)
            collected["quiet"].append(
                (np.abs(state[:, 22:29]).max(axis=1) <= DEFAULT_CONFIG["arm_speed_rad_s"])
                & (np.linalg.norm(state[:, 40:42], axis=1) <= DEFAULT_CONFIG["base_speed_m_s"])
            )
            for key in fields:
                collected[key].append(e[key])
            for key in ("seed", "success"):
                collected[key].append(np.full(n, e[key]))
    return {key: np.concatenate(parts) for key, parts in collected.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=("test", "challenge"), default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    files = partition_files(args.data, args.split)
    rows = audit(args.checkpoint, files, args.device)
    groups, joint_groups = group_metrics(rows, rows["prediction"])
    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "seeds_replayed": [int(file.stem[3:]) for file in files],
        "groups": groups,
        "joint_groups": joint_groups,
    }
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(args.out.with_suffix(".npz"), **rows)


if __name__ == "__main__":
    main()
