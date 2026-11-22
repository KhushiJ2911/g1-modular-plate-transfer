"""Fine-tune the version-1 observation-only estimator on Flywheel2 placement frames."""

import argparse
import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from torch.nn import functional as F

from method.harness.data import check_seed
from method.harness.estimator_audit import distance_cm, metrics, partition_files, predict
from submission.estimator import Estimator
from submission.harness import DEFAULT_CONFIG
from submission.model import PROPRIO_DIMS


def load_rows(files, keys):
    values = {key: [] for key in keys}
    seeds = []
    for file in files:
        with np.load(file, allow_pickle=False) as episode:
            for key in keys:
                values[key].append(episode[key])
            seeds.append(np.full(len(episode["target"]), episode["seed"]))
    return {key: np.concatenate(parts) for key, parts in values.items()} | {
        "seed": np.concatenate(seeds)
    }


def sampling_weights(rows):
    dist, tilt = distance_cm(rows["target"]), rows["target"][:, 6]
    near = dist <= 10
    tilted = near & (tilt > 20)
    held = rows["target"][:, 7].astype(bool)
    weights = np.ones(len(dist), np.float64)
    weights[near] += 0.75
    weights[near & (tilt >= 10)] += 1.5
    weights[tilted & (tilt <= 40)] += 2.0
    weights[tilted & held] += 2.0
    weights[tilted & rows["contact"]] += 1.25
    weights[(dist > 5) & (dist <= 7)] += 0.75
    # Long, nearly stationary sequences have more chances to be sampled.
    _, inverse, counts = np.unique(rows["seed"], return_inverse=True, return_counts=True)
    weights /= np.sqrt(counts[inverse])
    return weights / weights.sum()


def physical_loss(output, target, mean, std):
    physical = output[:, :7] * std + mean
    pred_dist = torch.linalg.vector_norm(physical[:, :3] - physical[:, 3:6], dim=1)
    dist = torch.linalg.vector_norm(target[:, :3] - target[:, 3:6], dim=1)
    tilt, pred_tilt = target[:, 6], physical[:, 6]
    near = dist <= 0.10
    geometry_weight = torch.where(near, 1.0, 0.25)
    distance_loss = F.smooth_l1_loss(pred_dist / 0.02, dist / 0.02, reduction="none")
    tilt_loss = F.smooth_l1_loss(pred_tilt / 5, tilt / 5, reduction="none")

    # The minimum margin is positive only when both geometry thresholds pass.
    margin = torch.minimum((0.05 - pred_dist) / 0.008, (12 - pred_tilt) / 2)
    gate = ((dist <= 0.05) & (tilt <= 12)).float()
    gate_loss = F.binary_cross_entropy_with_logits(margin, gate, reduction="none")
    # Soften labels right at the boundary without discarding those examples.
    certainty = torch.minimum((dist - 0.05).abs() / 0.008, (tilt - 12).abs() / 2)
    gate_weight = near * (0.25 + 0.75 * certainty.clamp(max=1))
    return (
        F.smooth_l1_loss(output[:, :7], (target[:, :7] - mean) / std)
        + 0.3 * F.binary_cross_entropy_with_logits(output[:, 7], target[:, 7])
        + (geometry_weight * (0.6 * distance_loss + tilt_loss)).mean()
        + 0.2 * (gate_weight * gate_loss).mean()
        + 0.08 * F.relu(-pred_tilt).mean()
    )


def validate(estimator, rows):
    true = rows["target"]
    pred = predict(estimator, rows["rgb"], rows["state"], batch_size=64)
    distance = distance_cm(true)
    near = distance <= 10
    groups = {
        "near": near,
        "near_held": near & true[:, 7].astype(bool),
        "near_held_tilt_gt_20": near & (true[:, 6] > 20) & true[:, 7].astype(bool),
        "near_tilt_gt_20": near & (true[:, 6] > 20),
        "rack_contact_tilt_gt_20": near & (true[:, 6] > 20) & rows["contact"],
        "hover_5_7": (distance > 5) & (distance <= 7) & ~rows["contact"],
        "far": distance > 14,
    }
    results = {
        name: metrics(true[mask], pred[mask], rows["seed"][mask]) for name, mask in groups.items()
    }
    results["near_default_gate"] = metrics(
        true[near],
        pred[near],
        rows["seed"][near],
        threshold=100 * DEFAULT_CONFIG["release_distance_m"],
    )
    region = results["near"]
    default_gate = results["near_default_gate"]
    high = results["near_tilt_gt_20"]
    high_held = results["near_held_tilt_gt_20"]
    far = results["far"]
    score = (
        region["distance_mae_cm"] / 4
        + region["tilt_mae_deg"] / 5
        + (high["tilt_mae_deg"] or region["tilt_mae_deg"]) / 10
        + 0.06 * (high_held["tilt_mae_deg"] or 0)
        + (region["gate_fpr"] or 0)
        + 0.75 * (region["gate_fnr"] or 0)
        + 0.20 * ((default_gate["gate_fpr"] or 0) + (default_gate["gate_fnr"] or 0))
        + 0.06 * (far["distance_mae_cm"] or 0)
    )
    return score, results


def finetune(data, init, run, steps, batch_size, val_every, smoke=False, device="cuda"):
    start = monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(22)
    rng = np.random.default_rng(22)
    train_files = partition_files(data / "train", "train")
    dev_files = partition_files(data / "dev", "dev")
    if smoke:
        train_files, dev_files = train_files[:2], dev_files[:2]
    else:
        if len(train_files) != 144 or len(dev_files) != 48:
            raise ValueError("Expected 144 train and 48 development replay episodes")
    if {p.stem for p in train_files} & {p.stem for p in dev_files}:
        raise ValueError("Train and development seed overlap")
    train = load_rows(train_files, ("rgb", "state", "target", "contact"))
    dev = load_rows(dev_files, ("rgb", "state", "target", "contact"))
    weights = sampling_weights(train)
    inherited = torch.load(init, map_location="cpu", weights_only=False)
    anchor_path = data / "anchors.npz"
    anchor_seeds = []
    anchor_rows = 0
    if anchor_path.is_file():
        with np.load(anchor_path, allow_pickle=False) as anchor:
            anchor_seeds = np.unique(anchor["seed"]).tolist()
            original_train = {Path(p).name for p in inherited["train_episodes"]}
            if not anchor_seeds or any(
                not 2_500_000 <= check_seed(seed) < 2_501_000
                or f"ep_{seed}.npz" not in original_train
                for seed in anchor_seeds
            ):
                raise ValueError("Anchor rows must come from the original training split")
            anchor_rows = len(anchor["seed"])
            weights = np.r_[0.85 * weights, np.full(anchor_rows, 0.15 / anchor_rows)]
            for key in ("rgb", "state", "target"):
                train[key] = np.concatenate((train[key], anchor[key]))
    elif not smoke:
        raise ValueError("Full training requires anchors.npz")

    current = Estimator(init, device)
    ema = deepcopy(current)
    ema.model.requires_grad_(False)
    # BN uses the checkpoint's running statistics. Fixed cameras also let us
    # keep the native pixel positions while fine-tuning precise pose estimates.
    current.model.eval()
    backbone = list(current.model.backbone.parameters())
    backbone_ids = {id(p) for p in backbone}
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone, "lr": 2e-5},
            {
                "params": [p for p in current.model.parameters() if id(p) not in backbone_ids],
                "lr": 1e-4,
            },
        ],
        weight_decay=1e-4,
    )
    amp = current.device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", init_scale=1024, enabled=amp)
    ema_parameters = list(ema.model.parameters())
    parameters = list(current.model.parameters())
    stats = current.stats
    mean, std = (
        torch.as_tensor(stats[f"target_{key}"], device=current.device) for key in ("mean", "std")
    )
    rgb = torch.from_numpy(train["rgb"]).to(current.device)
    state = train["state"][:, PROPRIO_DIMS]
    proprio = torch.from_numpy((state - stats["proprio_mean"]) / stats["proprio_std"]).to(
        current.device
    )
    target = torch.from_numpy(train["target"]).to(current.device)
    run.mkdir(parents=True, exist_ok=False)
    manifest = {
        "init_sha256": hashlib.sha256(init.read_bytes()).hexdigest(),
        "proprio_dims": PROPRIO_DIMS,
        "steps": steps,
        "batch_size": batch_size,
        "val_every": val_every,
        "smoke": smoke,
        "seed": 22,
        "train_rows": len(rgb),
        "dev_rows": len(dev["rgb"]),
        "train_episodes": [p.name for p in train_files],
        "dev_episodes": [p.name for p in dev_files],
        "anchor_rows": anchor_rows,
        "anchor_seeds": anchor_seeds,
    }
    (run / "config.json").write_text(json.dumps(manifest, indent=2) + "\n")
    train_episodes = [*inherited["train_episodes"], *manifest["train_episodes"]]
    val_episodes = [*inherited["val_episodes"], *manifest["dev_episodes"]]
    del inherited
    best = float("inf")
    for step in range(steps + 1):
        loss_value = None
        if step:
            indices = torch.as_tensor(
                rng.choice(len(rgb), size=batch_size, p=weights), device=current.device
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=current.device.type, dtype=torch.float16, enabled=amp):
                pred = current.model(rgb[indices], proprio[indices])
            loss = physical_loss(pred.float(), target[indices], mean, std)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                torch._foreach_lerp_(ema_parameters, parameters, max(0.025, 1 / step))
            loss_value = float(loss.detach())
        if step % val_every != 0 and step != steps:
            continue
        score, results = validate(ema, dev)
        selected = step > 0 and score < best
        record = {
            "step": step,
            "loss": loss_value,
            "seconds": monotonic() - start,
            "score": score,
            "selected": selected,
            "dev": results,
        }
        print(json.dumps(record, allow_nan=False), flush=True)
        with (run / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        if selected:
            best = score
            saved = {
                "version": 1,
                "config": manifest,
                "stats": stats,
                "model": ema.model.state_dict(),
                "step": step,
                "val_metrics": results,
                "train_episodes": train_episodes,
                "val_episodes": val_episodes,
            }
            torch.save(saved, run / "checkpoint.tmp")
            os.replace(run / "checkpoint.tmp", run / "last.pt")
    return {"seconds": monotonic() - start, "best_score": best}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-every", type=int, default=250)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.val_every) < 1:
        parser.error("Steps, batch size and evaluation interval must be positive")
    print(finetune(**vars(args)), flush=True)


if __name__ == "__main__":
    main()
