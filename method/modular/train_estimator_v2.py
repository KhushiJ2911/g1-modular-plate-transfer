"""Fine-tune v1b with reset context, replay labels and masked historical teacher samples."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from submission.estimator_v2 import PlateEstimatorV2, rotation_matrix
from submission.model import PROPRIO_DIMS

PHASES = (
    "APPROACH",
    "PREGRASP",
    "GRASP",
    "LIFT",
    "CARRY",
    "ALIGN",
    "LOWER",
    "RELEASE",
    "SETTLE",
    "RETREAT",
    "HOLD",
    "teacher",
)
# Sample phases explicitly so failed episodes in absorbing HOLD do not dominate.
WEIGHTS = np.array([10, 10, 10, 8, 10, 10, 14, 5, 5, 2, 2, 14], dtype=float)


class Frames:
    def __init__(self, paths, device):
        images, states, truth, baseline, phase, episode, first = [], [], [], [], [], [], []
        self.seeds = []
        offset = 0
        for i, path in enumerate(paths):
            with np.load(path) as f:
                meta = json.loads(str(f["meta"]))
                seed = meta["seed"]
                if seed != int(path.stem) or not (
                    3100000 <= seed < 3200000 or 2200000 <= seed < 2201000
                ):
                    raise ValueError(f"Invalid training source {path}")
                y = f["truth"]
                n = len(y)
                if y.shape != (n, 18) or f["step"][0] != 0:
                    raise ValueError(f"Invalid episode {path}")
                first.append(offset)
                offset += n
                self.seeds.append(seed)
                images.append(f["rgb"])
                states.append(f["state"])
                truth.append(y)
                p = f["phase"] if "phase" in f else ["teacher"] * n
                phase.extend(PHASES.index(x) for x in p)
                baseline.append(f["baseline"] if "baseline" in f else np.full_like(y, np.nan))
                episode.extend([i] * n)
        # All images stay resident on GPU. Gather uint8 batches, transferring only once.
        self.rgb = torch.from_numpy(np.concatenate(images)).to(device)
        self.state = torch.from_numpy(np.concatenate(states)).to(device)
        self.target = torch.from_numpy(np.concatenate(truth)).to(device)
        self.baseline = np.concatenate(baseline)
        self.phase = np.asarray(phase)
        self.episode = np.asarray(episode)
        self.first = torch.tensor(np.asarray(first)[self.episode], device=device)
        weights = np.zeros(len(self.phase))
        for i, w in enumerate(WEIGHTS):
            mask = self.phase == i
            if mask.any():
                weights[mask] = w / mask.sum()
        self.weights = torch.tensor(weights / weights.sum(), device=device, dtype=torch.float32)


def forward(model, ds, idx):
    first, inv = ds.first[idx].unique(return_inverse=True)
    z = model.encode(ds.rgb[idx], ds.state[idx])
    start = model.encode(ds.rgb[first], ds.state[first])
    return model.predict(z, start[inv], ds.state[idx], ds.state[first][inv])


def loss(pred, labels, pos_weight):
    fresh = torch.isfinite(labels[:, 8])
    y = labels.nan_to_num()

    def huber(delta):
        return F.smooth_l1_loss(delta, torch.zeros_like(delta))

    value = huber((pred[:, :3] - y[:, :3]) / 0.015) + 0.25 * huber((pred[:, 6] - y[:, 6]) / 4.0)
    value += 0.3 * F.binary_cross_entropy_with_logits(pred[:, 7], y[:, 7])
    if fresh.any():
        p, y = pred[fresh], y[fresh]
        actual = rotation_matrix(y[:, 8:14])
        est = rotation_matrix(p[:, 8:14])
        value += 0.4 * huber((p[:, 3:6] - y[:, 3:6]) / 0.012)
        value += 0.22 * huber((est - actual) / 0.08) + 0.28 * huber(
            (est[:, :, 2] - actual[:, :, 2]) / 0.06
        )
        value += 0.5 * F.binary_cross_entropy_with_logits(
            p[:, 14:18], y[:, 14:18], pos_weight=pos_weight
        )
    return value


@torch.inference_mode()
def evaluate(model, ds, batch=96):
    model.eval()
    parts = []
    for idx in torch.arange(len(ds.phase), device=ds.state.device).split(batch):
        p = forward(model, ds, idx)
        p[:, 7] = p[:, 7].sigmoid()
        p[:, 14:18] = p[:, 14:18].sigmoid()
        parts.append(p.cpu().numpy())
    return np.concatenate(parts)


def metrics(pred, ds):
    truth = ds.target.cpu().numpy()

    def rot(y):
        return np.cross(y[:, 8:11], y[:, 11:14])

    normal = np.degrees(np.arccos(np.clip((rot(pred) * rot(truth)).sum(1), -1, 1)))
    trace = (pred[:, 8:14] * truth[:, 8:14]).sum(1) + (rot(pred) * rot(truth)).sum(1)
    geodesic = np.degrees(np.arccos(np.clip((trace - 1) / 2, -1, 1)))
    delta = 100 * (pred[:, :3] - truth[:, :3])
    result = {}
    for name in ("all", *PHASES[:-1]):
        mask = np.ones(len(delta), bool) if name == "all" else ds.phase == PHASES.index(name)
        if not mask.any():
            continue
        groups = [delta[mask & (ds.episode == e)] for e in np.unique(ds.episode[mask])]
        # The lead's convention averages the magnitude of each episode mean and
        # the norm of its three within-episode standard deviations.
        d = dict(
            episodes=len(groups),
            frames=int(mask.sum()),
            episode_bias_cm=float(np.mean([np.linalg.norm(g.mean(0)) for g in groups])),
            noise_cm=float(np.mean([np.linalg.norm(g.std(0)) for g in groups])),
            global_bias_cm=delta[mask].mean(0).tolist(),
            plate_mae_cm=float(np.linalg.norm(delta[mask], axis=1).mean()),
            plate_p95_cm=float(np.percentile(np.linalg.norm(delta[mask], axis=1), 95)),
            site_mae_cm=float(
                100 * np.linalg.norm(pred[mask, 3:6] - truth[mask, 3:6], axis=1).mean()
            ),
            tilt_mae_deg=float(np.abs(pred[mask, 6] - truth[mask, 6]).mean()),
            normal_mae_deg=float(normal[mask].mean()),
            rotation_mae_deg=float(geodesic[mask].mean()),
            held_error=float(np.mean((pred[mask, 7] >= 0.5) != truth[mask, 7])),
        )
        for i, k in enumerate(("source_rack", "target_rack", "table", "floor"), 14):
            a = pred[mask, i] >= 0.5
            b = truth[mask, i] > 0
            d[k] = {
                "tp": int((a & b).sum()),
                "fn": int((~a & b).sum()),
                "fp": int((a & ~b).sum()),
                "tn": int((~a & ~b).sum()),
            }
        result[name] = d
    return result


def train(directory, out, base, start, count, steps, batch, smoke=False):
    torch.manual_seed(421)
    np.random.seed(421)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda"
    fresh = [directory / "onpolicy" / f"{seed}.npz" for seed in range(start, start + count)]
    if any(not f.is_file() for f in fresh):
        raise ValueError("Incomplete generation block")
    teacher = sorted((directory / "teacher").glob("*.npz"))
    if len(teacher) != 60:
        raise ValueError("Expected the disclosed 60 teacher episodes")
    train_files = [f for f in fresh if int(f.stem) % 10 >= 2] + teacher
    tune_files = [f for f in fresh if int(f.stem) % 10 == 1]
    test_files = [f for f in fresh if int(f.stem) % 10 == 0]
    if smoke:
        # IO/gradient plumbing only; no accuracy claims from this single episode.
        train_files = fresh[:1] + teacher[:1]
        tune_files = test_files = fresh[:1]
    elif (
        min(len(train_files), len(tune_files), len(test_files)) == 0
        or set(tune_files) & set(test_files)
        or set(train_files) & set(tune_files + test_files)
    ):
        raise ValueError("Need three disjoint episode splits")
    ds, val, test = [Frames(paths, device) for paths in (train_files, tune_files, test_files)]
    model = PlateEstimatorV2().to(device)
    saved = torch.load(base, map_location="cpu", weights_only=False)
    missing, extra = model.load_state_dict(saved["model"], strict=False)
    if extra or any(
        k.startswith(("backbone", "head", "proprio.", "spatial", "camera_embedding"))
        for k in missing
    ):
        raise ValueError("Incompatible v1b encoder")
    for k, v in saved["stats"].items():
        if hasattr(model, k):
            getattr(model, k).copy_(torch.as_tensor(v))
    for layer in model.backbone[:-1]:
        layer.requires_grad_(False)
    slow = list(model.backbone[-1].parameters()) + list(model.spatial.parameters())
    slow_ids = {id(x) for x in slow}
    fast = [p for p in model.parameters() if p.requires_grad and id(p) not in slow_ids]
    optimizer = torch.optim.AdamW(
        [{"params": slow, "lr": 1e-5}, {"params": fast, "lr": 2e-4}], weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-6)
    contacts = ds.target[torch.isfinite(ds.target[:, 8]), 14:18]
    rates = contacts.mean(0)
    pos_weight = ((1 - rates) / rates.clamp_min(0.001)).sqrt().clamp(1, 6)
    out.mkdir(parents=True, exist_ok=True)
    history = []
    best = float("inf")
    t0 = time.monotonic()
    for i in range(1, steps + 1):
        model.train()
        idx = torch.multinomial(ds.weights, batch, replacement=True)
        optimizer.zero_grad(set_to_none=True)
        p = forward(model, ds, idx)
        value = loss(p, ds.target[idx], pos_weight)
        if not torch.isfinite(value):
            raise ValueError("Nonfinite training objective")
        value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        if i % 300 == 0 or i == steps:
            score = metrics(evaluate(model, val), val)
            overall = score["all"]
            key = (
                overall["plate_mae_cm"]
                + 0.3 * overall["site_mae_cm"]
                + 0.08 * overall["normal_mae_deg"]
            )
            history.append(
                dict(step=i, loss=float(value.detach()), selection=key, validation=overall)
            )
            print(i, round(key, 4), flush=True)
            if key < best:
                best = key
                payload = dict(
                    version=2,
                    model=model.state_dict(),
                    config=dict(proprio_dims=PROPRIO_DIMS),
                    base_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
                    step=i,
                )
                torch.save(payload, out / "next.pt")
                os.replace(out / "next.pt", out / "last.pt")
    ck = torch.load(out / "last.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    predictions = evaluate(model, test)
    report = dict(
        train=ds.seeds,
        tune=val.seeds,
        test=test.seeds,
        train_frames=len(ds.phase),
        teacher_frames=int(np.sum(ds.phase == PHASES.index("teacher"))),
        test_frames=len(test.phase),
        gpu=torch.cuda.get_device_name(),
        seconds=time.monotonic() - t0,
        step=ck["step"],
        history=history,
        v1b=metrics(test.baseline, test),
        v2=metrics(predictions, test),
        sha256=hashlib.sha256((out / "last.pt").read_bytes()).hexdigest(),
    )
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return report["sha256"]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("base", type=Path)
    p.add_argument("--start", type=int, default=3100000)
    p.add_argument("--count", type=int, default=300)
    p.add_argument("--steps", type=int, default=1800)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    print(train(a.directory, a.out, a.base, a.start, a.count, a.steps, a.batch, a.smoke))
