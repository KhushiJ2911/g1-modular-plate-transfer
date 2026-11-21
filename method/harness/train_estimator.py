"""Train plate/site/tilt regression and held BCE on episode-disjoint observations."""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from method.bc.dataset import FRAME_SHAPE, EpisodeStore, episode_files
from method.bc.train import atomic_save
from method.harness.data import LABEL_KEYS, audit_files
from method.harness.estimator import PlateEstimator
from submission.model import PROPRIO_DIMS


class EstimatorDataset(Dataset):
    def __init__(self, store, episode_ids, target, stats, row_stride=1, max_rows=None):
        self.store, self.stats, self.target = store, stats, target
        # Supervise BOTH odd/even control steps using image t//2. Episode splits
        # keep repeated images out of the other split.
        self.rows = np.concatenate(
            [np.arange(store.steps[i], store.steps[i + 1], row_stride) for i in episode_ids]
        )
        if max_rows:
            self.rows = self.rows[
                np.linspace(0, len(self.rows) - 1, min(max_rows, len(self.rows)), dtype=int)
            ]
        self.images = None

    def __len__(self):
        return len(self.rows)

    def __getstate__(self):
        return {**self.__dict__, "images": None}

    def __getitem__(self, index):
        if self.images is None:
            self.images = np.memmap(
                self.store.image_path,
                mode="r",
                dtype=np.uint8,
                shape=(int(self.store.frames[-1]), *FRAME_SHAPE),
            )
        row = self.rows[index]
        ep = int(np.searchsorted(self.store.steps, row, side="right") - 1)
        t = row - self.store.steps[ep]
        proprio = self.store.state[row, list(PROPRIO_DIMS)]
        normalized = self.target[row].copy()
        normalized[:7] = (normalized[:7] - self.stats["target_mean"]) / self.stats["target_std"]
        return {
            "rgb": self.images[self.store.frames[ep] + t // 2].copy(),
            "proprio": ((proprio - self.stats["proprio_mean"]) / self.stats["proprio_std"]).astype(
                np.float32
            ),
            "target": normalized,
            "physical": self.target[row],
        }


def prepare_data(args):
    files = episode_files(args.data_dirs)
    audit_files(files)
    store = EpisodeStore(args.data_dirs, args.cache_dir, val_episodes=args.val_episodes)
    if not all(store.priv_available[key].all() for key in LABEL_KEYS):
        raise ValueError("Every estimator episode needs all four priv_* labels; label first")
    target = np.column_stack([store.priv[key] for key in LABEL_KEYS]).astype(np.float32)
    rows = np.concatenate([np.arange(store.steps[i], store.steps[i + 1]) for i in store.train_ids])
    values = target[rows, :7].astype(np.float64)
    stats = {key: store.stats[key] for key in ("proprio_mean", "proprio_std")}
    stats.update(
        target_mean=values.mean(0).astype(np.float32),
        target_std=np.maximum(values.std(0), 1e-3).astype(np.float32),
    )
    train = EstimatorDataset(
        store, store.train_ids, target, stats, args.row_stride, args.max_train_rows
    )
    val = EstimatorDataset(store, store.val_ids, target, stats, args.row_stride, args.max_val_rows)
    return store, stats, train, val


def to_device(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.inference_mode()
def validate(model, loader, device, stats):
    model.eval()
    totals = {key: np.zeros(6, np.float64) for key in ("all", "release_6cm")}
    mean = torch.as_tensor(stats["target_mean"], device=device)
    std = torch.as_tensor(stats["target_std"], device=device)
    for batch in loader:
        batch = to_device(batch, device)
        output = model(batch["rgb"], batch["proprio"]).float()
        pred, true = output[:, :7] * std + mean, batch["physical"]
        distance = torch.linalg.vector_norm(true[:, :3] - true[:, 3:6], dim=1)
        pred_distance = torch.linalg.vector_norm(pred[:, :3] - pred[:, 3:6], dim=1)
        errors = torch.stack(
            (
                torch.linalg.vector_norm(pred[:, :3] - true[:, :3], dim=1) * 100,
                torch.linalg.vector_norm(pred[:, 3:6] - true[:, 3:6], dim=1) * 100,
                (pred_distance - distance).abs() * 100,
                (pred[:, 6] - true[:, 6]).abs(),
                ((output[:, 7] >= 0) == true[:, 7].bool()).float(),
            ),
            dim=1,
        )
        for key, mask in (
            ("all", torch.ones_like(distance, dtype=torch.bool)),
            ("release_6cm", distance <= 0.06),
        ):
            totals[key][:5] += errors[mask].sum(0).cpu().numpy()
            totals[key][5] += int(mask.sum())
    names = ("plate_error_cm", "site_error_cm", "distance_mae_cm", "tilt_mae_deg", "held_accuracy")
    metrics = {}
    for key, sums in totals.items():
        metrics[key] = dict(
            zip(names, (sums[:5] / sums[5]).tolist() if sums[5] else [None] * 5, strict=True)
        )
        metrics[key]["count"] = int(sums[5])
    return metrics


def train(args):
    if (
        min(
            args.steps,
            args.batch_size,
            args.microbatch_size,
            args.val_every,
            args.row_stride,
            args.val_episodes,
        )
        < 1
        or not 0 <= args.num_workers <= 2
    ):
        raise ValueError("Positive steps/batches/cadences required; workers must be 0..2")
    if any(x is not None and x < 1 for x in (args.max_train_rows, args.max_val_rows)):
        raise ValueError("Smoke row caps must be positive")
    if not np.isfinite([args.lr, args.backbone_lr]).all() or min(args.lr, args.backbone_lr) <= 0:
        raise ValueError("Learning rates must be positive and finite")
    run = Path(args.run_dir)
    if run.exists():
        raise FileExistsError(f"Choose a fresh --run-dir: {run}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    cuda = device.type == "cuda"
    store, stats, training, validation = prepare_data(args)
    run.mkdir(parents=True, exist_ok=False)
    config = {**vars(args), "proprio_dims": PROPRIO_DIMS}
    (run / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    model = PlateEstimator(pretrained=args.pretrained).to(device)
    backbone = list(model.backbone.parameters())
    backbone_ids = {id(p) for p in backbone}
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone, "lr": args.backbone_lr},
            {"params": [p for p in model.parameters() if id(p) not in backbone_ids], "lr": args.lr},
        ],
        weight_decay=1e-4,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    options = dict(num_workers=args.num_workers, pin_memory=cuda)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        training, batch_size=args.batch_size, shuffle=True, generator=generator, **options
    )
    val_loader = DataLoader(validation, batch_size=args.microbatch_size, **options)
    batches = iter(train_loader)
    for step in range(1, args.steps + 1):
        try:
            batch = next(batches)
        except StopIteration:
            batches = iter(train_loader)
            batch = next(batches)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for start in range(0, len(batch["rgb"]), args.microbatch_size):
            micro = to_device(
                {k: v[start : start + args.microbatch_size] for k, v in batch.items()}, device
            )
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=cuda):
                pred = model(micro["rgb"], micro["proprio"]).float()
                target = micro["target"]
                loss = F.smooth_l1_loss(
                    pred[:, :7], target[:, :7]
                ) + F.binary_cross_entropy_with_logits(pred[:, 7], target[:, 7])
                loss = loss * len(pred) / len(batch["rgb"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite estimator loss")
            scaler.scale(loss).backward()
            loss_sum += float(loss.detach())
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        if step % args.val_every == 0 or step == args.steps:
            metrics = validate(model, val_loader, device, stats)
            record = {"step": step, "train_loss": loss_sum, "val": metrics}
            print(json.dumps(record, allow_nan=False), flush=True)
            with (run / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            atomic_save(
                {
                    "version": 1,
                    "config": config,
                    "stats": stats,
                    "model": model.state_dict(),
                    "step": step,
                    "val_metrics": metrics,
                    "data_digest": store.digest,
                    "train_episodes": [store.files[i] for i in store.train_ids],
                    "val_episodes": [store.files[i] for i in store.val_ids],
                },
                run / "last.pt",
            )
    return run / "last.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dirs", nargs="+", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--run-dir", required=True, help="New directory; refuses existing runs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--microbatch-size", type=int, default=8)
    parser.add_argument("--val-episodes", type=int, default=6, help="Per source directory")
    parser.add_argument("--val-every", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--row-stride", type=int, default=1)
    parser.add_argument("--max-train-rows", type=int, help="Smoke only: cap sampled training rows")
    parser.add_argument("--max-val-rows", type=int, help="Smoke only: cap validation rows")
    args = parser.parse_args()
    os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
    train(args)


if __name__ == "__main__":
    main()
