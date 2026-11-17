"""Train BC in the model environment; run from the repo root with python -m."""

import argparse
import ast
import copy
import csv
import json
import math
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from .dataset import EpisodeStore
from .model import BCModel, masked_l1


@dataclass
class Config:
    data_dirs: tuple = ("/home/keyu/advaitic/data/human60",)
    cache_dir: str = "/home/keyu/advaitic/data/cache"
    run_dir: str = "method/bc/checkpoints/human60"
    horizon: int = 20
    exec_steps: int = 10
    steps: int = 20_000
    batch_size: int = 64
    microbatch_size: int = 8
    val_episodes: int = 6
    val_every: int = 1000
    save_every: int = 1000
    log_every: int = 100
    lr: float = 3e-4
    backbone_lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    ema_decay: float = 0.999
    num_workers: int = 2
    seed: int = 42
    device: str = "cuda"
    pretrained: bool = True
    temporal_ensemble: bool = False
    ensemble_decay: float = 0.01


def validate_config(config):
    if not 1 <= config.exec_steps <= config.horizon <= 50:
        raise ValueError("Require 1 <= exec_steps <= horizon <= 50")
    if not 0 <= config.num_workers <= 2:
        raise ValueError("num_workers must be in [0, 2]")
    for key in (
        "steps",
        "batch_size",
        "microbatch_size",
        "val_episodes",
        "val_every",
        "save_every",
        "log_every",
    ):
        if getattr(config, key) < 1:
            raise ValueError(f"{key} must be positive")
    if config.warmup_steps < 0 or not 0 <= config.ema_decay < 1:
        raise ValueError("Invalid warmup_steps or ema_decay")
    if config.lr <= 0 or config.backbone_lr <= 0 or config.weight_decay < 0:
        raise ValueError("Invalid optimizer settings")
    if config.ensemble_decay < 0:
        raise ValueError("ensemble_decay must be nonnegative")


class StepBatches(Sampler):
    """Reproducible random batches independent of worker prefetch and restarts."""

    def __init__(self, size, batch_size, start, stop, seed):
        self.size, self.batch_size = size, batch_size
        self.start, self.stop, self.seed = start, stop, seed

    def __iter__(self):
        for step in range(self.start, self.stop):
            generator = torch.Generator().manual_seed(self.seed + step)
            yield torch.randint(self.size, (self.batch_size,), generator=generator).tolist()

    def __len__(self):
        return self.stop - self.start


def atomic_save(value, path):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def rng_state(cuda):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if cuda else None,
    }


def restore_rng(state, cuda):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if cuda and state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def lr_factor(step, config):
    if step < config.warmup_steps:
        return (step + 1) / config.warmup_steps
    progress = (step - config.warmup_steps) / max(1, config.steps - config.warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1)))


@torch.no_grad()
def update_ema(ema, model, decay):
    for average, current in zip(ema.parameters(), model.parameters(), strict=True):
        average.lerp_(current.detach(), 1 - decay)
    for average, current in zip(ema.buffers(), model.buffers(), strict=True):
        average.copy_(current)


def to_device(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.inference_mode()
def validate(model, loader, device, cuda):
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=cuda):
            prediction = model(batch["rgb"], batch["proprio"])
            loss = masked_l1(prediction.float(), batch["target"], batch["pad_mask"])
        valid = int((~batch["pad_mask"]).sum()) * prediction.shape[-1]
        total += float(loss) * valid
        count += valid
    return total / count


def train(config, *, stop_after=None):
    """stop_after tests interruption without changing the total LR schedule."""
    validate_config(config)
    device = torch.device(config.device)
    cuda = device.type == "cuda"
    if cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --device cpu for a CPU run")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    run_dir = Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "last.pt"
    previous = (
        torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint_path.exists()
        else None
    )
    store = EpisodeStore(config.data_dirs, config.cache_dir, config.val_episodes)
    if previous is not None:
        allowed_changes = {"run_dir", "cache_dir", "num_workers", "device"}
        for key, value in asdict(config).items():
            if key not in allowed_changes and value != previous["config"][key]:
                raise ValueError(f"Resume config changed: {key}; use a new run directory")
        if previous["data_digest"] != store.digest:
            raise ValueError("Resume data changed; use a new run directory")
    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    model = BCModel(config.horizon, pretrained=config.pretrained and previous is None).to(device)
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": config.backbone_lr},
            {
                "params": [p for p in model.parameters() if id(p) not in backbone_ids],
                "lr": config.lr,
            },
        ],
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    step = 0
    last_val = float("nan")
    if previous is not None:
        model.load_state_dict(previous["model"])
        ema.load_state_dict(previous["ema"])
        optimizer.load_state_dict(previous["optimizer"])
        scaler.load_state_dict(previous["scaler"])
        step = previous["step"]
        last_val = previous["val_loss"]
        restore_rng(previous["rng"], cuda)
        print(f"Resumed {checkpoint_path} at step {step}", flush=True)
    del previous
    train_data, val_data = (
        store.dataset("train", config.horizon),
        store.dataset("val", config.horizon),
    )
    stop = config.steps if stop_after is None else min(config.steps, stop_after)
    if stop <= step:
        print("Training already complete.", flush=True)
        return checkpoint_path
    # Dedicated loader generators preserve dropout/augmentation RNG on resume.
    loader_options = {"num_workers": config.num_workers, "pin_memory": cuda}
    train_loader = DataLoader(
        train_data,
        batch_sampler=StepBatches(len(train_data), config.batch_size, step, stop, config.seed),
        generator=torch.Generator().manual_seed(config.seed),
        **loader_options,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config.microbatch_size,
        generator=torch.Generator().manual_seed(config.seed),
        **loader_options,
    )
    log_path = run_dir / "log.csv"
    print(
        f"{len(store.train_ids)} train / {len(store.val_ids)} val episodes; "
        f"{len(train_data)} train frames; cache={store.image_path}",
        flush=True,
    )
    started, logged_step = time.monotonic(), step
    train_total, train_count = 0.0, 0
    with log_path.open("a", newline="") as log:
        writer = csv.writer(log)
        if log.tell() == 0:
            writer.writerow(["step", "train_loss", "val_loss", "lr", "it/s"])
            log.flush()
        model.train()
        for batch in train_loader:
            factor = lr_factor(step, config)
            for group, base_lr in zip(
                optimizer.param_groups, (config.backbone_lr, config.lr), strict=True
            ):
                group["lr"] = base_lr * factor
            optimizer.zero_grad(set_to_none=True)
            valid_count = (~batch["pad_mask"]).sum().item() * 10
            batch_loss = 0.0
            # Only microbatch_size * 3 images occupy backbone activation RAM.
            # Weight by valid elements to preserve the logical batch-64 mean.
            for start in range(0, len(batch["rgb"]), config.microbatch_size):
                micro = to_device(
                    {k: v[start : start + config.microbatch_size] for k, v in batch.items()}, device
                )
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=cuda):
                    prediction = model(micro["rgb"], micro["proprio"])
                    loss = masked_l1(prediction.float(), micro["target"], micro["pad_mask"])
                    fraction = (~micro["pad_mask"]).sum().item() * 10 / valid_count
                    loss = loss * fraction
                scaler.scale(loss).backward()
                batch_loss += float(loss.detach())
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not cuda and not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite CPU gradient")
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                update_ema(ema, model, config.ema_decay)
            step += 1
            train_total += batch_loss
            train_count += 1
            check = step % config.save_every == 0 or step == stop
            evaluate = step % config.val_every == 0 or check
            if evaluate:
                last_val = validate(ema, val_loader, device, cuda)
            if step % config.log_every == 0 or evaluate:
                if cuda:
                    torch.cuda.synchronize(device)
                elapsed = time.monotonic() - started
                throughput = (step - logged_step) / max(elapsed, 1e-9)
                train_loss = train_total / train_count
                writer.writerow(
                    [step, train_loss, last_val, optimizer.param_groups[1]["lr"], throughput]
                )
                log.flush()
                print(
                    f"step={step} train={train_loss:.5f} val={last_val:.5f} "
                    f"lr={optimizer.param_groups[1]['lr']:.3g} it/s={throughput:.2f}",
                    flush=True,
                )
                train_total, train_count = 0.0, 0
                started, logged_step = time.monotonic(), step
            if check:
                atomic_save(
                    {
                        "version": 1,
                        "config": asdict(config),
                        "model": model.state_dict(),
                        "ema": ema.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
                        "step": step,
                        "rng": rng_state(cuda),
                        "stats": store.stats,
                        "data_digest": store.digest,
                        "train_episodes": [store.files[i] for i in store.train_ids],
                        "val_episodes": [store.files[i] for i in store.val_ids],
                        "val_loss": last_val,
                    },
                    checkpoint_path,
                )
    peak = torch.cuda.max_memory_allocated(device) / 1e9 if cuda else 0.0
    reserved = torch.cuda.max_memory_reserved(device) / 1e9 if cuda else 0.0
    print(f"Peak CUDA memory: allocated={peak:.3f} GB, reserved={reserved:.3f} GB", flush=True)
    if cuda and reserved >= 3.0:
        raise RuntimeError("Memory budget exceeded; lower --microbatch-size in a new run")
    return checkpoint_path


def read_config(path):
    """Flat YAML (scalars/inline lists) or JSON; no PyYAML dependency."""
    text = Path(path).read_text()
    if text.lstrip().startswith("{"):
        return json.loads(text)
    result = {}
    for number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition(":")
        if not separator or not value.strip():
            raise ValueError(f"Config line {number}: expected key: scalar or inline list")
        value = value.strip()
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                parsed = value
        result[key.strip().replace("-", "_")] = parsed
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    for field in fields(Config):
        name = "--" + field.name.replace("_", "-")
        options = {"default": argparse.SUPPRESS}
        if field.name == "data_dirs":
            options["nargs"] = "+"
        elif field.type is bool:
            options["action"] = argparse.BooleanOptionalAction
        else:
            options["type"] = field.type
        parser.add_argument(name, **options)
    args = vars(parser.parse_args())
    config_path = args.pop("config")
    values = read_config(config_path) if config_path else {}
    values.update(args)
    if "data_dirs" in values:
        if not isinstance(values["data_dirs"], (list, tuple)):
            parser.error("data_dirs must be an inline list or --data-dirs DIR [DIR ...]")
        values["data_dirs"] = tuple(values["data_dirs"])
    train(Config(**values))


if __name__ == "__main__":
    main()
