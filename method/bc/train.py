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
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler

from .dataset import EpisodeStore
from .model import (
    PROPRIO_DIMS,
    BCModel,
    action_loss_terms,
    action_masks,
    auxiliary_loss,
    load_bc_weights,
    validate_gripper_config,
    validate_head_config,
    validate_proprio_dims,
)


@dataclass
class Config:
    data_dirs: tuple = ("/home/keyu/advaitic/data/human60",)
    data_weights: tuple = ()
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
    aux_weight: float = 0.1
    proprio_dims: tuple = PROPRIO_DIMS
    head_type: str = "mlp"
    head_blocks: int = 3
    fuse_skip: bool = False
    gripper_loss: str = "l1"
    gripper_weight: float = 1.0
    gripper_pos_weight: float = 1.0
    gripper_threshold: float = 0.5


def validate_config(config):
    validate_proprio_dims(config.proprio_dims)
    validate_head_config(config.head_type, config.head_blocks)
    validate_gripper_config(
        config.gripper_loss,
        config.gripper_weight,
        config.gripper_pos_weight,
        config.gripper_threshold,
    )
    if not math.isfinite(config.aux_weight) or config.aux_weight < 0:
        raise ValueError("aux_weight must be finite and nonnegative")
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

    def __init__(self, weights, batch_size, start, stop, seed):
        self.weights, self.batch_size = torch.as_tensor(weights, dtype=torch.double), batch_size
        self.start, self.stop, self.seed = start, stop, seed

    def __iter__(self):
        for step in range(self.start, self.stop):
            generator = torch.Generator().manual_seed(self.seed + step)
            yield list(
                WeightedRandomSampler(
                    self.weights, self.batch_size, replacement=True, generator=generator
                )
            )

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


def loss_terms(prediction, batch, config):
    return action_loss_terms(
        prediction.float(),
        batch["target"],
        batch["pad_mask"],
        batch["label_mask"],
        gripper_loss=config.gripper_loss,
        gripper_pos_weight=config.gripper_pos_weight,
        gripper_threshold=config.gripper_threshold,
    )


class ActionMetrics:
    """Aggregate sums before dividing, so partial labels have the right weight."""

    def __init__(self):
        self.totals = {}

    def add(self, terms):
        for key, value in terms.items():
            self.totals[key] = self.totals.get(key, 0.0) + float(value.detach())

    def means(self):
        def ratio(numerator, denominator, empty=0.0):
            count = self.totals.get(denominator, 0)
            return self.totals.get(numerator, 0) / count if count else empty

        return {
            "l1": ratio("l1_sum", "l1_count"),
            "gripper_bce": ratio("gripper_bce_sum", "gripper_count"),
            "gripper_accuracy": ratio("gripper_correct", "gripper_count", float("nan")),
            "gripper_open_accuracy": ratio(
                "gripper_open_correct", "gripper_open_count", float("nan")
            ),
        }


def prepare_log(path, columns):
    """Extend legacy CSV headers on resume while retaining their existing rows."""
    if not path.exists() or not path.stat().st_size:
        return
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames == columns:
            return
        rows = list(reader)
    with tempfile.NamedTemporaryFile(mode="w", newline="", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


@torch.inference_mode()
def validate(model, loader, device, cuda, config):
    model.eval()
    metrics = ActionMetrics()
    aux_total, aux_count = 0.0, 0
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=cuda):
            prediction, aux = model(batch["rgb"], batch["proprio"], return_aux=True)
            metrics.add(loss_terms(prediction, batch, config))
            if aux is not None:
                labeled = int(batch["aux_mask"].sum())
                aux_total += (
                    float(auxiliary_loss(aux.float(), batch["aux_target"], batch["aux_mask"]))
                    * labeled
                )
                aux_count += labeled
    means = metrics.means()
    action_loss = means["l1"] + config.gripper_weight * means["gripper_bce"]
    return action_loss, aux_total / aux_count if aux_count else 0.0, means


@torch.no_grad()
def initialize_from(model, path, stats):
    """Start from EMA weights, preserving their meaning under new normalization."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    source = checkpoint["config"]
    if (
        source["horizon"] != model.horizon
        or tuple(source.get("proprio_dims", PROPRIO_DIMS)) != model.proprio_dims
    ):
        raise ValueError("--init-from requires matching horizon and proprio_dims")
    load_bc_weights(model, checkpoint["ema"])
    old = checkpoint["stats"]
    layer = model.proprio[0]
    mean_shift = torch.as_tensor(
        (stats["proprio_mean"] - old["proprio_mean"]) / old["proprio_std"],
        device=layer.weight.device,
    )
    scale = torch.as_tensor(stats["proprio_std"] / old["proprio_std"], device=layer.weight.device)
    layer.bias.add_(layer.weight @ mean_shift)
    layer.weight.mul_(scale)
    if model.aux_head is not None and "aux_head.weight" in checkpoint["ema"]:
        # New normalized predictions correspond to the same physical labels.
        layer = model.aux_head
        scale = torch.as_tensor(old["aux_std"] / stats["aux_std"], device=layer.weight.device)
        shift = torch.as_tensor(
            (old["aux_mean"] - stats["aux_mean"]) / stats["aux_std"], device=layer.weight.device
        )
        layer.weight.mul_(scale[:, None])
        layer.bias.mul_(scale).add_(shift)


def train(config, *, stop_after=None, init_from=None):
    """stop_after tests interruption without changing the total LR schedule."""
    validate_config(config)
    torch.backends.cudnn.benchmark = True
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
    if previous is not None and init_from is not None:
        raise ValueError("--init-from requires a new run directory; omit it to resume")
    init_source = str(init_from) if init_from is not None else None
    if previous is not None:
        init_source = previous.get("init_from")
    store = EpisodeStore(
        config.data_dirs, config.cache_dir, config.val_episodes, config.proprio_dims
    )
    train_data, val_data = (
        store.dataset("train", config.horizon),
        store.dataset("val", config.horizon),
    )
    sample_weights = train_data.sample_weights(config.data_weights)
    if previous is not None:
        if previous.get("version") != 2:
            raise ValueError("Legacy training checkpoint: use --init-from in a new run directory")
        allowed_changes = {"run_dir", "cache_dir", "num_workers", "device"}
        # Fill in defaults for fields added since the first version-2 checkpoints.
        defaults = Config()
        old_config = {
            key: getattr(defaults, key)
            for key in (
                "head_type",
                "head_blocks",
                "fuse_skip",
                "gripper_loss",
                "gripper_weight",
                "gripper_pos_weight",
                "gripper_threshold",
            )
        }
        old_config.update(previous["config"])
        for key, value in asdict(config).items():
            if key not in allowed_changes and value != old_config[key]:
                raise ValueError(f"Resume config changed: {key}; use a new run directory")
        if previous["data_digest"] != store.digest:
            raise ValueError("Resume data changed; use a new run directory")
    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    model = BCModel(
        config.horizon,
        pretrained=config.pretrained and previous is None and init_from is None,
        proprio_dims=config.proprio_dims,
        aux_head=config.aux_weight > 0,
        head_type=config.head_type,
        head_blocks=config.head_blocks,
        fuse_skip=config.fuse_skip,
    ).to(device)
    if init_from is not None:
        initialize_from(model, init_from, store.stats)
        print(f"Initialized EMA weights from {init_from}; fresh optimizer/step", flush=True)
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
    last_val_aux = float("nan")
    last_val_metrics = dict.fromkeys(ActionMetrics().means(), float("nan"))
    if previous is not None:
        model.load_state_dict(previous["model"])
        ema.load_state_dict(previous["ema"])
        optimizer.load_state_dict(previous["optimizer"])
        scaler.load_state_dict(previous["scaler"])
        step = previous["step"]
        last_val = previous["val_loss"]
        last_val_aux = previous["val_aux_loss"]
        last_val_metrics.update(previous.get("val_metrics", {}))
        restore_rng(previous["rng"], cuda)
        print(f"Resumed {checkpoint_path} at step {step}", flush=True)
    del previous
    stop = config.steps if stop_after is None else min(config.steps, stop_after)
    if stop <= step:
        print("Training already complete.", flush=True)
        return checkpoint_path
    # Dedicated loader generators preserve dropout/augmentation RNG on resume.
    loader_options = {"num_workers": config.num_workers, "pin_memory": cuda}
    train_loader = DataLoader(
        train_data,
        batch_sampler=StepBatches(sample_weights, config.batch_size, step, stop, config.seed),
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
    metric_names = tuple(ActionMetrics().means())
    columns = ["step", "train_loss", "val_loss", "train_aux_loss", "val_aux_loss", "lr", "it/s"]
    columns += [f"{split}_{key}" for split in ("train", "val") for key in metric_names]
    prepare_log(log_path, columns)
    print(
        f"{len(store.train_ids)} train / {len(store.val_ids)} val episodes; "
        f"{len(train_data)} train frames; cache={store.image_path}",
        flush=True,
    )
    started, logged_step = time.monotonic(), step
    train_total, train_count, aux_total, aux_count = 0.0, 0, 0.0, 0
    train_metrics = ActionMetrics()
    with log_path.open("a", newline="") as log:
        writer = csv.writer(log)
        if log.tell() == 0:
            writer.writerow(columns)
            log.flush()
        model.train()
        for batch in train_loader:
            factor = lr_factor(step, config)
            for group, base_lr in zip(
                optimizer.param_groups, (config.backbone_lr, config.lr), strict=True
            ):
                group["lr"] = base_lr * factor
            optimizer.zero_grad(set_to_none=True)
            l1_mask, gripper_mask = action_masks(
                batch["target"],
                batch["pad_mask"],
                batch["label_mask"],
                gripper_loss=config.gripper_loss,
            )
            l1_count = int(l1_mask.sum())
            gripper_count = int(gripper_mask.sum())
            labeled_count = batch["aux_mask"].sum().item() if config.aux_weight > 0 else 0
            batch_loss = 0.0
            batch_aux = 0.0
            # Only microbatch_size * 3 images occupy backbone activation RAM.
            # Each loss uses its own logical-batch denominator, never a shared
            # fraction that would dilute gripper-only samples.
            for start in range(0, len(batch["rgb"]), config.microbatch_size):
                micro = to_device(
                    {k: v[start : start + config.microbatch_size] for k, v in batch.items()}, device
                )
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=cuda):
                    prediction, aux = model(micro["rgb"], micro["proprio"], return_aux=True)
                    terms = loss_terms(prediction, micro, config)
                    action_loss = terms["l1_sum"] / max(
                        l1_count, 1
                    ) + config.gripper_weight * terms["gripper_bce_sum"] / max(gripper_count, 1)
                    aux_loss = action_loss.new_zeros(())
                    if aux is not None and labeled_count:
                        aux_loss = auxiliary_loss(
                            aux.float(), micro["aux_target"], micro["aux_mask"]
                        )
                        aux_loss = aux_loss * micro["aux_mask"].sum() / labeled_count
                    loss = action_loss + config.aux_weight * aux_loss
                scaler.scale(loss).backward()
                train_metrics.add(terms)
                batch_loss += float(action_loss.detach())
                batch_aux += float(aux_loss.detach())
            # A gripper-only batch can contain no oracle-defined states. Do not
            # apply weight decay or Adam momentum without any supervision.
            action_labeled = l1_count or (
                config.gripper_loss == "bce" and config.gripper_weight > 0 and gripper_count
            )
            if action_labeled or labeled_count:
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
            aux_total += batch_aux * labeled_count
            aux_count += labeled_count
            check = step % config.save_every == 0 or step == stop
            evaluate = step % config.val_every == 0 or check
            if evaluate:
                last_val, last_val_aux, last_val_metrics = validate(
                    ema, val_loader, device, cuda, config
                )
            if step % config.log_every == 0 or evaluate:
                if cuda:
                    torch.cuda.synchronize(device)
                elapsed = time.monotonic() - started
                throughput = (step - logged_step) / max(elapsed, 1e-9)
                train_loss = train_total / train_count
                train_aux = aux_total / aux_count if aux_count else 0.0
                means = train_metrics.means()
                writer.writerow(
                    [
                        step,
                        train_loss,
                        last_val,
                        train_aux,
                        last_val_aux,
                        optimizer.param_groups[1]["lr"],
                        throughput,
                        *(means[key] for key in metric_names),
                        *(last_val_metrics[key] for key in metric_names),
                    ]
                )
                log.flush()
                print(
                    f"step={step} train={train_loss:.5f} val={last_val:.5f} "
                    f"aux={train_aux:.5f} val_aux={last_val_aux:.5f} "
                    f"l1={means['l1']:.5f}/{last_val_metrics['l1']:.5f} "
                    f"gripper_bce={means['gripper_bce']:.5f}/{last_val_metrics['gripper_bce']:.5f} "
                    f"gripper_acc={means['gripper_accuracy']:.3f}/"
                    f"{last_val_metrics['gripper_accuracy']:.3f} "
                    f"open_acc={means['gripper_open_accuracy']:.3f}/"
                    f"{last_val_metrics['gripper_open_accuracy']:.3f} "
                    f"lr={optimizer.param_groups[1]['lr']:.3g} it/s={throughput:.2f}",
                    flush=True,
                )
                train_total, train_count, aux_total, aux_count = 0.0, 0, 0.0, 0
                train_metrics = ActionMetrics()
                started, logged_step = time.monotonic(), step
            if check:
                atomic_save(
                    {
                        "version": 2,
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
                        "val_aux_loss": last_val_aux,
                        "val_metrics": last_val_metrics,
                        "init_from": init_source,
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
    parser.add_argument(
        "--init-from", type=Path, help="EMA weights only; requires a fresh run directory"
    )
    for field in fields(Config):
        name = "--" + field.name.replace("_", "-")
        options = {"default": argparse.SUPPRESS}
        if field.name in ("data_dirs", "data_weights", "proprio_dims"):
            options["nargs"] = "+"
            if field.name != "data_dirs":
                options["type"] = int if field.name == "proprio_dims" else float
        elif field.type is bool:
            options["action"] = argparse.BooleanOptionalAction
        else:
            options["type"] = field.type
        parser.add_argument(name, **options)
    args = vars(parser.parse_args())
    config_path = args.pop("config")
    init_from = args.pop("init_from")
    values = read_config(config_path) if config_path else {}
    values.update(args)
    for key in ("data_dirs", "data_weights", "proprio_dims"):
        if key in values:
            if not isinstance(values[key], (list, tuple)):
                parser.error(f"{key} must be an inline list or CLI list")
            values[key] = tuple(values[key])
    train(Config(**values), init_from=init_from)


if __name__ == "__main__":
    main()
