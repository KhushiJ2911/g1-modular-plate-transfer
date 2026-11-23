"""CPU/CUDA microVLA training using the existing BC dataset and read-only cache."""

import argparse
import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from method.bc.dataset import EpisodeStore
from method.bc.train import (
    StepBatches,
    atomic_save,
    lr_factor,
    read_config,
    restore_rng,
    rng_state,
    to_device,
    update_ema,
)
from method.vla.model import (
    ACTIVE_DIMS,
    GRIPPER_INDEX,
    PROPRIO_DIMS,
    STATE_SIGMAS,
    TOKENIZER,
    MicroVLA,
    flow_loss_terms,
    instruction_tokens,
    model_config,
    parameter_counts,
    prepare_images,
    token_counts,
    valid_mask,
)
from method.vla.muon import build_optimizer, print_optimizer_table


@dataclass
class Config:
    data_dirs: tuple = ()
    data_weights: tuple = ()
    cache_dir: str = "/home/keyu/advaitic/data/cache"
    run_dir: str = "method/vla/checkpoints/microvla_v1"
    proprio_dims: tuple = PROPRIO_DIMS
    horizon: int = 20
    exec_steps: int = 10
    steps: int = 10000
    batch_size: int = 64
    microbatch_size: int = 2
    val_episodes: int = 2
    val_every: int = 250
    val_batches: int = 0  # 0 evaluates every held-out frame.
    sample_every: int = 500
    sample_batches: int = 2
    save_every: int = 250
    log_every: int = 25
    flow_steps: int = 10
    lr: float = 3e-4
    optimizer: str = "adamw"
    muon_lr: float = None
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    ema_decay: float = 0.999
    num_workers: int = 0
    cpu_threads: int = 2
    seed: int = 42
    device: str = "cuda"
    color_jitter: bool = True
    distill: bool = False
    distill_steps: int = 1000
    siglip_path: str = "google/siglip-base-patch16-224"
    model_overrides: dict = field(default_factory=dict)


def validate_config(config):
    if not 1 <= config.exec_steps <= config.horizon <= 50:
        raise ValueError("Require 1 <= exec_steps <= horizon <= 50")
    for key in (
        "steps",
        "batch_size",
        "microbatch_size",
        "val_episodes",
        "val_every",
        "sample_every",
        "sample_batches",
        "save_every",
        "log_every",
        "flow_steps",
        "distill_steps",
        "cpu_threads",
    ):
        if getattr(config, key) < 1:
            raise ValueError(f"{key} must be positive")
    if (
        config.warmup_steps < 0
        or config.val_batches < 0
        or not 0 <= config.ema_decay < 1
        or config.num_workers < 0
        or not math.isfinite(config.lr)
        or config.lr <= 0
        or not math.isfinite(config.weight_decay)
        or config.weight_decay < 0
    ):
        raise ValueError("Invalid optimizer, EMA, warmup or worker settings")
    if (
        config.optimizer not in {"adamw", "muon"}
        or not 0 <= config.muon_momentum < 1
        or type(config.muon_ns_steps) is not int
        or config.muon_ns_steps < 1
        or (
            config.muon_lr is not None
            and (not math.isfinite(config.muon_lr) or config.muon_lr <= 0)
        )
        or (config.optimizer == "adamw" and config.muon_lr is not None)
    ):
        raise ValueError("Invalid optimizer or Muon settings")
    if tuple(config.proprio_dims) != PROPRIO_DIMS:
        raise ValueError("microvla v1 uses the BC student's exact 19 proprio dims")
    allowed = {
        "image_size",
        "width",
        "depth",
        "heads",
        "head_dim",
        "mlp",
        "expert_width",
        "expert_mlp",
        "vit_width",
        "vit_depth",
        "vit_heads",
    }
    if set(config.model_overrides) - allowed:
        raise ValueError("model_overrides may change architecture sizes only")
    c = model_config(config.horizon, **config.model_overrides)
    if (
        any(not isinstance(v, int) or v < 1 for v in asdict(c).values())
        or c.image_size % 8
        or c.vit_width % c.vit_heads
        or c.head_dim % 2
        or c.expert_width % 2
    ):
        raise ValueError("Invalid patch, head, or model dimensions")
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu")


def prepare(batch, config, device, training=False):
    batch = to_device(batch, device)
    images = prepare_images(
        batch["rgb"],
        model_config(config.horizon, **config.model_overrides).image_size,
        training=training,
        jitter=config.color_jitter,
    )
    return (
        images,
        (batch["proprio"] / STATE_SIGMAS).clamp(-1, 1),
        instruction_tokens(len(images), device),
        batch,
    )


def autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")


@torch.inference_mode()
def validate(model, loader, config, device, *, sample=False):
    model.eval()
    loss_sum, count = 0.0, 0
    l1, counts = torch.zeros(len(ACTIVE_DIMS)), torch.zeros(len(ACTIVE_DIMS))
    correct = gripper_count = open_correct = open_count = 0
    flow_batches = config.val_batches or len(loader)
    sample_ids = (
        set(np.linspace(0, len(loader) - 1, min(config.sample_batches, len(loader)), dtype=int))
        if sample
        else set()
    )
    # Validation consumes no training RNG, including at an interruption boundary.
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config.seed + 100000)
        for i, batch in enumerate(loader):
            if i >= max(flow_batches, max(sample_ids, default=-1) + 1):
                break
            if i >= flow_batches and i not in sample_ids:
                continue
            images, state, instr, batch = prepare(batch, config, device)
            mask = valid_mask(batch)
            with autocast(device):
                if i < flow_batches:
                    total, n = flow_loss_terms(model, images, state, instr, batch["target"], mask)
                    loss_sum += float(total)
                    count += int(n)
                if i in sample_ids:
                    generator = torch.Generator(device=device).manual_seed(config.seed + 200000 + i)
                    noise = torch.randn(
                        len(images),
                        config.horizon,
                        len(ACTIVE_DIMS),
                        device=device,
                        generator=generator,
                    )
                    prediction = model.sample(
                        images, state, instr, num_steps=config.flow_steps, noise=noise
                    )
                    error = torch.where(mask, (prediction.float() - batch["target"]).abs(), 0)
                    l1 += error.sum((0, 1)).cpu()
                    counts += mask.sum((0, 1)).cpu()
                    gm = mask[..., GRIPPER_INDEX]
                    opened = batch["target"][..., GRIPPER_INDEX] < 0
                    same = (prediction[..., GRIPPER_INDEX] < 0) == opened
                    correct += int((same & gm).sum())
                    gripper_count += int(gm.sum())
                    open_correct += int((same & gm & opened).sum())
                    open_count += int((gm & opened).sum())
    metrics = {"val_flow_loss": loss_sum / count if count else None, "val_elements": count}
    if sample:
        metrics.update(
            sampled_l1_per_dim={
                str(d): float(l1[j] / counts[j]) if counts[j] else None
                for j, d in enumerate(ACTIVE_DIMS)
            },
            sampled_elements_per_dim=counts.tolist(),
            gripper_sign_accuracy=correct / gripper_count if gripper_count else None,
            gripper_open_accuracy=open_correct / open_count if open_count else None,
            gripper_count=gripper_count,
            gripper_open_count=open_count,
        )
    return metrics


def load_teacher(config, device):
    # Never download implicitly or create a second cache. User provisions weights.
    from transformers import SiglipVisionModel

    teacher = (
        SiglipVisionModel.from_pretrained(
            config.siglip_path,
            local_files_only=True,
            torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    if teacher.config.patch_size != 16:
        raise ValueError("SigLIP distillation expects patch size 16")
    return teacher


def distill_loss(model, head, teacher, images, device):
    # Match vendor recipe: one random camera, twice student resolution so teacher
    # patch16 and student patch8 produce the same patch grid. No CLS token in SigLIP.
    images = images[
        torch.arange(len(images), device=device),
        torch.randint(images.shape[1], (len(images),), device=device),
    ]
    with torch.no_grad(), autocast(device):
        pixels = F.interpolate(
            images.float().div(127.5).sub(1), scale_factor=2, mode="bilinear", align_corners=False
        )
        target = teacher(
            pixel_values=pixels.to(next(teacher.parameters()).dtype), interpolate_pos_encoding=True
        ).last_hidden_state.float()
        target = F.layer_norm(target, target.shape[-1:])
    with autocast(device):
        prediction = head(model.vision(images)).float()
        return (1 - F.cosine_similarity(prediction, target, dim=-1)).mean() + F.mse_loss(
            prediction, target
        )


def validate_resume_config(config, previous):
    if previous.get("version") != "microvla-1":
        raise ValueError("Incompatible checkpoint")
    saved = dict(previous["config"])
    # Legacy checkpoints predate these fields and always used ordinary AdamW.
    defaults = Config()
    for key in ("optimizer", "muon_lr", "muon_momentum", "muon_ns_steps"):
        saved.setdefault(key, getattr(defaults, key))
    for key, value in asdict(config).items():
        if key not in {"device", "run_dir", "cache_dir", "num_workers"}:
            if value != saved[key]:
                raise ValueError(f"Resume config changed: {key}")


def schedule_optimizer(optimizer, config, phase, step):
    if phase == "distill":
        # Preserve flat-LR legacy AdamW distillation. Muon uses the same warmup
        # and cosine rule in both phases, with the relevant phase's length.
        factor = (
            lr_factor(step, replace(config, steps=config.distill_steps))
            if config.optimizer == "muon"
            else 1.0
        )
    else:
        factor = lr_factor(step, config)
    for group in optimizer.param_groups:
        group["lr"] = group.get("base_lr", config.lr) * factor


def train(config, *, resume=False, stop_after=None):
    """stop_after limits total updates for interruption tests, keeping LR unchanged."""
    validate_config(config)
    device = torch.device(config.device)
    cuda = device.type == "cuda"
    if not cuda:
        torch.set_num_threads(config.cpu_threads)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    run_dir = Path(config.run_dir)
    path = run_dir / "last.pt"
    if not resume and path.exists():
        raise FileExistsError(f"{path} exists; use --resume or a new run_dir")
    previous = torch.load(path, map_location="cpu", weights_only=False) if resume else None
    if previous:
        validate_resume_config(config, previous)
    store = EpisodeStore(
        config.data_dirs, config.cache_dir, config.val_episodes, config.proprio_dims, read_only=True
    )
    mc = model_config(config.horizon, **config.model_overrides)
    if previous:
        if previous["data_digest"] != store.digest:
            raise ValueError("Resume dataset changed")
    run_dir.mkdir(parents=True, exist_ok=True)
    model = MicroVLA(mc).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    step = distill_step = 0
    if previous:
        model.load_state_dict(previous["model"])
        ema.load_state_dict(previous["ema"])
        step, distill_step = previous["step"], previous["distill_step"]
    train_data, val_data = (
        store.dataset("train", config.horizon),
        store.dataset("val", config.horizon),
    )
    weights = train_data.sample_weights(config.data_weights)
    val_loader = DataLoader(
        val_data,
        batch_size=config.microbatch_size,
        num_workers=config.num_workers,
        generator=torch.Generator().manual_seed(config.seed),
    )
    print(
        json.dumps(
            {
                "params": parameter_counts(model),
                "tokens": token_counts(mc),
                "cache": str(store.image_path),
                "device": str(device),
            }
        ),
        flush=True,
    )
    phase = "distill" if config.distill and distill_step < config.distill_steps else "train"
    started = time.monotonic()
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    while True:
        teacher = head = None
        if phase == "distill":
            teacher = load_teacher(config, device)
            head = torch.nn.Linear(mc.width, teacher.config.hidden_size).to(device)
            parameters = [*model.vision.parameters(), *head.parameters()]
            first, total = distill_step, config.distill_steps
        else:
            parameters = list(model.parameters())
            first, total = step, config.steps
            if previous and previous["phase"] == "distill":
                ema.load_state_dict(model.state_dict())
        optimizer = build_optimizer(
            model,
            optimizer=config.optimizer,
            lr=config.lr,
            weight_decay=config.weight_decay,
            muon_lr=config.muon_lr,
            momentum=config.muon_momentum,
            ns_steps=config.muon_ns_steps,
            distill_head=head,
        )
        print_optimizer_table(optimizer, phase)
        if previous:
            if previous["phase"] == phase:
                if head is not None:
                    head.load_state_dict(previous["distill_head"])
                optimizer.load_state_dict(previous["optimizer"])
            restore_rng(previous["rng"], cuda)
            previous = None
        loader = DataLoader(
            train_data,
            batch_sampler=StepBatches(
                weights,
                config.batch_size,
                first,
                total,
                config.seed + (1000000 if phase == "distill" else 0),
            ),
            num_workers=config.num_workers,
            pin_memory=cuda,
            generator=torch.Generator().manual_seed(config.seed),
        )
        train_sum = train_count = 0
        last_metrics = {}
        for phase_index, batch in enumerate(loader, first + 1):
            model.train()
            denominator = int(valid_mask(batch).sum()) if phase == "train" else len(batch["rgb"])
            optimizer.zero_grad(set_to_none=True)
            schedule_optimizer(optimizer, config, phase, phase_index - 1)
            batch_sum = 0.0
            for start in range(0, len(batch["rgb"]), config.microbatch_size):
                micro = {k: v[start : start + config.microbatch_size] for k, v in batch.items()}
                images, state, instr, micro = prepare(micro, config, device, training=True)
                with autocast(device):
                    if phase == "train":
                        numerator, _ = flow_loss_terms(
                            model, images, state, instr, micro["target"], valid_mask(micro)
                        )
                    else:
                        numerator = distill_loss(model, head, teacher, images, device) * len(images)
                    loss = numerator / max(denominator, 1)
                loss.backward()
                batch_sum += float(numerator.detach())
            if denominator:
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                if not torch.isfinite(norm):
                    raise FloatingPointError("Nonfinite gradient; checkpoint not overwritten")
                optimizer.step()
                update_ema(ema, model, config.ema_decay)
            if phase == "train":
                step = phase_index
            else:
                distill_step = phase_index
            train_sum += batch_sum
            train_count += denominator
            updates = step + distill_step
            stopping = stop_after is not None and updates >= stop_after
            saving = phase_index % config.save_every == 0 or phase_index == total or stopping
            sampling = phase == "train" and (step % config.sample_every == 0 or step == total)
            checking = phase == "train" and (step % config.val_every == 0 or saving or sampling)
            if checking:
                last_metrics = validate(ema, val_loader, config, device, sample=sampling)
            if phase_index % config.log_every == 0 or saving or checking:
                record = dict(
                    phase=phase,
                    step=step,
                    distill_step=distill_step,
                    **{
                        "train_flow_loss" if phase == "train" else "distill_loss": train_sum
                        / train_count
                        if train_count
                        else None
                    },
                    lr=optimizer.param_groups[0]["lr"],
                    elapsed_seconds=time.monotonic() - started,
                    **last_metrics,
                )
                with (run_dir / "log.jsonl").open("a") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                print(json.dumps(record, allow_nan=False), flush=True)
                train_sum = train_count = 0
            if saving:
                atomic_save(
                    dict(
                        version="microvla-1",
                        phase=phase,
                        config=asdict(config),
                        model_config=asdict(mc),
                        model=model.state_dict(),
                        ema=ema.state_dict(),
                        optimizer=optimizer.state_dict(),
                        distill_head=head.state_dict() if head is not None else None,
                        step=step,
                        distill_step=distill_step,
                        rng=rng_state(cuda),
                        stats=store.stats,
                        state_sigmas=STATE_SIGMAS,
                        tokenizer=TOKENIZER,
                        data_digest=store.digest,
                        train_episodes=[store.files[i] for i in store.train_ids],
                        val_episodes=[store.files[i] for i in store.val_ids],
                        metrics=last_metrics,
                    ),
                    path,
                )
            if stopping:
                return path
        if phase == "train":
            break
        del teacher, head, optimizer, parameters
        phase = "train"
        ema.load_state_dict(model.state_dict())
    if cuda:
        print(
            json.dumps(
                {
                    "peak_cuda_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
                    "peak_cuda_reserved_gb": torch.cuda.max_memory_reserved(device) / 1e9,
                }
            )
        )
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("method/vla/microvla_v1.yaml"))
    parser.add_argument("--resume", action="store_true")
    for f in fields(Config):
        options = {"default": argparse.SUPPRESS}
        if f.type is bool:
            options["action"] = argparse.BooleanOptionalAction
        elif f.type is tuple:
            options.update(
                nargs="+",
                type=(
                    int if f.name == "proprio_dims" else float if f.name == "data_weights" else str
                ),
            )
        elif f.type is dict:
            options["type"] = json.loads
        else:
            options["type"] = f.type
        parser.add_argument("--" + f.name.replace("_", "-"), **options)
    args = vars(parser.parse_args())
    path, resume = args.pop("config"), args.pop("resume")
    values = read_config(path)
    values.update(args)
    for key in ("data_dirs", "data_weights", "proprio_dims"):
        if key in values:
            values[key] = tuple(values[key])
    train(Config(**values), resume=resume)


if __name__ == "__main__":
    main()
