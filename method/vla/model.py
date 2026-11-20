"""Task adapter for KhushiJ2911/microVLA (MIT); vendor/train.py stays unmodified.

Import the upstream architecture, tokenizer and augmentations directly. Only the
masked objective and this task's observation preprocessing are defined here.
"""

import torch
import torch.nn.functional as F

from method.vla.vendor.train import (
    Config,
    MicroVLA,
    augment,
    color_jitter,
    parameter_counts,
    tokenize,
    tokenizer,
)
from submission.model import ACTIVE_DIMS, GRIPPER_INDEX, PROPRIO_DIMS

INSTRUCTION = "move the plate from the left rack to the right rack"
TOKENIZER = tokenizer([INSTRUCTION])
STATE_SIGMAS = 3.0


def model_config(horizon=20, **overrides):
    return Config(
        **{
            **dict(
                image_size=96,
                num_cameras=3,
                vocab_size=len(TOKENIZER["vocab"]),
                horizon=horizon,
                action_dim=len(ACTIVE_DIMS),
                state_dim=len(PROPRIO_DIMS),
            ),
            **overrides,
        }
    )


def prepare_images(rgb, image_size, training=False, jitter=True):
    if training:
        rgb = augment(rgb)
        if jitter:
            rgb = color_jitter(rgb)
    # Identical interpolation for training/inference; no preprocessed image cache.
    return F.interpolate(
        rgb.flatten(0, 1).float(),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).unflatten(0, rgb.shape[:2])


def instruction_tokens(batch_size, device):
    return torch.as_tensor(tokenize([INSTRUCTION], TOKENIZER), device=device).expand(batch_size, -1)


def valid_mask(batch):
    return batch["label_mask"].bool() & ~batch["pad_mask"].bool()[..., None]


def flow_loss_terms(model, images, state, instr, actions, mask, *, noise=None, t=None):
    """Return a sum/count for exact logical-batch and validation normalization.

    Untrusted targets never enter the model, including as noisy input context.
    Empty supervision yields a differentiable zero and no optimizer update.
    """
    noise = torch.randn_like(actions) if noise is None else noise
    if t is None:
        t = torch.rand(len(actions), device=actions.device).pow(2 / 3) * 0.999 + 0.001
    clean = torch.where(mask, actions, 0.0)
    interpolated = t[:, None, None] * noise + (1 - t[:, None, None]) * clean
    noisy = torch.where(mask, interpolated, noise)
    velocity = model(images, state, instr, noisy, t).float()
    residual = torch.where(mask, velocity - (noise - clean), 0.0)
    return residual.square().sum(), mask.sum()


def token_counts(config):
    visual = config.num_cameras * (config.image_size // 8) ** 2
    prefix = visual + TOKENIZER["length"] + config.state_dim
    return {
        "image": visual,
        "language": TOKENIZER["length"],
        "state": config.state_dim,
        "prefix": prefix,
        "action": config.horizon,
        "total": prefix + config.horizon,
    }


__all__ = [
    "ACTIVE_DIMS",
    "Config",
    "GRIPPER_INDEX",
    "INSTRUCTION",
    "MicroVLA",
    "PROPRIO_DIMS",
    "STATE_SIGMAS",
    "TOKENIZER",
    "flow_loss_terms",
    "instruction_tokens",
    "model_config",
    "parameter_counts",
    "prepare_images",
    "token_counts",
    "valid_mask",
]
