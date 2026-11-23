"""Flow-matching student. ADV_CKPT selects a trusted microVLA checkpoint."""

import os
from pathlib import Path

import numpy as np
import torch

from method.vla.model import (
    ACTIVE_DIMS,
    STATE_SIGMAS,
    TOKENIZER,
    Config,
    MicroVLA,
    instruction_tokens,
    prepare_images,
)
from submission.model import validate_proprio_dims


class VLAPolicy:
    def __init__(self, checkpoint, *, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if checkpoint.get("version") != "microvla-1" or checkpoint.get("phase") != "train":
            raise ValueError("Expected an action-training microVLA checkpoint")
        config = checkpoint["config"]
        if self.device.type == "cpu":
            torch.set_num_threads(config.get("cpu_threads", 2))
        if checkpoint.get("tokenizer") != TOKENIZER:
            raise ValueError("Checkpoint instruction tokenizer does not match this policy")
        # Dev ablations only: final submission value belongs in the checkpoint config.
        self.exec_steps = int(os.environ.get("ADV_EXEC_STEPS", config["exec_steps"]))
        self.proprio_dims = validate_proprio_dims(config["proprio_dims"])
        self.model = MicroVLA(Config(**checkpoint["model_config"]))
        if not 1 <= self.exec_steps <= self.model.config.horizon <= 50:
            raise ValueError("Require 1 <= exec_steps <= horizon <= 50")
        if self.model.config.state_dim != len(self.proprio_dims):
            raise ValueError("State dimensions do not match checkpoint")
        if self.model.config.action_dim != len(ACTIVE_DIMS):
            raise ValueError("Action dimensions do not match checkpoint")
        self.model.load_state_dict(checkpoint["ema"])
        self.model.to(self.device).eval()
        stats = checkpoint["stats"]
        self.mean = np.asarray(stats["proprio_mean"], np.float32)
        self.std = np.asarray(stats["proprio_std"], np.float32)
        self.action_mean = np.asarray(stats["action_mean"], np.float32)
        self.state_sigmas = float(checkpoint.get("state_sigmas", STATE_SIGMAS))
        if (
            self.mean.shape != (len(self.proprio_dims),)
            or self.std.shape != self.mean.shape
            or self.action_mean.shape != (20,)
            or not np.all(self.std > 0)
            or not np.isfinite(self.state_sigmas)
            or self.state_sigmas <= 0
        ):
            raise ValueError("Invalid normalization stats")
        if not all(np.isfinite(x).all() for x in (self.mean, self.std, self.action_mean)):
            raise ValueError("Nonfinite normalization stats")
        self.flow_steps = int(os.environ.get("ADV_FLOW_STEPS", config["flow_steps"]))
        if self.flow_steps < 1:
            raise ValueError("ADV_FLOW_STEPS must be positive")
        self.seed = config["seed"]
        self.reset()

    def reset(self):
        # Same initial RNG state each episode; fresh reproducible noise each chunk.
        self.generator = torch.Generator(device="cpu").manual_seed(self.seed)

    @torch.inference_mode()
    def __call__(self, obs):
        rgb = np.asarray(obs["rgb"], dtype=np.uint8)
        state = np.asarray(obs["state"], dtype=np.float32)
        if rgb.shape != (3, 3, 84, 84) or state.shape != (50,):
            raise ValueError("Expected RGB (3,3,84,84) and state (50,)")
        if not np.isfinite(state).all():
            raise ValueError("Nonfinite observation state")
        proprio = np.clip(
            (state[list(self.proprio_dims)] - self.mean) / (self.state_sigmas * self.std), -1, 1
        )
        images = prepare_images(
            torch.from_numpy(rgb.copy())[None].to(self.device), self.model.config.image_size
        )
        noise = torch.randn(
            1, self.model.config.horizon, len(ACTIVE_DIMS), generator=self.generator
        ).to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            chunk = (
                self.model.sample(
                    images,
                    torch.from_numpy(proprio)[None].to(self.device),
                    instruction_tokens(1, self.device),
                    num_steps=self.flow_steps,
                    noise=noise,
                    cached=True,
                )[0]
                .float()
                .cpu()
                .numpy()
            )
        actions = np.broadcast_to(self.action_mean, (self.exec_steps, 20)).copy()
        actions[:, ACTIVE_DIMS] = chunk[: self.exec_steps]
        if not np.isfinite(actions).all():
            raise ValueError("Nonfinite microVLA prediction")
        actions[:, 18] = np.where(actions[:, 18] < 0, -1.0, 1.0)
        return np.clip(actions, -1, 1).astype(np.float32)


def make_policy():
    path = os.environ.get("ADV_CKPT") or Path(__file__).parent / "weights" / "microvla.pt"
    return VLAPolicy(path)
