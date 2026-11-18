"""Observation-only BC policy. Set ADV_CKPT to an EMA training checkpoint."""

import os
from pathlib import Path

import numpy as np
import torch

from .model import ACTIVE_DIMS, PROPRIO_DIMS, BCModel, load_bc_weights, validate_proprio_dims


class BCPolicy:
    def __init__(self, checkpoint):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Our own checkpoints contain optimizer/RNG objects as well as tensors.
        checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        self.horizon = config["horizon"]
        self.exec_steps = config["exec_steps"]
        if not 1 <= self.exec_steps <= self.horizon <= 50:
            raise ValueError("Require 1 <= exec_steps <= horizon <= 50")
        self.proprio_dims = validate_proprio_dims(config.get("proprio_dims", PROPRIO_DIMS))
        # The auxiliary head never participates in inference, even when saved.
        self.model = BCModel(self.horizon, pretrained=False, proprio_dims=self.proprio_dims)
        load_bc_weights(self.model, checkpoint["ema"])
        self.model.to(self.device).eval()
        stats = checkpoint["stats"]
        self.mean = np.asarray(stats["proprio_mean"], dtype=np.float32)
        self.std = np.asarray(stats["proprio_std"], dtype=np.float32)
        self.action_mean = np.asarray(stats["action_mean"], dtype=np.float32)
        shape = (len(self.proprio_dims),)
        if self.mean.shape != shape or self.std.shape != shape or self.action_mean.shape != (20,):
            raise ValueError("Checkpoint normalization shapes do not match this policy")
        self.temporal_ensemble = config.get("temporal_ensemble", False)
        self.ensemble_decay = config.get("ensemble_decay", 0.01)
        self.reset()

    def reset(self):
        self.time = 0
        self.predictions = []

    @torch.inference_mode()
    def __call__(self, obs):
        rgb = np.asarray(obs["rgb"], dtype=np.uint8)
        state = np.asarray(obs["state"], dtype=np.float32)
        if rgb.shape != (3, 3, 84, 84) or state.shape != (50,):
            raise ValueError("Expected RGB (3,3,84,84) and state (50,)")
        proprio = (state[list(self.proprio_dims)] - self.mean) / self.std
        chunk = (
            self.model(
                torch.from_numpy(rgb.copy())[None].to(self.device),
                torch.from_numpy(proprio)[None].to(self.device),
            )[0]
            .float()
            .cpu()
            .numpy()
        )
        active = chunk[: self.exec_steps].copy()
        if self.temporal_ensemble:
            # Observations still arrive only every exec_steps control steps.
            self.predictions = [(t, p) for t, p in self.predictions if t + self.horizon > self.time]
            self.predictions.append((self.time, chunk))
            for offset in range(self.exec_steps):
                candidates = [
                    p[self.time + offset - t]
                    for t, p in self.predictions
                    if 0 <= self.time + offset - t < self.horizon
                ]
                # ACT convention: older overlapping predictions get more weight.
                weights = np.exp(-self.ensemble_decay * np.arange(len(candidates)))
                active[offset] = np.average(candidates, axis=0, weights=weights)
        self.time += self.exec_steps
        actions = np.broadcast_to(self.action_mean, (self.exec_steps, 20)).copy()
        actions[:, ACTIVE_DIMS] = active
        actions[:, 18] = np.where(actions[:, 18] >= 0, 1.0, -1.0)
        if not np.isfinite(actions).all():
            raise ValueError("BC prediction contains nonfinite actions")
        return np.clip(actions, -1, 1).astype(np.float32)


def make_policy():
    path = os.environ.get("ADV_CKPT") or Path(__file__).parent / "weights" / "policy.pt"
    return BCPolicy(path)
