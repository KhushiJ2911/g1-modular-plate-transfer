"""Checkpoint ensemble of BC policies + ACT-style temporal ensembling (observation-only).

ADV_CKPTS = comma-separated BC checkpoints. Each model predicts a raw chunk from the same observation
(proprio normalised with its own training stats); raw chunks (incl. gripper logits) are averaged across
models, then temporally ensembled exactly as submission/policy.py does, and the gripper is classified once.
Env: ADV_EXEC_STEPS (default 2), ADV_ENSEMBLE_DECAY (default 0.0 = uniform).
"""
import os

import numpy as np
import torch

from .model import ACTIVE_DIMS
from .policy import BCPolicy


class MultiBCPolicy:
    def __init__(self, paths, exec_steps=2, decay=0.0):
        self.members = [BCPolicy(p) for p in paths]
        first = self.members[0]
        for m in self.members[1:]:
            if m.horizon != first.horizon or m.proprio_dims != first.proprio_dims:
                raise ValueError("Ensemble members must share horizon and proprio dims")
            if m.gripper_loss != first.gripper_loss:
                raise ValueError("Ensemble members must share the gripper loss")
        self.horizon, self.exec_steps, self.decay = first.horizon, exec_steps, decay
        self.gripper_loss, self.gripper_threshold = first.gripper_loss, first.gripper_threshold
        self.action_mean = np.mean([m.action_mean for m in self.members], axis=0)
        self.reset()

    def reset(self):
        for m in self.members:
            m.reset()
        self.predictions, self.time = [], 0

    def _chunk(self, m, rgb, state):
        proprio = (state[list(m.proprio_dims)] - m.mean) / m.std
        with torch.inference_mode():
            out = m.model(torch.from_numpy(rgb.copy())[None].to(m.device),
                          torch.from_numpy(proprio)[None].to(m.device))[0]
        return out.float().cpu().numpy()

    def __call__(self, obs):
        rgb = np.asarray(obs["rgb"], dtype=np.uint8)
        state = np.asarray(obs["state"], dtype=np.float32)
        if rgb.shape != (3, 3, 84, 84) or state.shape != (50,):
            raise ValueError("Expected RGB (3,3,84,84) and state (50,)")
        chunk = np.mean([self._chunk(m, rgb, state) for m in self.members], axis=0)
        active = chunk[: self.exec_steps].copy()
        self.predictions = [(t, p) for t, p in self.predictions if t + self.horizon > self.time]
        self.predictions.append((self.time, chunk))
        for offset in range(self.exec_steps):
            cands = [p[self.time + offset - t] for t, p in self.predictions
                     if 0 <= self.time + offset - t < self.horizon]
            weights = np.exp(-self.decay * np.arange(len(cands)))
            active[offset] = np.average(cands, axis=0, weights=weights)
        self.time += self.exec_steps
        actions = np.broadcast_to(self.action_mean, (self.exec_steps, 20)).copy()
        actions[:, ACTIVE_DIMS] = active
        if self.gripper_loss == "bce":
            p_open = torch.from_numpy(actions[:, 18].copy()).sigmoid().numpy()
            actions[:, 18] = np.where(p_open > self.gripper_threshold, -1.0, 1.0)
        else:
            actions[:, 18] = np.where(actions[:, 18] >= 0, 1.0, -1.0)
        if not np.isfinite(actions).all():
            raise ValueError("Ensemble prediction contains nonfinite actions")
        return np.clip(actions, -1, 1).astype(np.float32)


def make_policy():
    paths = [p for p in os.environ["ADV_CKPTS"].split(",") if p]
    return MultiBCPolicy(paths, int(os.environ.get("ADV_EXEC_STEPS", 2)),
                         float(os.environ.get("ADV_ENSEMBLE_DECAY", 0.0)))
