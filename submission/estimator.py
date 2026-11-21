"""Observation-only plate estimator; privileged labels are training targets only."""

import numpy as np
import torch
from torch import nn

from .model import PROPRIO_DIMS, BCModel, random_shift


class PlateEstimator(nn.Module):
    """BC image encoder plus a small head: plate xyz, site xyz, tilt, held logit."""

    def __init__(self, pretrained=True):
        super().__init__()
        encoder = BCModel(horizon=1, pretrained=pretrained)
        self.backbone = encoder.backbone
        self.spatial = encoder.spatial
        self.camera_embedding = encoder.camera_embedding
        self.proprio = encoder.proprio
        self.register_buffer("image_mean", encoder.image_mean)
        self.register_buffer("image_std", encoder.image_std)
        self.head = nn.Sequential(nn.Linear(368, 256), nn.GELU(), nn.Linear(256, 8))
        self.train()

    def train(self, mode=True):
        super().train(mode)
        for layer in self.backbone.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
        return self

    def forward(self, rgb, proprio):
        batch = rgb.shape[0]
        images = rgb.reshape(-1, 3, 84, 84).float().div(255)
        if self.training:
            images = random_shift(images)
        images = ((images - self.image_mean) / self.image_std).contiguous(
            memory_format=torch.channels_last
        )
        features = self.spatial(self.backbone(images)).reshape(batch, 3, 64)
        cameras = self.camera_embedding.weight[None].expand(batch, -1, -1)
        vision = torch.cat((features, cameras), dim=-1).flatten(1)
        return self.head(torch.cat((vision, self.proprio(proprio)), dim=-1))


class Estimator:
    def __init__(self, checkpoint, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["version"] != 1 or tuple(saved["config"]["proprio_dims"]) != PROPRIO_DIMS:
            raise ValueError("Unsupported estimator checkpoint")
        self.model = PlateEstimator(pretrained=False)
        self.model.load_state_dict(saved["model"])
        self.model.to(self.device).eval()
        self.stats = {key: np.asarray(value, np.float32) for key, value in saved["stats"].items()}
        for name, size in (("proprio", 19), ("target", 7)):
            for suffix in ("mean", "std"):
                value = self.stats[f"{name}_{suffix}"]
                if value.shape != (size,) or not np.isfinite(value).all():
                    raise ValueError(f"Invalid estimator {name}_{suffix}")
            if (self.stats[f"{name}_std"] <= 0).any():
                raise ValueError("Estimator standard deviations must be positive")

    @torch.inference_mode()
    def __call__(self, obs):
        rgb = np.asarray(obs["rgb"])
        state = np.asarray(obs["state"], np.float32)
        if rgb.shape != (3, 3, 84, 84) or rgb.dtype != np.uint8 or state.shape != (50,):
            raise ValueError("Expected RGB uint8 (3,3,84,84), state (50,)")
        if not np.isfinite(state).all():
            raise ValueError("Nonfinite proprioception")
        proprio = (state[list(PROPRIO_DIMS)] - self.stats["proprio_mean"]) / self.stats[
            "proprio_std"
        ]
        output = (
            self.model(
                torch.from_numpy(rgb.copy())[None].to(self.device),
                torch.from_numpy(proprio)[None].to(self.device),
            )[0]
            .float()
            .cpu()
            .numpy()
        )
        physical = output[:7] * self.stats["target_std"] + self.stats["target_mean"]
        if not np.isfinite(output).all() or not np.isfinite(physical).all():
            raise ValueError("Nonfinite estimator prediction")
        return {
            "plate_rel": physical[:3],
            "target_site_rel": physical[3:6],
            "tilt_deg": float(physical[6]),
            "held": float(torch.tensor(output[7]).sigmoid()),
        }
