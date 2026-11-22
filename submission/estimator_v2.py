"""Public-observation estimator using current RGB and an episode's first RGB/state."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .estimator import PlateEstimator
from .model import PROPRIO_DIMS

CONTACTS = ("source_rack", "target_rack", "table", "floor")


def rotate_z(points, angle):
    c, s = angle.cos(), angle.sin()
    x, y, z = points.unbind(-1)
    return torch.stack((c * x - s * y, s * x + c * y, z), dim=-1)


def from_initial_frame(points, start, current):
    world = start[:, 18:21] + rotate_z(points, start[:, 21])
    return rotate_z(world - current[:, 18:21], -current[:, 21])


def rotation_matrix(pair):
    """Orthonormalize the first and second columns, then recover the plate normal."""
    x = F.normalize(pair[..., :3], dim=-1, eps=1e-8)
    v = pair[..., 3:]
    y = F.normalize(v - (x * v).sum(-1, keepdim=True) * x, dim=-1, eps=1e-8)
    return torch.stack((x, y, torch.cross(x, y, dim=-1)), dim=-1)


class PlateEstimatorV2(PlateEstimator):
    def __init__(self):
        super().__init__(pretrained=False)
        self.correction = nn.Sequential(nn.Linear(368 * 2 + 6, 256), nn.GELU(), nn.Linear(256, 16))
        self.reset_geometry = nn.Sequential(nn.Linear(368, 96), nn.GELU(), nn.Linear(96, 6))
        for last in (self.correction[-1], self.reset_geometry[-1]):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        # Start with a vertical circular plate, with +x along the rack.
        self.register_buffer("upright", torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
        for key, n in (("proprio", len(PROPRIO_DIMS)), ("target", 7)):
            self.register_buffer(key + "_mean", torch.zeros(n))
            self.register_buffer(key + "_std", torch.ones(n))

    def encode(self, rgb, state):
        proprio = (state[:, list(PROPRIO_DIMS)] - self.proprio_mean) / self.proprio_std
        b = len(state)
        images = rgb.reshape(-1, 3, 84, 84).float().div(255)
        # Metric localization depends on pixel coordinates. Keep the camera
        # calibration fixed for both current and reset views.
        images = ((images - self.image_mean) / self.image_std).contiguous(
            memory_format=torch.channels_last
        )
        features = self.spatial(self.backbone(images)).reshape(b, 3, 64)
        cameras = self.camera_embedding.weight[None].expand(b, -1, -1)
        return torch.cat(
            (torch.cat((features, cameras), dim=-1).flatten(1), self.proprio(proprio)), dim=-1
        )

    def predict(self, features, initial, state, start):
        live = self.head(features).float()
        reset = self.head(initial).float()[:, :6] * self.target_std[:6] + self.target_mean[:6]
        reset = reset + 0.025 * self.reset_geometry(initial).float()
        source = from_initial_frame(reset[:, :3], start, state)
        site = from_initial_frame(reset[:, 3:6], start, state)
        physical = live[:, :7] * self.target_std + self.target_mean
        delta = self.correction(
            torch.cat(
                (
                    features,
                    initial,
                    (physical[:, :3] - source) * 10,
                    (physical[:, 3:6] - site) * 10,
                ),
                dim=-1,
            )
        ).float()
        # A moving plate requires live vision; stationary source measurements
        # remain informative while the gripper or arm occludes the plate.
        gate = (delta[:, 15:16] - 2).sigmoid()
        plate = torch.lerp(physical[:, :3], source, gate) + 0.025 * delta[:, :3]
        tilt = physical[:, 6:7] + 6 * delta[:, 3:4]
        held = live[:, 7:8] + delta[:, 4:5]
        rot = rotation_matrix(self.upright + delta[:, 5:11])
        return torch.cat(
            (plate, site, tilt, held, rot[:, :, 0], rot[:, :, 1], delta[:, 11:15]), dim=-1
        )


class EstimatorV2:
    def __init__(self, saved, device):
        self.device = torch.device(device)
        self.model = PlateEstimatorV2()
        self.model.load_state_dict(saved["model"])
        self.model.to(self.device).eval()
        self.reset()

    def reset(self):
        self.initial = None
        self.start = None

    @torch.inference_mode()
    def __call__(self, obs):
        rgb = np.asarray(obs["rgb"])
        state = np.asarray(obs["state"], np.float32)
        if (
            rgb.shape != (3, 3, 84, 84)
            or rgb.dtype != np.uint8
            or state.shape != (50,)
            or not np.isfinite(state).all()
        ):
            raise ValueError("Expected three 84-pixel RGB views and finite 50-vector state")
        s = torch.as_tensor(state.copy(), device=self.device)[None]
        z = self.model.encode(torch.as_tensor(rgb.copy(), device=self.device)[None], s)
        if self.initial is None:
            self.initial, self.start = z, s
        y = self.model.predict(z, self.initial, s, self.start)[0]
        rot = rotation_matrix(y[8:14]).cpu().numpy()
        prob = y[14:18].sigmoid().cpu().numpy()
        if not torch.isfinite(y).all():
            raise ValueError("Nonfinite v2 prediction")
        return dict(
            plate_rel=y[:3].cpu().numpy(),
            target_site_rel=y[3:6].cpu().numpy(),
            tilt_deg=float(y[6].clamp(0, 180)),
            held=float(y[7].sigmoid()),
            plate_rot_rel=rot,
            contacts={k: bool(v >= 0.5) for k, v in zip(CONTACTS, prob, strict=True)},
        )
