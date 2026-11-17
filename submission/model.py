"""Shared BC network; kept in submission so evaluator digests cover inference code."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18

ACTIVE_DIMS = (0, 1, 4, 5, 6, 7, 8, 9, 10, 18)
PROPRIO_DIMS = (*range(9), *range(18, 22), *range(40, 44), 44, 45)


class SpatialSoftmax(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Conv2d(256, 32, 1)

    def forward(self, features):
        features = self.project(features)
        h, w = features.shape[-2:]
        # Accumulate probabilities in fp32 even under autocast.
        probabilities = features.flatten(2).float().softmax(-1)
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=features.device),
            torch.linspace(-1, 1, w, device=features.device),
            indexing="ij",
        )
        return torch.stack(
            (probabilities @ x.flatten(), probabilities @ y.flatten()), dim=-1
        ).flatten(1)


def random_shift(images, pad=4):
    """DrQ replication padding + independent integer translation per camera image."""
    n, _, h, w = images.shape
    padded = F.pad(images, (pad, pad, pad, pad), mode="replicate")
    offsets = torch.randint(2 * pad + 1, (n, 2), device=images.device)
    ys = offsets[:, 0, None, None] + torch.arange(h, device=images.device)[None, :, None]
    xs = offsets[:, 1, None, None] + torch.arange(w, device=images.device)[None, None, :]
    return padded.permute(0, 2, 3, 1)[
        torch.arange(n, device=images.device)[:, None, None], ys, xs
    ].permute(0, 3, 1, 2)


class BCModel(nn.Module):
    def __init__(self, horizon=20, pretrained=True):
        super().__init__()
        if not 1 <= horizon <= 50:
            raise ValueError("horizon must be in [1, 50]")
        self.horizon = horizon
        network = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.backbone = nn.Sequential(*list(network.children())[:7])  # through layer3
        self.spatial = SpatialSoftmax()
        self.camera_embedding = nn.Embedding(3, 16)
        self.proprio = nn.Sequential(nn.Linear(len(PROPRIO_DIMS), 128), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(3 * (64 + 16) + 128, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, horizon * len(ACTIVE_DIMS)),
        )
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])
        self.train()

    def train(self, mode=True):
        super().train(mode)
        # Freeze ImageNet BN statistics: small correlated robot batches give noisy
        # estimates. Affine parameters and all convolution weights still train.
        for layer in self.backbone.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
        return self

    def forward(self, rgb, proprio):
        batch = rgb.shape[0]
        images = rgb.reshape(-1, 3, 84, 84).float().div(255)
        if self.training:
            images = random_shift(images)
        images = (images - self.image_mean) / self.image_std
        features = self.spatial(self.backbone(images)).reshape(batch, 3, 64)
        cameras = self.camera_embedding.weight[None].expand(batch, -1, -1)
        vision = torch.cat((features, cameras), dim=-1).flatten(1)
        return self.head(torch.cat((vision, self.proprio(proprio)), dim=-1)).reshape(
            batch, self.horizon, len(ACTIVE_DIMS)
        )


def masked_l1(prediction, target, pad_mask):
    """True means padding; padding contributes neither loss nor denominator."""
    valid = (~pad_mask).unsqueeze(-1)
    return ((prediction - target).abs() * valid).sum() / (
        valid.sum().clamp_min(1) * prediction.shape[-1]
    )
