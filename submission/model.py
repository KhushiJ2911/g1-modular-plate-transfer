"""Shared BC network; kept in submission so evaluator digests cover inference code."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18

ACTIVE_DIMS = (0, 1, 4, 5, 6, 7, 8, 9, 10, 18)
PROPRIO_DIMS = (*range(9), *range(18, 22), *range(40, 44), 44, 45)


def validate_proprio_dims(dims):
    dims = tuple(dims)
    if (
        not dims
        or any(type(i) is not int or not 0 <= i < 50 for i in dims)
        or len(set(dims)) != len(dims)
    ):
        raise ValueError("proprio_dims must be unique integer state indices in [0, 50)")
    return dims


def load_bc_weights(model, state):
    """Allow only the optional auxiliary head to differ between checkpoints."""
    state = dict(state)
    aux_keys = {"aux_head.weight", "aux_head.bias"}
    if aux_keys.intersection(state) not in (set(), aux_keys):
        raise ValueError("Checkpoint has an incomplete auxiliary head")
    if model.aux_head is None:
        state.pop("aux_head.weight", None)
        state.pop("aux_head.bias", None)
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed = aux_keys if model.aux_head is not None else set()
    if set(missing) - allowed or unexpected:
        raise ValueError(f"Checkpoint weights mismatch: missing={missing}, unexpected={unexpected}")


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
    def __init__(self, horizon=20, pretrained=True, proprio_dims=PROPRIO_DIMS, aux_head=False):
        super().__init__()
        if not 1 <= horizon <= 50:
            raise ValueError("horizon must be in [1, 50]")
        self.horizon = horizon
        self.proprio_dims = validate_proprio_dims(proprio_dims)
        network = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.backbone = nn.Sequential(*list(network.children())[:7])  # through layer3
        self.spatial = SpatialSoftmax()
        self.camera_embedding = nn.Embedding(3, 16)
        self.proprio = nn.Sequential(nn.Linear(len(self.proprio_dims), 128), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(3 * (64 + 16) + 128, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, horizon * len(ACTIVE_DIMS)),
        )
        self.aux_head = nn.Linear(3 * (64 + 16) + 128, 9) if aux_head else None
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

    def forward(self, rgb, proprio, *, return_aux=False):
        batch = rgb.shape[0]
        images = rgb.reshape(-1, 3, 84, 84).float().div(255)
        if self.training:
            images = random_shift(images)
        images = (images - self.image_mean) / self.image_std
        images = images.contiguous(memory_format=torch.channels_last)
        features = self.spatial(self.backbone(images)).reshape(batch, 3, 64)
        cameras = self.camera_embedding.weight[None].expand(batch, -1, -1)
        vision = torch.cat((features, cameras), dim=-1).flatten(1)
        fused = torch.cat((vision, self.proprio(proprio)), dim=-1)
        actions = self.head(fused).reshape(batch, self.horizon, len(ACTIVE_DIMS))
        if return_aux:
            return actions, self.aux_head(fused) if self.aux_head is not None else None
        return actions


def masked_l1(prediction, target, pad_mask, label_mask=None):
    """Mean over supervised, nonpadding elements; an empty mask has zero loss."""
    valid = (~pad_mask).unsqueeze(-1).expand_as(prediction)
    if label_mask is not None:
        valid = valid & label_mask
    return ((prediction - target).abs() * valid).sum() / valid.sum().clamp_min(1)


def auxiliary_loss(prediction, target, mask):
    """Mean over labeled dimensions: eight MSE terms and a held BCE logit."""
    errors = torch.cat(
        (
            (prediction[..., :8] - target[..., :8]).square(),
            F.binary_cross_entropy_with_logits(
                prediction[..., 8:], target[..., 8:], reduction="none"
            ),
        ),
        dim=-1,
    )
    return (errors * mask).sum() / mask.sum().clamp_min(1)
