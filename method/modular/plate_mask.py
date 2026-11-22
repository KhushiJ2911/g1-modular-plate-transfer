"""Small RGB segmenter for the held plate in the head and left-wrist cameras."""

import torch
from torch import nn
from torch.nn import functional as F


def _block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.GroupNorm(4, out_channels),
        nn.SiLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.GroupNorm(4, out_channels),
        nn.SiLU(),
    )


class PlateMask(nn.Module):
    """The fourth channel identifies the camera, zero for head and one for wrist."""

    def __init__(self):
        super().__init__()
        self.full = _block(4, 12)
        self.half_resolution = _block(12, 20)
        self.quarter = _block(20, 32)
        self.eighth = _block(32, 40)
        self.up_quarter = _block(40 + 32, 24)
        self.up_half = _block(24 + 20, 16)
        self.up_full = _block(16 + 12, 12)
        self.mask = nn.Conv2d(12, 1, 1)

    def forward(self, rgb, camera):
        camera = camera.reshape(-1, 1, 1, 1).expand(-1, 1, *rgb.shape[-2:])
        full = self.full(torch.cat((rgb.float() / 255, camera.float()), dim=1))
        half = self.half_resolution(F.avg_pool2d(full, 2))
        quarter = self.quarter(F.avg_pool2d(half, 2))
        x = self.eighth(F.avg_pool2d(quarter, 2))
        for block, skip in ((self.up_quarter, quarter), (self.up_half, half), (self.up_full, full)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat((x, skip), dim=1))
        return self.mask(x)[:, 0]
