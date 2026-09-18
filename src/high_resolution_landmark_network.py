from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int, preferred: int = 8) -> int:
    groups = min(int(preferred), int(channels))
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return max(1, groups)


class ConvGNAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        *,
        groups: int = 8,
        activation: bool = True,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(_group_count(out_ch, groups), out_ch),
        ]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualGNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, groups: int = 8) -> None:
        super().__init__()
        self.conv1 = ConvGNAct(in_ch, out_ch, 3, groups=groups)
        self.conv2 = ConvGNAct(out_ch, out_ch, 3, groups=groups, activation=False)
        if in_ch == out_ch:
            self.skip = nn.Identity()
        else:
            self.skip = ConvGNAct(in_ch, out_ch, 1, groups=groups, activation=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv2(self.conv1(x)) + self.skip(x), inplace=True)


class HighResolutionBackbone(nn.Module):
    """Multi-scale backbone that keeps a full-resolution branch to localize small landmarks."""

    def __init__(self, in_ch: int = 1, base_ch: int = 32, *, groups: int = 8) -> None:
        super().__init__()
        c1 = int(base_ch)
        c2 = c1 * 2
        c4 = c1 * 4
        c8 = c1 * 4

        self.stem = nn.Sequential(
            ConvGNAct(in_ch, c1, 3, groups=groups),
            ResidualGNBlock(c1, c1, groups=groups),
            ResidualGNBlock(c1, c1, groups=groups),
        )
        self.down2 = ConvGNAct(c1, c2, 3, stride=2, groups=groups)
        self.stage2 = nn.Sequential(
            ResidualGNBlock(c2, c2, groups=groups),
            ResidualGNBlock(c2, c2, groups=groups),
        )
        self.down4 = ConvGNAct(c2, c4, 3, stride=2, groups=groups)
        self.stage4 = nn.Sequential(
            ResidualGNBlock(c4, c4, groups=groups),
            ResidualGNBlock(c4, c4, groups=groups),
        )
        self.down8 = ConvGNAct(c4, c8, 3, stride=2, groups=groups)
        self.stage8 = nn.Sequential(
            ResidualGNBlock(c8, c8, groups=groups),
            ResidualGNBlock(c8, c8, groups=groups),
        )

        self.proj1 = ConvGNAct(c1, c1, 1, groups=groups)
        self.proj2 = ConvGNAct(c2, c1, 1, groups=groups)
        self.proj4 = ConvGNAct(c4, c1, 1, groups=groups)
        self.proj8 = ConvGNAct(c8, c1, 1, groups=groups)
        self.fuse = nn.Sequential(
            ConvGNAct(c1 * 4, c1 * 2, 3, groups=groups),
            ResidualGNBlock(c1 * 2, c1 * 2, groups=groups),
            ResidualGNBlock(c1 * 2, c1 * 2, groups=groups),
            ConvGNAct(c1 * 2, c1, 3, groups=groups),
        )
        self.out_channels = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.stem(x)
        x2 = self.stage2(self.down2(x1))
        x4 = self.stage4(self.down4(x2))
        x8 = self.stage8(self.down8(x4))

        size = x1.shape[2:]
        fused = torch.cat(
            [
                self.proj1(x1),
                F.interpolate(self.proj2(x2), size=size, mode="bilinear", align_corners=False),
                F.interpolate(self.proj4(x4), size=size, mode="bilinear", align_corners=False),
                F.interpolate(self.proj8(x8), size=size, mode="bilinear", align_corners=False),
            ],
            dim=1,
        )
        return self.fuse(fused)


class HighResolutionLandmarkModel(nn.Module):
    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 32,
        out_ch: int = 1,
        task_head_channels: int | None = None,
    ) -> None:
        super().__init__()
        if task_head_channels is None:
            task_head_channels = base_ch
        self.backbone = HighResolutionBackbone(in_ch=in_ch, base_ch=base_ch)
        self.hm_head = nn.Sequential(
            ConvGNAct(self.backbone.out_channels, int(task_head_channels), 3),
            ResidualGNBlock(int(task_head_channels), int(task_head_channels)),
            nn.Conv2d(int(task_head_channels), out_ch, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hm_head(self.backbone(x))
