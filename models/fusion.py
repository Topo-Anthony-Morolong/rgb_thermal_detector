"""
Feature-level fusion between the RGB and thermal streams.

Three strategies, selected via config.MODEL.fusion_type:

  "concat"    - channel-concat the two streams, 1x1 conv back down to the
                original channel count. Simple, strong baseline.

  "add"       - project each stream to a shared channel count and sum.
                Cheapest option, assumes both modalities are similarly
                reliable at every location (often not true, see below).

  "attention" - learn a per-pixel gate (0..1) from the concatenated features
                that decides how much to trust RGB vs. thermal at each
                spatial location. This matters in practice: thermal is far
                more reliable at night / in glare, RGB is more reliable for
                fine texture and color cues in daylight. A fixed 50/50 blend
                throws away exactly the signal that makes multispectral
                fusion useful.
"""

import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb_feat, thermal_feat):
        return self.reduce(torch.cat([rgb_feat, thermal_feat], dim=1))


class AddFusion(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.proj_rgb = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.proj_thermal = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, rgb_feat, thermal_feat):
        return torch.relu(self.bn(self.proj_rgb(rgb_feat) + self.proj_thermal(thermal_feat)))


class AttentionFusion(nn.Module):
    """Per-pixel gated fusion. gate=1 -> all RGB, gate=0 -> all thermal."""

    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb_feat, thermal_feat):
        g = self.gate(torch.cat([rgb_feat, thermal_feat], dim=1))
        fused = g * rgb_feat + (1 - g) * thermal_feat
        return self.out_conv(fused)


def build_fusion(fusion_type: str, channels: int) -> nn.Module:
    return {
        "concat": ConcatFusion,
        "add": AddFusion,
        "attention": AttentionFusion,
    }[fusion_type](channels)


class MultiStageFusion(nn.Module):
    """Applies one fusion module per requested backbone stage, and projects
    every stage to a common channel count (`out_channels`) so the FPN can
    combine them regardless of the backbone's native channel widths."""

    def __init__(self, stage_channels: dict, fusion_type: str, out_channels: int):
        super().__init__()
        self.stages = list(stage_channels.keys())
        self.pre_proj_rgb = nn.ModuleDict({
            s: nn.Conv2d(c, out_channels, kernel_size=1) for s, c in stage_channels.items()
        })
        self.pre_proj_thermal = nn.ModuleDict({
            s: nn.Conv2d(c, out_channels, kernel_size=1) for s, c in stage_channels.items()
        })
        self.fusions = nn.ModuleDict({
            s: build_fusion(fusion_type, out_channels) for s in stage_channels
        })

    def forward(self, rgb_feats: dict, thermal_feats: dict) -> dict:
        fused = {}
        for s in self.stages:
            r = self.pre_proj_rgb[s](rgb_feats[s])
            t = self.pre_proj_thermal[s](thermal_feats[s])
            fused[s] = self.fusions[s](r, t)
        return fused
