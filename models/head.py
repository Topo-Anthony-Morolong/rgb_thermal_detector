"""
FPN neck (top-down pathway with lateral connections) plus a shared
RetinaNet-style head: parallel conv towers for classification and box
regression, applied identically at every pyramid level.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FPNNeck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, levels=("layer2", "layer3", "layer4")):
        super().__init__()
        self.levels = levels
        self.lateral = nn.ModuleDict({
            lv: nn.Conv2d(in_channels, out_channels, kernel_size=1) for lv in levels
        })
        self.smooth = nn.ModuleDict({
            lv: nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for lv in levels
        })

    def forward(self, feats: dict) -> list:
        laterals = {lv: self.lateral[lv](feats[lv]) for lv in self.levels}

        # top-down pathway: coarsest level first
        ordered = list(self.levels)
        for i in range(len(ordered) - 2, -1, -1):
            hi, lo = ordered[i + 1], ordered[i]
            up = F.interpolate(laterals[hi], size=laterals[lo].shape[-2:], mode="nearest")
            laterals[lo] = laterals[lo] + up

        return [self.smooth[lv](laterals[lv]) for lv in ordered]


class DetectionHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, num_anchors: int, num_convs: int = 4):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        def tower():
            layers = []
            for _ in range(num_convs):
                layers += [
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(32, in_channels),
                    nn.ReLU(inplace=True),
                ]
            return nn.Sequential(*layers)

        self.cls_tower = tower()
        self.box_tower = tower()
        self.cls_out = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, padding=1)
        self.box_out = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, padding=1)

        # Focal-loss init: start classification bias so that, pre-training,
        # every anchor predicts a low foreground probability (~0.01). Without
        # this the loss is dominated by the ~thousands of easy negatives in
        # the first few hundred steps.
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_out.bias, bias_value)

    def forward(self, feature_maps: list):
        cls_logits, box_deltas = [], []
        for f in feature_maps:
            c = self.cls_out(self.cls_tower(f))
            b = self.box_out(self.box_tower(f))

            B, _, H, W = c.shape
            c = c.permute(0, 2, 3, 1).reshape(B, H * W * self.num_anchors, self.num_classes)
            b = b.permute(0, 2, 3, 1).reshape(B, H * W * self.num_anchors, 4)
            cls_logits.append(c)
            box_deltas.append(b)

        return torch.cat(cls_logits, dim=1), torch.cat(box_deltas, dim=1)
