"""
Per-modality backbones. RGB uses standard 3-channel ResNet (optionally
ImageNet-pretrained). Thermal uses the same ResNet architecture but with the
first conv replaced to accept 1 channel, since thermal imagery here is a
single-channel radiometric/8-bit map, not a 3-channel photo.
"""

import torch
import torch.nn as nn
import torchvision


def _make_resnet(name: str, pretrained: bool, in_channels: int) -> nn.Module:
    ctor = {"resnet18": torchvision.models.resnet18, "resnet34": torchvision.models.resnet34}[name]
    weights = "DEFAULT" if pretrained else None
    net = ctor(weights=weights)

    if in_channels != 3:
        old_conv = net.conv1
        new_conv = nn.Conv2d(
            in_channels, old_conv.out_channels, kernel_size=old_conv.kernel_size,
            stride=old_conv.stride, padding=old_conv.padding, bias=False,
        )
        with torch.no_grad():
            if pretrained and in_channels == 1:
                # Average the RGB kernels down to 1 channel rather than
                # discarding the pretrained weights entirely.
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
            else:
                nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
        net.conv1 = new_conv

    return net


class TinyCNN(nn.Module):
    """Lightweight fallback backbone (no pretrained weights needed) exposing
    the same layer1..layer4 interface as torchvision ResNets, useful for
    quick iteration or CPU debugging."""

    def __init__(self, in_channels: int):
        super().__init__()

        def block(cin, cout, stride):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = block(32, 64, 1)     # stride 4 overall
        self.layer2 = block(64, 128, 2)    # stride 8
        self.layer3 = block(128, 256, 2)   # stride 16
        self.layer4 = block(256, 512, 2)   # stride 32

    def forward(self, x):
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return {"layer1": c1, "layer2": c2, "layer3": c3, "layer4": c4}


class StreamBackbone(nn.Module):
    """Wraps a ResNet (or TinyCNN) and exposes intermediate feature maps by
    name, matching the `fusion_stages` config keys."""

    OUT_CHANNELS = {
        "resnet18": {"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512},
        "resnet34": {"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512},
        "tinycnn": {"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512},
    }

    def __init__(self, arch: str, pretrained: bool, in_channels: int):
        super().__init__()
        self.arch = arch
        if arch == "tinycnn":
            self.net = TinyCNN(in_channels)
            self._forward_impl = self._forward_tiny
        else:
            self.net = _make_resnet(arch, pretrained, in_channels)
            self._forward_impl = self._forward_resnet

    def _forward_resnet(self, x):
        n = self.net
        x = n.conv1(x); x = n.bn1(x); x = n.relu(x); x = n.maxpool(x)
        c1 = n.layer1(x)
        c2 = n.layer2(c1)
        c3 = n.layer3(c2)
        c4 = n.layer4(c3)
        return {"layer1": c1, "layer2": c2, "layer3": c3, "layer4": c4}

    def _forward_tiny(self, x):
        return self.net(x)

    def forward(self, x) -> dict:
        return self._forward_impl(x)

    def out_channels(self, stage: str) -> int:
        return self.OUT_CHANNELS[self.arch][stage]
