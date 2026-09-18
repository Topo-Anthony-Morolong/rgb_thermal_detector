"""
Multi-scale anchor generation, RetinaNet-style.
Each FPN level gets `len(scales) * len(ratios)` anchors per spatial location.
"""

from typing import List, Tuple
import torch


class AnchorGenerator:
    def __init__(
        self,
        base_sizes: List[int] = (32, 64, 128),   # one base size per feature level
        scales: List[float] = (1.0, 1.26, 1.587),  # 2**(0/3), 2**(1/3), 2**(2/3)
        ratios: List[float] = (0.5, 1.0, 2.0),
        strides: List[int] = (8, 16, 32),
    ):
        assert len(base_sizes) == len(strides)
        self.base_sizes = base_sizes
        self.scales = scales
        self.ratios = ratios
        self.strides = strides
        self.num_anchors_per_loc = len(scales) * len(ratios)

    def _level_anchors(self, base_size: float) -> torch.Tensor:
        """Anchor (w, h) pairs centered at origin for one pyramid level."""
        anchors = []
        for scale in self.scales:
            area = (base_size * scale) ** 2
            for ratio in self.ratios:
                w = (area / ratio) ** 0.5
                h = w * ratio
                anchors.append([-w / 2, -h / 2, w / 2, h / 2])
        return torch.tensor(anchors, dtype=torch.float32)  # [num_anchors_per_loc, 4]

    def generate(self, feature_shapes: List[Tuple[int, int]], device="cpu") -> torch.Tensor:
        """
        feature_shapes: list of (H, W) for each FPN level, same order as strides.
        Returns: all anchors concatenated, [total_anchors, 4] in xyxy image coords.
        """
        all_anchors = []
        for (fh, fw), stride, base_size in zip(feature_shapes, self.strides, self.base_sizes):
            level_anchors = self._level_anchors(base_size).to(device)  # [A, 4]

            shift_x = (torch.arange(0, fw, device=device) + 0.5) * stride
            shift_y = (torch.arange(0, fh, device=device) + 0.5) * stride
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
            shifts = torch.stack(
                [shift_x.reshape(-1), shift_y.reshape(-1), shift_x.reshape(-1), shift_y.reshape(-1)], dim=1
            )  # [H*W, 4]

            # [H*W, 1, 4] + [1, A, 4] -> [H*W, A, 4] -> [H*W*A, 4]
            anchors = (shifts[:, None, :] + level_anchors[None, :, :]).reshape(-1, 4)
            all_anchors.append(anchors)

        return torch.cat(all_anchors, dim=0)
