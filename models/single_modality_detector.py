"""
Single-modality detector: one backbone -> FPN neck -> shared detection head.
Same RetinaNet-style architecture as before, minus the cross-modal fusion
step -- used to train an RGB-only and a thermal-only model independently,
since FLIR ADAS v2's RGB/thermal train sets aren't frame-paired (see
diagnose_pairing.py and the README).
"""

import torch
import torch.nn as nn

from models.backbone import StreamBackbone
from models.head import FPNNeck, DetectionHead
from utils.anchors import AnchorGenerator
from utils.box_utils import decode_boxes, batched_nms


class SingleModalityDetector(nn.Module):
    def __init__(self, model_cfg, data_cfg, in_channels: int, pretrained: bool):
        super().__init__()
        self.fpn_stages = list(model_cfg.fpn_stages)
        self.num_classes = data_cfg.num_classes

        self.backbone = StreamBackbone(model_cfg.backbone, pretrained, in_channels=in_channels)

        stage_channels = {s: self.backbone.out_channels(s) for s in self.fpn_stages}
        self.lateral_proj = nn.ModuleDict({
            s: nn.Conv2d(c, model_cfg.fpn_channels, kernel_size=1) for s, c in stage_channels.items()
        })

        self.neck = FPNNeck(model_cfg.fpn_channels, model_cfg.fpn_channels, levels=self.fpn_stages)
        self.head = DetectionHead(model_cfg.fpn_channels, self.num_classes, model_cfg.num_anchors_per_loc)

        strides = {"layer1": 4, "layer2": 8, "layer3": 16, "layer4": 32}
        self.anchor_gen = AnchorGenerator(
            base_sizes=[strides[s] * 4 for s in self.fpn_stages],  # base size ~4x stride, standard RetinaNet choice
            strides=[strides[s] for s in self.fpn_stages],
        )
        self._anchor_cache = {}

    def _get_anchors(self, feature_maps, device):
        shapes = tuple(f.shape[-2:] for f in feature_maps)
        key = (shapes, device)
        if key not in self._anchor_cache:
            self._anchor_cache[key] = self.anchor_gen.generate([tuple(s) for s in shapes], device=device)
        return self._anchor_cache[key]

    def forward(self, x: torch.Tensor):
        feats = self.backbone(x)
        projected = {s: self.lateral_proj[s](feats[s]) for s in self.fpn_stages}
        feature_maps = self.neck(projected)
        cls_logits, box_deltas = self.head(feature_maps)
        anchors = self._get_anchors(feature_maps, x.device)
        return cls_logits, box_deltas, anchors

    @torch.no_grad()
    def predict(self, x, score_thresh=0.3, nms_thresh=0.5, max_detections=100):
        """Returns a list (len == batch) of dicts: {boxes, scores, labels}."""
        cls_logits, box_deltas, anchors = self.forward(x)
        scores_all = torch.sigmoid(cls_logits)  # [B, A, C]

        results = []
        for i in range(x.shape[0]):
            scores, labels = scores_all[i].max(dim=1)
            keep = scores > score_thresh
            if keep.sum() == 0:
                results.append({"boxes": torch.zeros(0, 4), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.long)})
                continue

            boxes = decode_boxes(anchors[keep], box_deltas[i][keep])
            boxes[:, 0::2] = boxes[:, 0::2].clamp(0, x.shape[-1])
            boxes[:, 1::2] = boxes[:, 1::2].clamp(0, x.shape[-2])

            keep_idx = batched_nms(boxes, scores[keep], labels[keep], nms_thresh)[:max_detections]
            results.append({
                "boxes": boxes[keep_idx].cpu(),
                "scores": scores[keep][keep_idx].cpu(),
                "labels": labels[keep][keep_idx].cpu(),
            })
        return results
