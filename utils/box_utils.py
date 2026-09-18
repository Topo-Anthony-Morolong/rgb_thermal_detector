"""
Box math shared by anchor generation, target assignment, and inference.
Boxes are always (x1, y1, x2, y2) in absolute pixel coords unless noted.
"""

import torch


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU. boxes1: [N,4], boxes2: [M,4] -> [N,M]"""
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    cx = (boxes[..., 0] + boxes[..., 2]) / 2
    cy = (boxes[..., 1] + boxes[..., 3]) / 2
    w = boxes[..., 2] - boxes[..., 0]
    h = boxes[..., 3] - boxes[..., 1]
    return torch.stack([cx, cy, w, h], dim=-1)


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def encode_boxes(anchors: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    """RCNN-style regression targets: (dx, dy, dw, dh) relative to anchors."""
    a = xyxy_to_cxcywh(anchors)
    g = xyxy_to_cxcywh(gt_boxes)
    dx = (g[..., 0] - a[..., 0]) / a[..., 2].clamp(min=1e-6)
    dy = (g[..., 1] - a[..., 1]) / a[..., 3].clamp(min=1e-6)
    dw = torch.log((g[..., 2] / a[..., 2].clamp(min=1e-6)).clamp(min=1e-6))
    dh = torch.log((g[..., 3] / a[..., 3].clamp(min=1e-6)).clamp(min=1e-6))
    return torch.stack([dx, dy, dw, dh], dim=-1)


def decode_boxes(anchors: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    a = xyxy_to_cxcywh(anchors)
    dx, dy, dw, dh = deltas.unbind(-1)
    cx = dx * a[..., 2] + a[..., 0]
    cy = dy * a[..., 3] + a[..., 1]
    w = torch.exp(dw.clamp(max=4.0)) * a[..., 2]
    h = torch.exp(dh.clamp(max=4.0)) * a[..., 3]
    return cxcywh_to_xyxy(torch.stack([cx, cy, w, h], dim=-1))


def batched_nms(boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor, iou_thresh: float = 0.5):
    """Class-aware NMS. Returns keep indices."""
    from torchvision.ops import batched_nms as tv_nms
    return tv_nms(boxes, scores, labels, iou_thresh)
