"""
Anchor-target assignment (IoU-based matching) and the RetinaNet loss:
sigmoid focal loss for classification, smooth-L1 for box regression,
box loss averaged only over positive (foreground) anchors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.box_utils import box_iou, encode_boxes


def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction="sum"):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.sum() if reduction == "sum" else loss.mean()


@torch.no_grad()
def assign_targets(anchors, gt_boxes, gt_labels, num_classes,
                    pos_iou_thresh=0.5, neg_iou_thresh=0.4):
    """
    anchors: [A, 4]
    gt_boxes: [N, 4], gt_labels: [N] (0-indexed class ids)
    Returns:
      cls_targets: [A, num_classes] one-hot (all zero for background/ignored)
      box_targets: [A, 4] regression targets (only meaningful at positive anchors)
      pos_mask: [A] bool, anchors used for box loss
      ignore_mask: [A] bool, anchors excluded from classification loss entirely
    """
    A = anchors.shape[0]
    cls_targets = torch.zeros((A, num_classes), device=anchors.device)
    box_targets = torch.zeros((A, 4), device=anchors.device)
    pos_mask = torch.zeros(A, dtype=torch.bool, device=anchors.device)
    ignore_mask = torch.zeros(A, dtype=torch.bool, device=anchors.device)

    if gt_boxes.numel() == 0:
        return cls_targets, box_targets, pos_mask, ignore_mask

    iou = box_iou(anchors, gt_boxes)  # [A, N]
    max_iou, matched_gt = iou.max(dim=1)

    pos_mask = max_iou >= pos_iou_thresh
    ignore_mask = (max_iou >= neg_iou_thresh) & (~pos_mask)  # gray zone: excluded, not negative

    # Guarantee every GT box has at least one positive anchor (its best match).
    best_anchor_per_gt = iou.argmax(dim=0)
    pos_mask[best_anchor_per_gt] = True
    ignore_mask[best_anchor_per_gt] = False
    matched_gt[best_anchor_per_gt] = torch.arange(gt_boxes.shape[0], device=anchors.device)

    pos_idx = pos_mask.nonzero(as_tuple=True)[0]
    matched_labels = gt_labels[matched_gt[pos_idx]]
    cls_targets[pos_idx, matched_labels] = 1.0
    box_targets[pos_idx] = encode_boxes(anchors[pos_idx], gt_boxes[matched_gt[pos_idx]])

    return cls_targets, box_targets, pos_mask, ignore_mask


class DetectionLoss(nn.Module):
    def __init__(self, num_classes, alpha=0.25, gamma=2.0, box_weight=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.box_weight = box_weight

    def forward(self, cls_logits, box_deltas, anchors, targets: list):
        """
        cls_logits: [B, A, C], box_deltas: [B, A, 4], anchors: [A, 4]
        targets: list of length B, each a dict {"boxes": [N,4], "labels": [N]}
        """
        B = cls_logits.shape[0]
        total_cls_loss = cls_logits.new_tensor(0.0)
        total_box_loss = cls_logits.new_tensor(0.0)
        total_pos = 0

        for i in range(B):
            gt_boxes = targets[i]["boxes"].to(cls_logits.device)
            gt_labels = targets[i]["labels"].to(cls_logits.device)

            cls_t, box_t, pos_mask, ignore_mask = assign_targets(
                anchors, gt_boxes, gt_labels, self.num_classes
            )
            valid_mask = ~ignore_mask  # everything except the gray zone contributes to cls loss

            n_pos = pos_mask.sum().clamp(min=1)
            total_pos += n_pos.item()

            cls_loss = sigmoid_focal_loss(
                cls_logits[i][valid_mask], cls_t[valid_mask],
                alpha=self.alpha, gamma=self.gamma, reduction="sum",
            )
            total_cls_loss = total_cls_loss + cls_loss / n_pos

            if pos_mask.any():
                box_loss = F.smooth_l1_loss(
                    box_deltas[i][pos_mask], box_t[pos_mask], beta=0.11, reduction="sum"
                )
                total_box_loss = total_box_loss + box_loss / n_pos

        cls_loss = total_cls_loss / B
        box_loss = self.box_weight * total_box_loss / B
        return {"cls_loss": cls_loss, "box_loss": box_loss, "loss": cls_loss + box_loss}
