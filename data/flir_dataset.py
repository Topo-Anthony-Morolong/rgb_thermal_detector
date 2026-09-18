"""
Dataset loader for FLIR ADAS v2 (also works for similarly structured
paired RGB/thermal COCO datasets, e.g. M3FD, with light path tweaks).

FLIR ADAS ships thermal-frame annotations (COCO json) as the primary
label source, since thermal is the "always works" sensor day and night.
The RGB image is loaded from the corresponding path and matched by
filename stem. If your RGB/thermal pairs use different stems, edit
`_match_rgb_path` below.

IMPORTANT — alignment caveat: FLIR's RGB and thermal cameras have
different FOV and are physically offset, so a thermal-frame bounding box
does not land on the exact same pixels in the RGB frame without applying
FLIR's per-scene calibration (homography). `align_mode="resize"` (the
default) just resizes both images to a common size and reuses the
thermal boxes as-is — a reasonable first baseline, but expect the fusion
to help less near frame edges until you plug in real calibration via
`align_mode="homography"`.
"""

import os
import json
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF


class FlirAdasDataset(Dataset):
    def __init__(self, root: str, thermal_ann_path: str, thermal_dir: str, rgb_dir: str,
                 img_size=(512, 640), align_mode="resize", homography_path="", train=True):
        self.root = root
        self.thermal_dir = os.path.join(root, thermal_dir)
        self.rgb_dir = os.path.join(root, rgb_dir)
        self.img_size = img_size  # (H, W)
        self.align_mode = align_mode
        self.train = train

        self.homography = None
        if align_mode == "homography":
            if not homography_path:
                raise ValueError("align_mode='homography' requires DATA.homography_path to be set")
            self.homography = np.load(homography_path)

        with open(os.path.join(root, thermal_ann_path)) as f:
            coco = json.load(f)

        self.images: Dict[int, dict] = {im["id"]: im for im in coco["images"]}
        self.cat_id_to_idx = {c["id"]: i for i, c in enumerate(sorted(coco["categories"], key=lambda c: c["id"]))}

        self.anns_by_image: Dict[int, List[dict]] = {}
        for ann in coco["annotations"]:
            self.anns_by_image.setdefault(ann["image_id"], []).append(ann)

        self.image_ids = list(self.images.keys())

    def __len__(self):
        return len(self.image_ids)

    def _match_rgb_path(self, thermal_file_name: str) -> str:
        stem = os.path.splitext(os.path.basename(thermal_file_name))[0]
        # FLIR ADAS v2 typically keeps matching stems across the thermal/RGB
        # folders (e.g. video-xxxxx-frame-000123). Try a few common extensions.
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = os.path.join(self.rgb_dir, stem + ext)
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(f"No matching RGB image found for thermal frame '{thermal_file_name}' (stem={stem})")

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        meta = self.images[img_id]

        thermal_path = os.path.join(self.thermal_dir, meta["file_name"])
        rgb_path = self._match_rgb_path(meta["file_name"])

        thermal_img = Image.open(thermal_path).convert("L")  # single channel
        rgb_img = Image.open(rgb_path).convert("RGB")

        orig_w, orig_h = thermal_img.size  # annotations are in thermal-frame coords

        if self.align_mode == "homography":
            rgb_img = self._warp_rgb(rgb_img, orig_w, orig_h)
        # else: "resize" mode just resizes both streams independently below;
        # boxes are scaled from the thermal frame, which is an approximation
        # for the RGB stream (see module docstring).

        H, W = self.img_size
        thermal_img = thermal_img.resize((W, H), Image.BILINEAR)
        rgb_img = rgb_img.resize((W, H), Image.BILINEAR)

        sx, sy = W / orig_w, H / orig_h

        boxes, labels = [], []
        for ann in self.anns_by_image.get(img_id, []):
            if ann["category_id"] not in self.cat_id_to_idx:
                continue
            x, y, w, h = ann["bbox"]  # COCO format: x, y, width, height
            x1, y1, x2, y2 = x * sx, y * sy, (x + w) * sx, (y + h) * sy
            boxes.append([x1, y1, x2, y2])
            labels.append(self.cat_id_to_idx[ann["category_id"]])

        rgb_tensor = TF.to_tensor(rgb_img)
        rgb_tensor = TF.normalize(rgb_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        thermal_tensor = TF.to_tensor(thermal_img)  # [1, H, W] in [0, 1]
        thermal_tensor = TF.normalize(thermal_tensor, mean=[0.5], std=[0.5])

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long),
            "image_id": img_id,
        }
        return rgb_tensor, thermal_tensor, target

    def _warp_rgb(self, rgb_img: Image.Image, out_w: int, out_h: int) -> Image.Image:
        import cv2
        rgb_np = np.array(rgb_img)
        warped = cv2.warpPerspective(rgb_np, self.homography, (out_w, out_h))
        return Image.fromarray(warped)


def collate_fn(batch):
    rgb = torch.stack([b[0] for b in batch])
    thermal = torch.stack([b[1] for b in batch])
    targets = [b[2] for b in batch]
    return rgb, thermal, targets
