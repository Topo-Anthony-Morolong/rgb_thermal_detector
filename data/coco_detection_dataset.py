"""
Plain single-modality COCO-format detection dataset. Works for either
FLIR ADAS's images_rgb_* or images_thermal_* sets -- each is used as a
normal, self-contained annotated image set, no RGB/thermal pairing
involved.
"""

import os
import json
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF


class CocoDetectionDataset(Dataset):
    def __init__(self, root: str, ann_path: str, image_dir: str,
                 img_size=(512, 640), in_channels: int = 3):
        self.image_dir = os.path.join(root, image_dir)
        self.img_size = img_size  # (H, W)
        self.in_channels = in_channels

        with open(os.path.join(root, ann_path)) as f:
            coco = json.load(f)

        self.images: Dict[int, dict] = {im["id"]: im for im in coco["images"]}
        self.cat_id_to_idx = {c["id"]: i for i, c in enumerate(sorted(coco["categories"], key=lambda c: c["id"]))}

        self.anns_by_image: Dict[int, List[dict]] = {}
        for ann in coco["annotations"]:
            self.anns_by_image.setdefault(ann["image_id"], []).append(ann)

        self.image_ids = list(self.images.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        meta = self.images[img_id]

        # coco.json file_name entries already include the "data/" subdir
        # prefix (e.g. "data/video-xxx-frame-000108-xxx.jpg"), so join
        # directly against image_dir's parent, not image_dir itself.
        img_path = os.path.join(os.path.dirname(self.image_dir), meta["file_name"])

        mode = "RGB" if self.in_channels == 3 else "L"
        img = Image.open(img_path).convert(mode)
        orig_w, orig_h = img.size

        H, W = self.img_size
        img = img.resize((W, H), Image.BILINEAR)
        sx, sy = W / orig_w, H / orig_h

        boxes, labels = [], []
        for ann in self.anns_by_image.get(img_id, []):
            if ann["category_id"] not in self.cat_id_to_idx:
                continue
            x, y, w, h = ann["bbox"]  # COCO format: x, y, width, height
            x1, y1, x2, y2 = x * sx, y * sy, (x + w) * sx, (y + h) * sy
            boxes.append([x1, y1, x2, y2])
            labels.append(self.cat_id_to_idx[ann["category_id"]])

        img_tensor = TF.to_tensor(img)
        if self.in_channels == 3:
            img_tensor = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            img_tensor = TF.normalize(img_tensor, mean=[0.5], std=[0.5])

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long),
            "image_id": img_id,
        }
        return img_tensor, target


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return imgs, targets
