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

from utils.image_preprocessing import image_to_tensor, letterbox_image


class CocoDetectionDataset(Dataset):
    def __init__(self, root: str, ann_path: str, image_dir: str,
                 img_size=(512, 640), in_channels: int = 3, augment: bool = False):
        self.image_dir = os.path.join(root, image_dir)
        self.img_size = img_size  # (H, W)
        self.in_channels = in_channels
        self.augment = augment

        with open(os.path.join(root, ann_path)) as f:
            coco = json.load(f)

        self.images: Dict[int, dict] = {im["id"]: im for im in coco["images"]}

        # Explicitly map only the 3 target classes: person, bike (bicycle), and car
        target_categories = {"person": 0, "bike": 1, "bicycle": 1, "car": 2}

        self.cat_id_to_idx = {}
        for cat in coco["categories"]:
            name = cat["name"].lower()
            if name in target_categories:
                self.cat_id_to_idx[cat["id"]] = target_categories[name]

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

        img = Image.open(img_path)
        img, (sx, sy, pad_left, pad_top) = letterbox_image(img, self.img_size, self.in_channels)

        boxes, labels = [], []
        for ann in self.anns_by_image.get(img_id, []):
            if ann["category_id"] not in self.cat_id_to_idx:
                continue
            x, y, w, h = ann["bbox"]  # COCO format: x, y, width, height
            x1 = x * sx + pad_left
            y1 = y * sy + pad_top
            x2 = (x + w) * sx + pad_left
            y2 = (y + h) * sy + pad_top
            boxes.append([x1, y1, x2, y2])
            labels.append(self.cat_id_to_idx[ann["category_id"]])

        if self.augment:
            if torch.rand(()) < 0.5:
                img = TF.hflip(img)
                image_width = img.width
                boxes = [[image_width - x2, y1, image_width - x1, y2]
                         for x1, y1, x2, y2 in boxes]

            if self.in_channels == 1 and torch.rand(()) < 0.5:
                brightness = float(torch.empty(1).uniform_(0.85, 1.15).item())
                contrast = float(torch.empty(1).uniform_(0.8, 1.2).item())
                img = TF.adjust_brightness(img, brightness)
                img = TF.adjust_contrast(img, contrast)

        img_tensor = image_to_tensor(img, self.in_channels)

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
