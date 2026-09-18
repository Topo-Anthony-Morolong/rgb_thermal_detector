"""
Central configuration for the RGB+Thermal two-stream detector.
Edit paths and hyperparameters here rather than scattering magic numbers
through the codebase.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DataConfig:
    # FLIR ADAS v2 layout (adjust to match your download):
    #   root/images_rgb_train/data/*.jpg
    #   root/images_thermal_train/data/*.jpg (or .tiff/.png, 8-bit thermal)
    #   root/images_thermal_train/coco.json      <- primary annotations (thermal FOV)
    #   root/images_rgb_train/coco.json          <- optional, only used if align_mode="none"
    root: str = "/path/to/FLIR_ADAS_v2"
    train_thermal_ann: str = "images_thermal_train/coco.json"
    train_thermal_dir: str = "images_thermal_train/data"
    train_rgb_dir: str = "images_rgb_train/data"

    val_thermal_ann: str = "images_thermal_val/coco.json"
    val_thermal_dir: str = "images_thermal_val/data"
    val_rgb_dir: str = "images_rgb_val/data"

    # FLIR's RGB and thermal sensors have different FOV/resolution and are
    # NOT pixel-aligned out of the box. Options:
    #   "resize"  - naive resize-to-match (fast, works for a first baseline,
    #               but boxes drawn from thermal annotations will be slightly
    #               off in the RGB frame near image edges)
    #   "homography" - supply a 3x3 homography (per-camera, from FLIR's
    #               calibration files) to warp RGB into the thermal frame
    #               before resizing. Set `homography_path` below if used.
    align_mode: str = "resize"
    homography_path: str = ""  # numpy .npy file with a 3x3 matrix, if align_mode="homography"

    img_size: Tuple[int, int] = (512, 640)  # (H, W), fed to both streams
    num_classes: int = 3  # FLIR ADAS core classes: person, bicycle, car (extend as needed)
    class_names: List[str] = field(default_factory=lambda: ["person", "bicycle", "car"])


@dataclass
class ModelConfig:
    backbone: str = "resnet18"  # "resnet18" | "resnet34" | "tinycnn"
    pretrained_rgb: bool = True     # ImageNet weights make sense for the 3-channel RGB stream
    pretrained_thermal: bool = False  # thermal is single-channel; ImageNet weights don't transfer directly
    fusion_stages: Tuple[str, ...] = ("layer2", "layer3", "layer4")  # which backbone stages to fuse
    fusion_type: str = "attention"  # "concat" | "add" | "attention"
    fpn_channels: int = 128
    num_anchors_per_loc: int = 9  # 3 scales x 3 aspect ratios


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    device: str = "cuda"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    box_loss_weight: float = 1.0
    grad_clip_norm: float = 10.0
    checkpoint_dir: str = "./checkpoints"
    log_every: int = 20
    val_every: int = 1
    amp: bool = True  # mixed precision


DATA = DataConfig()
MODEL = ModelConfig()
TRAIN = TrainConfig()
