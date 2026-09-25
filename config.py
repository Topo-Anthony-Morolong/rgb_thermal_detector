"""
Central configuration for two independent single-modality detectors
(one RGB, one thermal) trained on FLIR ADAS v2's images_rgb_train and
images_thermal_train sets as-is.

Why two separate models instead of one fused model: FLIR ADAS v2's
images_thermal_train/images_rgb_train are NOT frame-paired (confirmed via
diagnose_pairing.py -- separate independently-sampled stills, 0% filename
overlap, no cross-modal pointer in the annotation metadata). Real
frame-synced pairs only exist in the much smaller video_rgb_test /
video_thermal_test split. So rather than force a fusion architecture onto
data that isn't actually paired, we train one detector per modality on the
full annotated set each modality actually has.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ModalityConfig:
    """Paths + settings for one modality (RGB or thermal)."""
    name: str                      # "rgb" | "thermal" -- used in checkpoint/log naming
    in_channels: int                # 3 for RGB, 1 for thermal
    train_ann: str
    train_dir: str
    val_ann: str
    val_dir: str
    pretrained: bool                # ImageNet weights make sense for RGB, not for raw thermal


@dataclass
class DataConfig:
    # FLIR ADAS v2 layout (adjust to match your download):
    #   root/images_rgb_train/{data/, coco.json}
    #   root/images_thermal_train/{data/, coco.json}
    #   root/images_rgb_val/{data/, coco.json}
    #   root/images_thermal_val/{data/, coco.json}
    root: str = "/path/to/FLIR_ADAS_v2"

    img_size: Tuple[int, int] = (512, 640)  # (H, W)
    num_classes: int = 3  # FLIR ADAS core classes: person, bicycle, car (extend as needed)
    class_names: List[str] = field(default_factory=lambda: ["person", "bicycle", "car"])

    rgb: ModalityConfig = field(default_factory=lambda: ModalityConfig(
        name="rgb", in_channels=3,
        train_ann="images_rgb_train/coco.json", train_dir="images_rgb_train/data",
        val_ann="images_rgb_val/coco.json", val_dir="images_rgb_val/data",
        pretrained=True,
    ))
    thermal: ModalityConfig = field(default_factory=lambda: ModalityConfig(
        name="thermal", in_channels=1,
        train_ann="images_thermal_train/coco.json", train_dir="images_thermal_train/data",
        val_ann="images_thermal_val/coco.json", val_dir="images_thermal_val/data",
        pretrained=False,
    ))

    def modality(self, name: str) -> ModalityConfig:
        return {"rgb": self.rgb, "thermal": self.thermal}[name]


@dataclass
class ModelConfig:
    backbone: str = "resnet18"  # "resnet18" | "resnet34" | "tinycnn"
    fpn_stages: Tuple[str, ...] = ("layer2", "layer3", "layer4")
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
