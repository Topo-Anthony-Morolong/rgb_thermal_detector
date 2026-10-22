"""Shared FLIR image preprocessing used by training and inference."""

from typing import Tuple

import torch
from PIL import Image
import torchvision.transforms.functional as TF


RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)
THERMAL_MEAN = (0.5,)
THERMAL_STD = (0.5,)
LETTERBOX_VALUE = 114


def letterbox_image(image: Image.Image, img_size: Tuple[int, int], in_channels: int):
    """Resize without changing aspect ratio and pad to (height, width).

    Returns the padded PIL image and the exact resize/padding values needed to
    transform COCO boxes consistently (sx, sy, pad_left, pad_top).
    """
    height, width = img_size
    mode = "RGB" if in_channels == 3 else "L"
    image = image.convert(mode)
    orig_w, orig_h = image.size

    scale = min(width / orig_w, height / orig_h)
    resized_w = max(1, round(orig_w * scale))
    resized_h = max(1, round(orig_h * scale))
    resampling = getattr(Image, "Resampling", Image)
    resized = image.resize((resized_w, resized_h), resampling.BILINEAR)

    pad_left = (width - resized_w) // 2
    pad_top = (height - resized_h) // 2
    fill = (LETTERBOX_VALUE,) * 3 if in_channels == 3 else LETTERBOX_VALUE
    canvas = Image.new(mode, (width, height), color=fill)
    canvas.paste(resized, (pad_left, pad_top))

    # Use actual rounded output dimensions so the annotation transform exactly
    # matches the pixels produced by PIL.resize.
    sx, sy = resized_w / orig_w, resized_h / orig_h
    return canvas, (sx, sy, pad_left, pad_top)


def image_to_tensor(image: Image.Image, in_channels: int) -> torch.Tensor:
    """Convert a letterboxed RGB or 8-bit thermal image to normalized tensor."""
    tensor = TF.to_tensor(image)
    if in_channels == 3:
        return TF.normalize(tensor, mean=RGB_MEAN, std=RGB_STD)
    return TF.normalize(tensor, mean=THERMAL_MEAN, std=THERMAL_STD)

