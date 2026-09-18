"""
Run a trained checkpoint on a single RGB+thermal image pair and save an
annotated image.

Usage:
    python inference.py --rgb path/to/rgb.jpg --thermal path/to/thermal.jpg \
        --checkpoint checkpoints/best.pt --out result.jpg
"""

import argparse
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw

from config import DATA, MODEL
from models.detector import RGBThermalDetector


def load_pair(rgb_path, thermal_path, img_size):
    H, W = img_size
    rgb = Image.open(rgb_path).convert("RGB").resize((W, H), Image.BILINEAR)
    thermal = Image.open(thermal_path).convert("L").resize((W, H), Image.BILINEAR)

    rgb_t = TF.normalize(TF.to_tensor(rgb), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    thermal_t = TF.normalize(TF.to_tensor(thermal), mean=[0.5], std=[0.5])
    return rgb, rgb_t.unsqueeze(0), thermal_t.unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", required=True)
    parser.add_argument("--thermal", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="result.jpg")
    parser.add_argument("--score-thresh", type=float, default=0.4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RGBThermalDetector(MODEL, DATA).to(device).eval()

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])

    display_img, rgb_t, thermal_t = load_pair(args.rgb, args.thermal, DATA.img_size)
    rgb_t, thermal_t = rgb_t.to(device), thermal_t.to(device)

    results = model.predict(rgb_t, thermal_t, score_thresh=args.score_thresh)[0]

    draw = ImageDraw.Draw(display_img)
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        x1, y1, x2, y2 = box.tolist()
        name = DATA.class_names[label.item()]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, max(0, y1 - 10)), f"{name} {score:.2f}", fill="red")

    display_img.save(args.out)
    print(f"Saved {len(results['boxes'])} detections to {args.out}")


if __name__ == "__main__":
    main()
