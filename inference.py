"""
Run a trained single-modality checkpoint on one image and save an
annotated result.

Usage:
    python inference.py --modality rgb --image path/to/rgb.jpg \
        --checkpoint checkpoints/rgb/best.pt --out result.jpg

Optional late fusion: if you have both an RGB and a thermal checkpoint and
(separately) a genuinely time-synced RGB+thermal image pair -- e.g. from
FLIR's video_rgb_test/video_thermal_test split -- you can run both models
and merge detections at the box level (union + NMS across both sets of
predictions) rather than fusing features. This is a much weaker form of
fusion than the feature-level approach in the original two-stream design
(models/detector.py, models/fusion.py -- kept in this repo for when you
have real paired training data), but it's a reasonable way to combine two
independently-trained detectors post-hoc. See --thermal-image/--thermal-checkpoint below.
"""

import argparse
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
from torchvision.ops import batched_nms

from config import DATA, MODEL
from models.single_modality_detector import SingleModalityDetector


def load_image(path, img_size, in_channels):
    H, W = img_size
    mode = "RGB" if in_channels == 3 else "L"
    img = Image.open(path).convert(mode).resize((W, H), Image.BILINEAR)

    tensor = TF.to_tensor(img)
    if in_channels == 3:
        tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        tensor = TF.normalize(tensor, mean=[0.5], std=[0.5])

    display_img = Image.open(path).convert("RGB").resize((W, H), Image.BILINEAR)
    return display_img, tensor.unsqueeze(0)


def load_model(checkpoint_path, in_channels, device):
    model = SingleModalityDetector(MODEL, DATA, in_channels, pretrained=False).to(device).eval()
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    return model


def draw_results(display_img, results, label_prefix=""):
    draw = ImageDraw.Draw(display_img)
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        x1, y1, x2, y2 = box.tolist()
        name = DATA.class_names[label.item()]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, max(0, y1 - 10)), f"{label_prefix}{name} {score:.2f}", fill="red")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=["rgb", "thermal"], required=True,
                         help="Primary modality to run.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="result.jpg")
    parser.add_argument("--score-thresh", type=float, default=0.4)

    # optional late-fusion pass with a second, genuinely paired image
    parser.add_argument("--fuse-modality", choices=["rgb", "thermal"], default=None)
    parser.add_argument("--fuse-image", default=None)
    parser.add_argument("--fuse-checkpoint", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modality_cfg = DATA.modality(args.modality)

    model = load_model(args.checkpoint, modality_cfg.in_channels, device)
    display_img, tensor = load_image(args.image, DATA.img_size, modality_cfg.in_channels)
    tensor = tensor.to(device)
    results = model.predict(tensor, score_thresh=args.score_thresh)[0]

    all_boxes, all_scores, all_labels = [results["boxes"]], [results["scores"]], [results["labels"]]

    if args.fuse_modality:
        assert args.fuse_image and args.fuse_checkpoint, "--fuse-image and --fuse-checkpoint required with --fuse-modality"
        fuse_cfg = DATA.modality(args.fuse_modality)
        fuse_model = load_model(args.fuse_checkpoint, fuse_cfg.in_channels, device)
        _, fuse_tensor = load_image(args.fuse_image, DATA.img_size, fuse_cfg.in_channels)
        fuse_results = fuse_model.predict(fuse_tensor.to(device), score_thresh=args.score_thresh)[0]
        all_boxes.append(fuse_results["boxes"])
        all_scores.append(fuse_results["scores"])
        all_labels.append(fuse_results["labels"])

    boxes = torch.cat(all_boxes)
    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)
    if boxes.shape[0] > 0:
        keep = batched_nms(boxes, scores, labels, iou_threshold=0.5)
        results = {"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]}
    else:
        results = {"boxes": boxes, "scores": scores, "labels": labels}

    draw_results(display_img, results)
    display_img.save(args.out)
    print(f"Saved {len(results['boxes'])} detections to {args.out}")


if __name__ == "__main__":
    main()
