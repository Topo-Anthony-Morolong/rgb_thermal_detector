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
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision
from PIL import Image, ImageDraw
from torchvision.ops import batched_nms

from config import DATA, MODEL
from data.coco_detection_dataset import CocoDetectionDataset, collate_fn
from models.single_modality_detector import SingleModalityDetector
from utils.image_preprocessing import image_to_tensor, letterbox_image


def load_image(path, img_size, in_channels):
    source = Image.open(path)
    display_img, _ = letterbox_image(source, img_size, in_channels=3)
    model_img, _ = letterbox_image(source, img_size, in_channels=in_channels)
    tensor = image_to_tensor(model_img, in_channels)
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


@torch.no_grad()
def run_coco(args, model, device, modality_cfg):
    ann_path = Path(args.coco_json).resolve()
    split_dir = ann_path.parent
    dataset = CocoDetectionDataset(
        root=str(split_dir.parent), ann_path=str(ann_path.relative_to(split_dir.parent)),
        image_dir=str(Path(split_dir.name) / "data"), img_size=DATA.img_size,
        in_channels=modality_cfg.in_channels,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, collate_fn=collate_fn)
    coco = json.loads(ann_path.read_text())
    label_to_category = {}
    names_to_labels = {"person": 0, "bike": 1, "bicycle": 1, "car": 2}
    for category in coco["categories"]:
        label = names_to_labels.get(category["name"].lower())
        if label is not None:
            label_to_category[label] = int(category["id"])
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric = MeanAveragePrecision(box_format="xyxy")
    coco_predictions = []
    model.eval()

    for imgs, targets in loader:
        predictions = model.predict(imgs.to(device), score_thresh=args.score_thresh)
        metric.update(predictions, [{"boxes": t["boxes"], "labels": t["labels"]} for t in targets])
        for prediction, target in zip(predictions, targets):
            image_id = int(target["image_id"])
            meta = dataset.images[image_id]
            src_h, src_w = int(meta["height"]), int(meta["width"])
            dst_h, dst_w = DATA.img_size
            scale = min(dst_w / src_w, dst_h / src_h)
            pad_x = (dst_w - src_w * scale) / 2
            pad_y = (dst_h - src_h * scale) / 2
            for (x1, y1, x2, y2), score, label in zip(
                    prediction["boxes"].cpu().tolist(), prediction["scores"].cpu().tolist(),
                    prediction["labels"].cpu().tolist()):
                x1, x2 = (x1 - pad_x) / scale, (x2 - pad_x) / scale
                y1, y2 = (y1 - pad_y) / scale, (y2 - pad_y) / scale
                x1, x2 = max(0.0, x1), min(float(src_w), x2)
                y1, y2 = max(0.0, y1), min(float(src_h), y2)
                coco_predictions.append({
                    "image_id": image_id,
                    "category_id": label_to_category[int(label)],
                    "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                    "score": float(score),
                })

            image_path = Path(dataset.image_dir).parent / meta["file_name"]
            display_img, _ = load_image(str(image_path), DATA.img_size, modality_cfg.in_channels)
            draw_results(display_img, prediction)
            display_img.save(output_dir / f"{image_id}_{Path(meta['file_name']).stem}.jpg")

    scores = {k: float(v) for k, v in metric.compute().items()
              if k in ("map", "map_50", "map_75", "mar_100")}
    (output_dir / "predictions.json").write_text(json.dumps(coco_predictions, indent=2))
    (output_dir / "metrics.json").write_text(json.dumps(scores, indent=2))
    print(f"Evaluated {len(dataset)} images; metrics: {scores}")
    print(f"Saved visualizations and predictions to {output_dir}")


@torch.no_grad()
def run_video(args, model, device, modality_cfg):
    import cv2

    source = int(args.source) if args.source.isdigit() else args.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video/camera source: {args.source}")

    writer = None
    frames = 0
    model.eval()
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        display_img, tensor = load_image_from_pil(pil_frame, DATA.img_size, modality_cfg.in_channels)
        prediction = model.predict(tensor.to(device), score_thresh=args.score_thresh)[0]
        draw_results(display_img, prediction)
        rendered = cv2.cvtColor(np.array(display_img), cv2.COLOR_RGB2BGR)

        if args.out_video and writer is None:
            height, width = rendered.shape[:2]
            fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
            writer = cv2.VideoWriter(args.out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if writer is not None:
            writer.write(rendered)
        if args.show:
            cv2.imshow("Detector (press q to quit)", rendered)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        frames += 1

    capture.release()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()
    print(f"Processed {frames} frames" + (f"; saved {args.out_video}" if args.out_video else ""))


def load_image_from_pil(source, img_size, in_channels):
    display_img, _ = letterbox_image(source, img_size, in_channels=3)
    model_img, _ = letterbox_image(source, img_size, in_channels=in_channels)
    return display_img, image_to_tensor(model_img, in_channels).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=["rgb", "thermal"], required=True,
                         help="Primary modality to run.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", help="Run one image.")
    input_group.add_argument("--coco-json", help="Evaluate and visualize a COCO split.")
    input_group.add_argument("--source", help="Video path or camera index (for example 0).")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="result.jpg")
    parser.add_argument("--out-dir", default="inference_results")
    parser.add_argument("--out-video", default=None)
    parser.add_argument("--show", action="store_true", help="Show video/camera output in a window.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--score-thresh", type=float, default=0.3)

    # optional late-fusion pass with a second, genuinely paired image
    parser.add_argument("--fuse-modality", choices=["rgb", "thermal"], default=None)
    parser.add_argument("--fuse-image", default=None)
    parser.add_argument("--fuse-checkpoint", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modality_cfg = DATA.modality(args.modality)

    model = load_model(args.checkpoint, modality_cfg.in_channels, device)
    if args.coco_json:
        run_coco(args, model, device, modality_cfg)
        return
    if args.source:
        run_video(args, model, device, modality_cfg)
        return

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
