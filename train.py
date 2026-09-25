"""
Train one single-modality detector (RGB or thermal).

Usage:
    python train.py --modality rgb
    python train.py --modality thermal

Run both to get two independent detectors. Edit config.py first (DATA.root,
num_classes, hyperparameters).
"""

import os
import time
import argparse
import torch
from torch.utils.data import DataLoader

from config import DATA, MODEL, TRAIN
from data.coco_detection_dataset import CocoDetectionDataset, collate_fn
from models.single_modality_detector import SingleModalityDetector
from models.loss import DetectionLoss


def build_dataloaders(modality_cfg):
    train_set = CocoDetectionDataset(
        root=DATA.root, ann_path=modality_cfg.train_ann, image_dir=modality_cfg.train_dir,
        img_size=DATA.img_size, in_channels=modality_cfg.in_channels,
    )
    val_set = CocoDetectionDataset(
        root=DATA.root, ann_path=modality_cfg.val_ann, image_dir=modality_cfg.val_dir,
        img_size=DATA.img_size, in_channels=modality_cfg.in_channels,
    )
    print(f"[{modality_cfg.name}] train images: {len(train_set)}, val images: {len(val_set)}")

    train_loader = DataLoader(
        train_set, batch_size=TRAIN.batch_size, shuffle=True,
        num_workers=TRAIN.num_workers, collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=TRAIN.batch_size, shuffle=False,
        num_workers=TRAIN.num_workers, collate_fn=collate_fn,
    )
    return train_loader, val_loader


def run_epoch(model, loader, criterion, optimizer, scaler, device, train=True, epoch=0):
    model.train(train)
    total_loss, total_cls, total_box, n_batches = 0.0, 0.0, 0.0, 0
    t0 = time.time()

    for step, (imgs, targets) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with torch.autocast(device_type="cuda", enabled=(TRAIN.amp and device.type == "cuda")):
                cls_logits, box_deltas, anchors = model(imgs)
                losses = criterion(cls_logits, box_deltas, anchors, targets)

            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(losses["loss"]).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN.grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    losses["loss"].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN.grad_clip_norm)
                    optimizer.step()

        total_loss += losses["loss"].item()
        total_cls += losses["cls_loss"].item()
        total_box += losses["box_loss"].item()
        n_batches += 1

        if train and step % TRAIN.log_every == 0:
            elapsed = time.time() - t0
            print(f"epoch {epoch} step {step}/{len(loader)} "
                  f"loss={losses['loss'].item():.4f} cls={losses['cls_loss'].item():.4f} "
                  f"box={losses['box_loss'].item():.4f} ({elapsed:.1f}s)")

    return {"loss": total_loss / n_batches, "cls_loss": total_cls / n_batches, "box_loss": total_box / n_batches}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=["rgb", "thermal"], required=True)
    args = parser.parse_args()

    modality_cfg = DATA.modality(args.modality)
    device = torch.device(TRAIN.device if torch.cuda.is_available() else "cpu")
    print(f"Training modality='{modality_cfg.name}' (in_channels={modality_cfg.in_channels}) on device: {device}")

    train_loader, val_loader = build_dataloaders(modality_cfg)
    model = SingleModalityDetector(MODEL, DATA, modality_cfg.in_channels, modality_cfg.pretrained).to(device)
    criterion = DetectionLoss(DATA.num_classes, TRAIN.focal_alpha, TRAIN.focal_gamma, TRAIN.box_loss_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN.lr, weight_decay=TRAIN.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN.epochs)
    scaler = torch.cuda.amp.GradScaler() if (TRAIN.amp and device.type == "cuda") else None

    ckpt_dir = os.path.join(TRAIN.checkpoint_dir, modality_cfg.name)
    os.makedirs(ckpt_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(TRAIN.epochs):
        train_stats = run_epoch(model, train_loader, criterion, optimizer, scaler, device, train=True, epoch=epoch)
        print(f"[{modality_cfg.name}][epoch {epoch}] train: {train_stats}")
        scheduler.step()

        if epoch % TRAIN.val_every == 0:
            val_stats = run_epoch(model, val_loader, criterion, optimizer, scaler, device, train=False, epoch=epoch)
            print(f"[{modality_cfg.name}][epoch {epoch}] val:   {val_stats}")

            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_stats["loss"],
                        "modality": modality_cfg.name}, os.path.join(ckpt_dir, "last.pt"))

            if val_stats["loss"] < best_val_loss:
                best_val_loss = val_stats["loss"]
                torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_stats["loss"],
                            "modality": modality_cfg.name}, os.path.join(ckpt_dir, "best.pt"))
                print(f"[{modality_cfg.name}][epoch {epoch}] new best val_loss={best_val_loss:.4f}, checkpoint saved")


if __name__ == "__main__":
    main()
