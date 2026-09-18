"""
Training entry point.

Usage:
    python train.py

Edit config.py first (dataset paths, num_classes, hyperparameters).
"""

import os
import time
import torch
from torch.utils.data import DataLoader

from config import DATA, MODEL, TRAIN
from data.flir_dataset import FlirAdasDataset, collate_fn
from models.detector import RGBThermalDetector
from models.loss import DetectionLoss


def build_dataloaders():
    train_set = FlirAdasDataset(
        root=DATA.root, thermal_ann_path=DATA.train_thermal_ann,
        thermal_dir=DATA.train_thermal_dir, rgb_dir=DATA.train_rgb_dir,
        img_size=DATA.img_size, align_mode=DATA.align_mode,
        homography_path=DATA.homography_path, train=True,
    )
    val_set = FlirAdasDataset(
        root=DATA.root, thermal_ann_path=DATA.val_thermal_ann,
        thermal_dir=DATA.val_thermal_dir, rgb_dir=DATA.val_rgb_dir,
        img_size=DATA.img_size, align_mode=DATA.align_mode,
        homography_path=DATA.homography_path, train=False,
    )

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

    for step, (rgb, thermal, targets) in enumerate(loader):
        rgb, thermal = rgb.to(device, non_blocking=True), thermal.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with torch.autocast(device_type="cuda", enabled=(TRAIN.amp and device.type == "cuda")):
                cls_logits, box_deltas, anchors = model(rgb, thermal)
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
    device = torch.device(TRAIN.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = build_dataloaders()
    model = RGBThermalDetector(MODEL, DATA).to(device)
    criterion = DetectionLoss(DATA.num_classes, TRAIN.focal_alpha, TRAIN.focal_gamma, TRAIN.box_loss_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN.lr, weight_decay=TRAIN.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN.epochs)
    scaler = torch.cuda.amp.GradScaler() if (TRAIN.amp and device.type == "cuda") else None

    os.makedirs(TRAIN.checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(TRAIN.epochs):
        train_stats = run_epoch(model, train_loader, criterion, optimizer, scaler, device, train=True, epoch=epoch)
        print(f"[epoch {epoch}] train: {train_stats}")
        scheduler.step()

        if epoch % TRAIN.val_every == 0:
            val_stats = run_epoch(model, val_loader, criterion, optimizer, scaler, device, train=False, epoch=epoch)
            print(f"[epoch {epoch}] val:   {val_stats}")

            ckpt_path = os.path.join(TRAIN.checkpoint_dir, "last.pt")
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_stats["loss"]}, ckpt_path)

            if val_stats["loss"] < best_val_loss:
                best_val_loss = val_stats["loss"]
                torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_stats["loss"]},
                           os.path.join(TRAIN.checkpoint_dir, "best.pt"))
                print(f"[epoch {epoch}] new best val_loss={best_val_loss:.4f}, checkpoint saved")


if __name__ == "__main__":
    main()
