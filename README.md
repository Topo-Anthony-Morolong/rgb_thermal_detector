# RGB and Thermal Single-Modality Detectors (FLIR ADAS v2)

Two independent RetinaNet-style object detectors — one trained on FLIR
ADAS v2's RGB images, one on its thermal images — sharing the same
architecture and training code, just different input channels and weights.

```
RGB image     ──► ResNet backbone (3ch) ──► FPN neck ──► cls/box head ──► detections
Thermal image ──► ResNet backbone (1ch) ──► FPN neck ──► cls/box head ──► detections
```

## Why two separate models instead of one fused model

The original plan was a two-stream fusion detector (RGB + thermal feeding
one model). That requires frame-paired RGB/thermal images. We checked this
against the actual downloaded data with `diagnose_pairing.py` and found:

- `images_thermal_train` has 10,742 annotated frames, `images_rgb_train` has
  10,319 — different counts, so they can't be 1:1 pairs.
- Filename stems don't match between the two folders at all (0% overlap).
- Each image's `extra_info` field only carries scene-level metadata
  (`hours`, `scene`, `video_id`, `weather`) — no pointer to a corresponding
  image in the other modality.
- Frame numbers sampled per video don't line up either (thermal sampled
  every 15 frames from one offset, RGB sampled independently from another).

So `images_rgb_train`/`images_thermal_train` are two separate,
independently-sampled single-modality annotated sets — not a paired
fusion dataset. (Real synced pairs do exist, in the much smaller
`video_rgb_test`/`video_thermal_test` + `rgb_to_thermal_vid_map.json`,
but that's sized as a test/eval split, not enough to train on.) Training
one detector per modality uses each annotated set as it's actually built,
instead of forcing a fusion architecture onto unpaired data.

## Dataset setup

1. Download FLIR ADAS v2 from https://www.flir.com/oem/adas/adas-dataset-form/
2. Expected layout (matches what `diagnose_pairing.py` confirmed against
   your actual download):
   ```
   FLIR_ADAS_v2/
     images_rgb_train/{data/, coco.json}
     images_rgb_val/{data/, coco.json}
     images_thermal_train/{data/, coco.json}
     images_thermal_val/{data/, coco.json}
   ```
3. Edit `config.py` → `DATA.root`. The `rgb`/`thermal` sub-configs already
   point at the standard v2 paths; only change them if your layout differs.
4. `DATA.num_classes` / `DATA.class_names` default to FLIR's 3 core classes
   (person, bicycle, car) — extend if you're using more of FLIR's 15
   annotated categories. Category sets are independent per modality in
   principle; if RGB and thermal use different category id → name mappings
   in their respective `coco.json`, this is handled automatically since
   each `CocoDetectionDataset` builds its own `cat_id_to_idx` from its own
   file — just make sure `DATA.class_names`' order is what you want both
   models to share for reporting.

## Usage

```bash
pip install -r requirements.txt

# Train each modality separately
python train.py --modality rgb
python train.py --modality thermal

# Checkpoints land in ./checkpoints/rgb/ and ./checkpoints/thermal/

# Run inference with one modality
python inference.py --modality rgb --image sample_rgb.jpg \
    --checkpoint checkpoints/rgb/best.pt --out result.jpg

# Optional: combine both models' detections at the box level, if you
# separately have a genuinely time-synced RGB+thermal pair (e.g. from
# video_rgb_test/video_thermal_test)
python inference.py --modality rgb --image sample_rgb.jpg \
    --checkpoint checkpoints/rgb/best.pt \
    --fuse-modality thermal --fuse-image sample_thermal.jpg \
    --fuse-checkpoint checkpoints/thermal/best.pt \
    --out result_fused.jpg
```

Note on that late-fusion option: it merges each model's independent
detections via NMS across both sets of boxes, which is a much weaker
form of fusion than combining features inside one network (what the
original two-stream design did). It's useful as a quick way to combine
two already-trained detectors, not a substitute for real feature-level
fusion — that would need actual paired training data.

## What's implemented vs. what you'll likely want to add

Implemented: single-modality ResNet18/34 (or TinyCNN debug) backbone, FPN
neck, focal-loss classification + smooth-L1 box regression with anchor
assignment, AMP training, per-modality checkpointing, inference with NMS,
optional post-hoc box-level fusion across two trained models.

Not implemented (reasonable next steps):
- mAP evaluation (currently only reports val loss) — plug in
  `torchmetrics.detection.MeanAveragePrecision` on top of `model.predict()`.
- Data augmentation (random flip/crop/color-jitter).
- Multi-GPU / DDP training.
- If you want real feature-level fusion later: `video_rgb_test` +
  `video_thermal_test` + `rgb_to_thermal_vid_map.json` is the one part of
  this dataset that's actually frame-paired — worth inspecting if you want
  to revisit a fused architecture on a smaller, genuinely paired set.

## File layout

```
config.py                        - paths/hyperparameters, per-modality sub-configs
data/coco_detection_dataset.py   - plain single-modality COCO dataset + collate_fn
models/backbone.py               - ResNet/TinyCNN backbone (in_channels configurable)
models/head.py                   - FPN neck + classification/regression head
models/loss.py                   - focal loss, smooth-L1, anchor target assignment
models/single_modality_detector.py - assembles backbone + FPN + head + predict()
utils/anchors.py                 - multi-scale anchor generation
utils/box_utils.py               - IoU, box encode/decode, NMS
train.py                          - training loop, --modality rgb|thermal
inference.py                      - single-modality inference + optional late fusion
diagnose_pairing.py               - checks whether an RGB/thermal split is frame-paired
```
