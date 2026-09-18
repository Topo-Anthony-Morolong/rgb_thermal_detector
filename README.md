# RGB + Thermal Two-Stream Object Detector

A two-stream CNN detector for fused RGB/thermal object detection, built for
FLIR ADAS-style data. RetinaNet-style single-stage architecture: two
backbones, a fusion module, an FPN neck, and a shared detection head.

```
RGB image  ──► ResNet stream (3ch)  ──┐
                                        ├──► fusion (per FPN stage) ──► FPN neck ──► cls/box head ──► detections
Thermal img──► ResNet stream (1ch)  ──┘
```

## Why two streams instead of stacking channels

You could concatenate RGB+thermal into a 4-channel input and run one
backbone. Two independent streams are usually better here because:

- The two modalities have very different statistics (thermal is
  single-channel, low dynamic range, no color/texture cues) — a shared
  early conv stack has to compromise on filters that suit both.
- It lets you initialize the RGB stream from ImageNet while training the
  thermal stream from scratch, without averaging away the pretrained RGB
  filters.
- The fusion module can weight each modality per-location (see
  `models/fusion.py`, `AttentionFusion`) — useful because thermal is more
  reliable at night/in glare and RGB is more reliable for fine texture in
  daylight. A single shared backbone can't express that trade-off as cleanly.

## Fusion strategies

Set `MODEL.fusion_type` in `config.py`:
- `"concat"` — channel concat + 1x1 conv. Strong, simple baseline.
- `"add"` — projected sum. Cheapest.
- `"attention"` — learned per-pixel gate between RGB and thermal features.
  Recommended starting point; typically outperforms the other two on FLIR
  ADAS in published two-stream fusion work, since it adapts to
  day/night conditions instead of blending fixed proportions.

Fusion happens independently at each of `MODEL.fusion_stages` (default:
layer2/3/4), i.e. multi-scale fusion feeding into the FPN, not just a
single late-fusion point.

## Dataset setup (FLIR ADAS v2)

1. Download FLIR ADAS v2 from https://www.flir.com/oem/adas/adas-dataset-form/
2. Expected layout:
   ```
   FLIR_ADAS_v2/
     images_thermal_train/{data/, coco.json}
     images_rgb_train/{data/}
     images_thermal_val/{data/, coco.json}
     images_rgb_val/{data/}
   ```
3. Edit `config.py` → `DATA.root` to point at this folder.
4. **Alignment caveat**: FLIR's RGB and thermal sensors have different FOV
   and aren't pixel-aligned. `DATA.align_mode = "resize"` (default) just
   resizes both frames independently and reuses thermal-frame boxes as an
   approximation — a fine first baseline. For precise alignment, obtain
   FLIR's per-scene calibration, convert it to a 3x3 homography, save it as
   a `.npy` file, and set `align_mode = "homography"` +
   `homography_path`.
5. `DATA.num_classes` / `DATA.class_names` default to FLIR's 3 core classes
   (person, bicycle, car) — extend if you're using more of FLIR's 15
   annotated categories.

## Usage

```bash
pip install -r requirements.txt

# 1. Edit config.py: DATA.root, num_classes, and any hyperparameters.

# 2. Train
python train.py

# 3. Run on a single pair
python inference.py --rgb sample_rgb.jpg --thermal sample_thermal.jpg \
    --checkpoint checkpoints/best.pt --out result.jpg
```

## What's implemented vs. what you'll likely want to add

Implemented: two-stream backbones (ResNet18/34 or a TinyCNN fallback for
CPU debugging), 3 fusion strategies, FPN neck, focal-loss classification +
smooth-L1 box regression with proper anchor assignment, AMP training,
checkpointing, single-pair inference with NMS.

Not implemented (reasonable next steps once the baseline trains):
- mAP evaluation (currently only reports val loss) — plug in
  `torchmetrics.detection.MeanAveragePrecision` on top of `model.predict()`.
- Data augmentation (random flip/crop/color-jitter). Thermal-safe
  augmentation needs care — avoid color jitter on the thermal stream since
  it's not a photometric RGB signal.
- Multi-GPU / DDP training.
- The `align_mode="homography"` path needs an actual calibration matrix
  per FLIR camera pair, which FLIR provides separately from the dataset
  download.

## File layout

```
config.py            - all paths and hyperparameters
data/flir_dataset.py  - paired RGB/thermal dataset + collate_fn
models/backbone.py    - per-modality ResNet/TinyCNN backbones
models/fusion.py      - concat / add / attention fusion modules
models/head.py         - FPN neck + classification/regression head
models/loss.py         - focal loss, smooth-L1, anchor target assignment
models/detector.py     - assembles the full model + predict()
utils/anchors.py       - multi-scale anchor generation
utils/box_utils.py     - IoU, box encode/decode, NMS
train.py               - training loop
inference.py            - single-pair inference + visualization
```
