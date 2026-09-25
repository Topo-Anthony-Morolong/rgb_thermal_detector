"""
Run this FIRST, before train.py, on your actual downloaded dataset.

FLIR ADAS v2's images_thermal_train / images_rgb_train folders are NOT
guaranteed to be frame-paired by filename (they're separate annotated sets
with different image counts). This script inspects your actual coco.json
to find how pairing info is actually stored, and checks how many thermal
images have a matching RGB file on disk under a few candidate strategies.

Usage:
    python diagnose_pairing.py --root /path/to/FLIR_ADAS_v2 \
        --thermal-ann images_thermal_train/coco.json \
        --thermal-dir images_thermal_train/data \
        --rgb-dir images_rgb_train/data
"""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--thermal-ann", default="images_thermal_train/coco.json")
    parser.add_argument("--thermal-dir", default="images_thermal_train/data")
    parser.add_argument("--rgb-dir", default="images_rgb_train/data")
    parser.add_argument("--sample", type=int, default=5, help="How many image entries to print in full")
    args = parser.parse_args()

    ann_path = os.path.join(args.root, args.thermal_ann)
    with open(ann_path) as f:
        coco = json.load(f)

    images = coco["images"]
    print(f"Loaded {ann_path}")
    print(f"Total thermal image entries: {len(images)}")
    print(f"Top-level keys in coco.json: {list(coco.keys())}")
    print()

    print(f"--- First {args.sample} image entries (full contents) ---")
    for im in images[: args.sample]:
        print(json.dumps(im, indent=2))
        print()

    # Check whether 'extra_info' (or similarly named field) exists and what it holds
    keys_seen = set()
    for im in images:
        keys_seen.update(im.keys())
    print(f"All keys seen across image entries: {sorted(keys_seen)}")
    print()

    extra_info_key = next((k for k in keys_seen if "extra" in k.lower() or "meta" in k.lower()), None)
    if extra_info_key:
        print(f"Found likely metadata field: '{extra_info_key}'. Sample values:")
        for im in images[: args.sample]:
            print(f"  {im.get('file_name')}: {im.get(extra_info_key)}")
    else:
        print("No field containing 'extra' or 'meta' found in image entries.")
    print()

    # Strategy check: how many thermal images have a same-stem RGB file on disk?
    rgb_dir_abs = os.path.join(args.root, args.rgb_dir)
    if not os.path.isdir(rgb_dir_abs):
        print(f"WARNING: RGB dir not found at {rgb_dir_abs}")
        return

    rgb_files = set(os.listdir(rgb_dir_abs))
    rgb_stems = {os.path.splitext(f)[0] for f in rgb_files}

    matched = 0
    for im in images:
        stem = os.path.splitext(os.path.basename(im["file_name"]))[0]
        if stem in rgb_stems:
            matched += 1

    print(f"RGB files on disk: {len(rgb_files)}")
    print(f"Thermal images with a same-stem match in RGB dir: {matched} / {len(images)} "
          f"({100 * matched / max(len(images),1):.1f}%)")
    print()
    if matched / max(len(images), 1) < 0.9:
        print("=> Same-stem matching is NOT reliable for this split.")
        print("   This confirms images_thermal_train/images_rgb_train are not frame-paired.")
        print("   Paired data lives in video_rgb_test / video_thermal_test, matched via")
        print("   rgb_to_thermal_vid_map.json in the dataset root — inspect that file next,")
        print("   or check the metadata field printed above (if any) for a direct pointer")
        print("   from each thermal entry to its RGB counterpart.")
    else:
        print("=> Same-stem matching looks reliable here — real frame pairing may exist")
        print("   for this split. Note: this project now trains two independent")
        print("   single-modality detectors (data/coco_detection_dataset.py, no pairing")
        print("   needed) since FLIR ADAS v2's main train/val split isn't paired. If you")
        print("   want to revisit feature-level fusion on this paired data, see the")
        print("   'What's implemented vs. what you'll likely want to add' section of")
        print("   README.md.")


if __name__ == "__main__":
    main()
