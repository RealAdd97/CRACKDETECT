"""
prepare_dataset.py

Organizes raw crack images (and optional YOLO-format label files) into the
directory structure Ultralytics YOLO expects for training, and generates the
matching data.yaml config file.

Expected YOLO label format (one .txt file per image, same base filename):
    <class_id> <x_center> <y_center> <width> <height>
    (all values normalized 0-1, relative to image width/height)

If you don't have labels yet, run this script with --labels_dir omitted to
just organize the images. You'll then need to annotate them with a free tool
such as:
    - LabelImg : https://github.com/HumanSignal/labelImg
    - CVAT     : https://www.cvat.ai/
    - Roboflow : https://roboflow.com  (can export straight to YOLO format)

Drop the resulting .txt files into dataset/labels/train and
dataset/labels/val, matching the image filenames, then run train.py.

Usage:
    python prepare_dataset.py \
        --images_dir /path/to/raw_images \
        --labels_dir /path/to/raw_labels \
        --output_dir ./dataset \
        --val_split 0.2 \
        --class_names crack
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_images(images_dir: Path) -> list[Path]:
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise FileNotFoundError(f"No images found in {images_dir}")
    return images


def split_train_val(images: list[Path], val_split: float, seed: int = 42) -> tuple[list[Path], list[Path]]:
    shuffled = images.copy()
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_split)) if len(shuffled) > 1 else 0
    return shuffled[val_count:], shuffled[:val_count]


def copy_split(
    images: list[Path],
    labels_dir: Path | None,
    out_images: Path,
    out_labels: Path,
    missing_label_action: str,
) -> int:
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    missing = 0
    for img_path in images:
        shutil.copy2(img_path, out_images / img_path.name)
        label_path = (labels_dir / f"{img_path.stem}.txt") if labels_dir else None
        if label_path and label_path.exists():
            shutil.copy2(label_path, out_labels / label_path.name)
        else:
            missing += 1
            if missing_label_action == "empty":
                (out_labels / f"{img_path.stem}.txt").touch()
            # "skip_copy" -> leave no label file at all for this image
    return missing


def write_data_yaml(output_dir: Path, class_names: list[str]) -> Path:
    data = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(class_names)},
    }
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Prepare a YOLO-format crack detection dataset.")
    parser.add_argument("--images_dir", required=True, type=Path, help="Folder containing raw crack images.")
    parser.add_argument(
        "--labels_dir", type=Path, default=None,
        help="Folder containing YOLO-format .txt labels (same basename as images). Omit if not yet annotated.",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("./dataset"), help="Where to write the organized dataset.")
    parser.add_argument("--val_split", type=float, default=0.2, help="Fraction of images reserved for validation.")
    parser.add_argument("--class_names", nargs="+", default=["crack"], help="Class names, in label-index order.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--missing_label_action", choices=["empty", "skip_copy"], default="empty",
        help="What to do for images with no matching label file. 'empty' writes an empty .txt "
             "(image treated as a negative/background example). 'skip_copy' leaves no label file.",
    )
    args = parser.parse_args()

    images = find_images(args.images_dir)
    logger.info(f"Found {len(images)} images in {args.images_dir}")

    train_imgs, val_imgs = split_train_val(images, args.val_split, args.seed)
    logger.info(f"Split: {len(train_imgs)} train / {len(val_imgs)} val")

    if args.labels_dir is None:
        logger.warning(
            "No --labels_dir provided. Images will be copied without labels. "
            "Annotate them (e.g. with LabelImg, CVAT, or Roboflow), then place the resulting "
            ".txt files in dataset/labels/train and dataset/labels/val with filenames matching "
            "the images, before running train.py."
        )

    missing_train = copy_split(
        train_imgs, args.labels_dir,
        args.output_dir / "images" / "train", args.output_dir / "labels" / "train",
        args.missing_label_action,
    )
    missing_val = copy_split(
        val_imgs, args.labels_dir,
        args.output_dir / "images" / "val", args.output_dir / "labels" / "val",
        args.missing_label_action,
    )

    if args.labels_dir is not None and (missing_train or missing_val):
        logger.warning(f"{missing_train + missing_val} image(s) had no matching label file.")

    yaml_path = write_data_yaml(args.output_dir, args.class_names)
    logger.info(f"Wrote dataset config to {yaml_path}")
    logger.info("Dataset ready. Train with:")
    logger.info(f"  python train.py --data {yaml_path}")


if __name__ == "__main__":
    main()
