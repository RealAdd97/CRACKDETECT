"""
train.py

Trains a YOLOv8 object-detection model to find cracks in images, using
Ultralytics' YOLO API. Point it at the data.yaml produced by
prepare_dataset.py.

Usage:
    python train.py --data ./dataset/data.yaml --epochs 100 --model yolov8n.pt

After training, the best weights are saved to:
    <project>/<name>/weights/best.pt
which you then pass to detect_video.py via --model.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train a YOLO crack-detection model.")
    parser.add_argument("--data", required=True, type=Path, help="Path to data.yaml (from prepare_dataset.py).")
    parser.add_argument(
        "--model", default="yolov8s.pt",
        help="Starting point: a pretrained checkpoint (yolov8n/s/m/l/x.pt) for transfer learning "
             "(recommended — 'n' is fastest, 'x' is most accurate but slower), a .yaml model config "
             "to train from scratch, or a previous best.pt to keep fine-tuning.",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=896)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="e.g. '0' for first GPU, 'cpu', or leave unset for auto-detect.")
    parser.add_argument("--patience", type=int, default=50, help="Early-stopping patience (epochs with no improvement).")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final learning rate factor")
    parser.add_argument("--mosaic", type=float, default=1.0, help="Mosaic augmentation probability")
    parser.add_argument("--mixup", type=float, default=0.15, help="MixUp augmentation probability")
    parser.add_argument("--hsv_h", type=float, default=0.015, help="HSV-Hue augmentation")
    parser.add_argument("--hsv_s", type=float, default=0.7, help="HSV-Saturation augmentation")
    parser.add_argument("--hsv_v", type=float, default=0.4, help="HSV-Value augmentation")
    # Training optimization parameters
    parser.add_argument("--warmup_epochs", type=float, default=3.0, help="Number of warmup epochs (fraction allowed)")
    parser.add_argument("--warmup_momentum", type=float, default=0.8, help="Warmup initial momentum")
    parser.add_argument("--warmup_bias_lr", type=float, default=0.1, help="Warmup initial bias learning rate")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["linear", "cosine", "cosine2"], help="Learning rate scheduler")
    parser.add_argument("--freeze", type=str, default=None, help="Freeze layers: backbone=21, 10=6, 6=3, 3=2, 2=1, 1=0")
    parser.add_argument("--close_mosaic", type=int, default=10, help="Final N epochs to stop mosaic augmentation")
    parser.add_argument("--box", type=float, default=7.5, help="Box loss gain")
    parser.add_argument("--cls", type=float, default=0.5, help="Class loss gain")
    parser.add_argument("--dfl", type=float, default=1.5, help="DFL loss gain")
    parser.add_argument("--project", default="runs/detect", help="Where to save training runs.")
    parser.add_argument("--name", default="crack_yolo", help="Run name (subfolder under --project).")
    parser.add_argument("--resume", action="store_true", help="Resume the last interrupted training run.")
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"data.yaml not found at {args.data}")

    logger.info(f"Loading base model: {args.model}")
    model = YOLO(args.model)

    logger.info("Starting training...")
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        lr0=args.lr0,
        lrf=args.lrf,
        mosaic=args.mosaic,
        mixup=args.mixup,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        warmup_epochs=args.warmup_epochs,
        warmup_momentum=args.warmup_momentum,
        warmup_bias_lr=args.warmup_bias_lr,
        # lr_scheduler=args.lr_scheduler,
        freeze=args.freeze,
        close_mosaic=args.close_mosaic,
        box=args.box,
        cls=args.cls,
        dfl=args.dfl,
        project=args.project,
        name=args.name,
        resume=args.resume,
    )

    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    logger.info("Training complete.")
    logger.info(f"Best weights saved to: {best_weights}")

    logger.info("Running validation on the best weights...")
    metrics = model.val()
    logger.info(f"mAP50: {metrics.box.map50:.3f}  mAP50-95: {metrics.box.map:.3f}")

    logger.info("Next step — run detection on a video:")
    logger.info(f"  python detect_video.py --model {best_weights} --source your_video.mp4")


if __name__ == "__main__":
    main()
