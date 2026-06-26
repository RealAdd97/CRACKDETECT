"""
detect_video.py

Runs a trained YOLO crack-detection model over MP4 video(s), producing:
  - an annotated output video with bounding boxes drawn around detected cracks
  - a JSON + CSV report listing every detection (frame, timestamp, bbox, confidence, severity)
  - a console summary per video

Usage:
    # single video
    python detect_video.py --model runs/detect/crack_yolo/weights/best.pt --source inspection.mp4

    # folder of videos
    python detect_video.py --model best.pt --source ./videos --output ./results

    # report only, skip writing the annotated video (faster)
    python detect_video.py --model best.pt --source inspection.mp4 --no_save_video
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}

BOX_COLOR = (0, 0, 255)  # red, BGR
TEXT_COLOR = (255, 255, 255)


@dataclass
class Detection:
    video: str
    frame: int
    timestamp_sec: float
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    bbox_width: int
    bbox_height: int
    severity: str


def classify_severity(bbox_w: int, bbox_h: int, frame_w: int, frame_h: int, confidence: float) -> str:
    """
    Enhanced severity classification based on bounding box size and confidence.
    This is NOT a real-world crack-width measurement — that requires a calibrated
    camera distance/scale. Treat it as a relative visual-prominence indicator
    for triaging which clips a human inspector should look at first.

    Uses a combined score: 60% size-based + 40% confidence-based
    """
    # Calculate size-based score (normalized diagonal fraction)
    diag_fraction = ((bbox_w**2 + bbox_h**2) ** 0.5) / ((frame_w**2 + frame_h**2) ** 0.5)
    size_score = min(diag_fraction * 20, 1.0)  # Scale to 0-1 range

    # Confidence score is already 0-1
    confidence_score = confidence

    # Combined score with weights
    final_score = 0.6 * size_score + 0.4 * confidence_score

    if final_score < 0.3:
        return "minor"
    elif final_score < 0.6:
        return "moderate"
    else:
        return "severe"


def _boxes_distance(box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]) -> float:
    """Calculate Euclidean distance between centers of two bounding boxes."""
    x1_center = (box1[0] + box1[2]) / 2
    y1_center = (box1[1] + box1[3]) / 2
    x2_center = (box2[0] + box2[2]) / 2
    y2_center = (box2[1] + box2[3]) / 2
    return ((x1_center - x2_center) ** 2 + (y1_center - y2_center) ** 2) ** 0.5


def process_video(
    model: YOLO,
    video_path: Path,
    output_dir: Path,
    conf: float,
    iou: float,
    device: str | None,
    save_video: bool,
    class_names: dict[int, str],
    temporal_window: int = 3,
    temporal_min_hits: int = 2,
    distance_threshold: float = 50.0,
) -> list[Detection]:
    """
    Process video for crack detection with temporal filtering to reduce false positives.

    Args:
        model: YOLO model for object detection
        video_path: Path to input video file
        output_dir: Directory for output files
        conf: Confidence threshold for detections
        iou: IoU threshold for non-maximum suppression
        device: Device to run inference on (e.g., '0' for GPU, 'cpu')
        save_video: Whether to save annotated output video
        class_names: Mapping of class IDs to class names
        temporal_window: Number of previous frames to consider for temporal filtering
        temporal_min_hits: Minimum number of detections required in temporal window
        distance_threshold: Maximum distance (pixels) between box centers to consider as same detection

    Returns:
        List of detections that passed temporal filtering
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    writer = None
    out_video_path = output_dir / f"{video_path.stem}_annotated.mp4"
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (frame_w, frame_h))

    detections: list[Detection] = []
    # For temporal filtering: keep track of detections from recent frames
    recent_detections: deque = deque(maxlen=temporal_window)
    results_stream = model.predict(
        source=str(video_path), conf=conf, iou=iou, device=device, stream=True, verbose=False,
    )

    for frame_idx, result in enumerate(results_stream):
        frame = result.orig_img

        # Collect detections for current frame
        current_frame_detections: list[Detection] = []
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = class_names.get(cls_id, str(cls_id))
            bbox_w, bbox_h = x2 - x1, y2 - y1
            severity = classify_severity(bbox_w, bbox_h, frame_w, frame_h, confidence)

            detection = Detection(
                video=video_path.name,
                frame=frame_idx,
                timestamp_sec=round(frame_idx / fps, 2),
                class_name=cls_name,
                confidence=round(confidence, 4),
                x1=x1, y1=y1, x2=x2, y2=y2,
                bbox_width=bbox_w, bbox_height=bbox_h,
                severity=severity,
            )
            current_frame_detections.append(detection)

        # Apply temporal filtering: keep detections that appear in enough recent frames
        filtered_detections: list[Detection] = []
        for det in current_frame_detections:
            # Count how many recent frames had a similar detection
            consecutive_count = 0
            # Check recent frames in reverse order (most recent first)
            for past_frame_dets in reversed(recent_detections):
                # Check if this detection matches any detection in the past frame
                for past_det in past_frame_dets:
                    # Calculate center point distance
                    curr_center_x = (det.x1 + det.x2) / 2
                    curr_center_y = (det.y1 + det.y2) / 2
                    past_center_x = (past_det.x1 + past_det.x2) / 2
                    past_center_y = (past_det.y1 + past_det.y2) / 2
                    distance = ((curr_center_x - past_center_x) ** 2 + (curr_center_y - past_center_y) ** 2) ** 0.5

                    # Also check if same class
                    if distance < distance_threshold and det.class_name == past_det.class_name:
                        consecutive_count += 1
                        break  # Found a match in this frame, check next frame
                else:
                    # No break occurred, meaning no match found in this frame
                    # For temporal consistency, we need consecutive matches
                    # If we break the chain, reset counter (optional: could be more flexible)
                    break

            # Keep detection if it appears in enough consecutive frames
            # We need at least temporal_min_hits including current frame
            if consecutive_count >= (temporal_min_hits - 1):  # -1 because we don't count current frame yet
                filtered_detections.append(det)

        # Update recent detections history
        recent_detections.append(current_frame_detections)

        # Use filtered detections for output and drawing
        detections.extend(filtered_detections)

        if save_video:
            for det in filtered_detections:
                x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
                cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
                label = f"{det.class_name} {det.confidence:.2f} ({det.severity})"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), BOX_COLOR, -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)

        if save_video and writer is not None:
            writer.write(frame)

    if writer is not None:
        writer.release()
        logger.info(f"Annotated video saved to {out_video_path}")

    return detections


def write_reports(detections: list[Detection], output_dir: Path, source_name: str) -> None:
    json_path = output_dir / f"{source_name}_report.json"
    csv_path = output_dir / f"{source_name}_report.csv"

    with open(json_path, "w") as f:
        json.dump([asdict(d) for d in detections], f, indent=2)
    logger.info(f"Detection report: {json_path}")

    if detections:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(detections[0]).keys()))
            writer.writeheader()
            for d in detections:
                writer.writerow(asdict(d))
        logger.info(f"Detection report (CSV): {csv_path}")

    severity_counts = {"minor": 0, "moderate": 0, "severe": 0}
    for d in detections:
        severity_counts[d.severity] += 1
    logger.info(
        f"{source_name}: {len(detections)} crack detection(s) — "
        f"minor={severity_counts['minor']}, moderate={severity_counts['moderate']}, severe={severity_counts['severe']}"
    )


def main():
    parser = argparse.ArgumentParser(description="Detect cracks in MP4 video(s) using a trained YOLO model.")
    parser.add_argument("--model", required=True, type=Path, help="Path to trained weights (e.g. best.pt).")
    parser.add_argument("--source", required=True, type=Path, help="A video file or a folder of videos.")
    parser.add_argument("--output", type=Path, default=Path("./results"), help="Output directory.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for non-max suppression.")
    parser.add_argument("--device", default=None, help="e.g. '0' for first GPU, 'cpu', or leave unset for auto-detect.")
    parser.add_argument(
        "--no_save_video", action="store_true",
        help="Skip writing the annotated output video (report only — faster).",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    if not args.model.exists():
        raise FileNotFoundError(f"Model weights not found: {args.model}")

    logger.info(f"Loading model from {args.model}")
    model = YOLO(str(args.model))
    class_names = model.names  # dict[int, str]

    if args.source.is_dir():
        videos = sorted(p for p in args.source.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)
        if not videos:
            raise FileNotFoundError(f"No video files found in {args.source}")
    else:
        videos = [args.source]

    for video_path in videos:
        logger.info(f"Processing {video_path}...")
        detections = process_video(
            model, video_path, args.output, args.conf, args.iou, args.device,
            save_video=not args.no_save_video, class_names=class_names,
            temporal_window=3, temporal_min_hits=2, distance_threshold=50.0,
        )
        write_reports(detections, args.output, video_path.stem)

    logger.info("Done.")


if __name__ == "__main__":
    main()
