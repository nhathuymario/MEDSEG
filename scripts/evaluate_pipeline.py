"""Evaluate the integrated ISIC detection -> ROI segmentation pipeline."""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate import _detection_counts, _find_mask, _mask_to_box, _read_split
from src.evaluation.metrics_segmentation import compute_segmentation_metrics
from src.models.attention_unet import AttentionUNet
from src.models.faster_rcnn import LesionDetector
from src.pipeline.full_pipeline import MedSegPipeline


def _confidence_interval(values, seed=42, samples=2000):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(values, size=(samples, values.size), replace=True), axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detection-checkpoint", default="outputs/checkpoints/best_detection.pth")
    parser.add_argument("--segmentation-checkpoint", default="outputs/checkpoints/best_segmentation.pth")
    parser.add_argument("--image-dir", default="data/processed/isic2018/images")
    parser.add_argument("--mask-dir", default="data/processed/isic2018/masks")
    parser.add_argument("--split-dir", default="data/splits/isic2018")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--detection-threshold", type=float, default=0.5)
    parser.add_argument("--segmentation-threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--output-csv", default="outputs/metrics/isic2018_pipeline_test_per_image.csv")
    args = parser.parse_args()
    if args.max_images is not None and args.max_images < 1:
        parser.error("--max-images must be at least 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = LesionDetector(num_classes=2, pretrained=False)
    detector.load_state_dict(torch.load(args.detection_checkpoint, map_location=device))
    segmentor = AttentionUNet(in_ch=3, out_ch=1, features=(64, 128, 256, 512))
    segmentor.load_state_dict(torch.load(args.segmentation_checkpoint, map_location=device))
    pipeline = MedSegPipeline(detector.eval().to(device), segmentor.eval().to(device), device)

    files = _read_split(args.split_dir, args.split)
    if args.max_images is not None:
        files = files[: args.max_images]
    rows = []
    image_dir, mask_dir = Path(args.image_dir), Path(args.mask_dir)

    for index, filename in enumerate(files, 1):
        image = cv2.imread(str(image_dir / filename))
        mask_path = _find_mask(mask_dir, filename)
        target = (cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        result = pipeline.run(image, args.detection_threshold, args.segmentation_threshold)
        predicted = result["full_mask"]
        metrics = compute_segmentation_metrics(predicted, target, threshold=0.5)
        detection = _detection_counts(result["detection"]["boxes"], _mask_to_box(mask_path), args.iou_threshold)
        rows.append({
            "filename": filename,
            "split": args.split,
            "detected": int(bool(result["detection"]["boxes"])),
            "num_boxes": len(result["detection"]["boxes"]),
            "detection_tp": detection["tp"], "detection_fp": detection["fp"], "detection_fn": detection["fn"],
            "best_box_iou": detection["best_iou"], "time_ms": result["time_ms"], **metrics,
        })
        print(f"[{index}/{len(files)}] {filename}: Dice={metrics['dice']:.4f}, {result['time_ms']:.1f} ms")

    metric_names = ["dice", "iou", "sensitivity", "specificity", "pixel_accuracy", "time_ms"]
    aggregate = {}
    for name in metric_names:
        values = [row[name] for row in rows]
        aggregate[name] = {
            "mean": float(np.mean(values)), "std": float(np.std(values)),
            "median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
            "ci95_mean": _confidence_interval(values),
        }
    total_tp = sum(row["tp"] for row in rows)
    total_fp = sum(row["fp"] for row in rows)
    total_fn = sum(row["fn"] for row in rows)
    total_tn = sum(row["tn"] for row in rows)
    summary = {
        "evaluation_level": "full_pipeline", "pipeline": "detection_roi_segmentation",
        "dataset": "isic2018", "split": args.split, "num_images": len(rows),
        "detection_checkpoint": args.detection_checkpoint, "segmentation_checkpoint": args.segmentation_checkpoint,
        "detection_threshold": args.detection_threshold, "segmentation_threshold": args.segmentation_threshold,
        "success_rate": float(np.mean([row["detected"] for row in rows])),
        "zero_detection_images": sum(not row["detected"] for row in rows),
        "detection_tp": sum(row["detection_tp"] for row in rows),
        "detection_fp": sum(row["detection_fp"] for row in rows),
        "detection_fn": sum(row["detection_fn"] for row in rows),
        "pixel_confusion": {"tp": total_tp, "fp": total_fp, "fn": total_fn, "tn": total_tn},
        "global_metrics": {
            "dice": float((2 * total_tp + 1) / (2 * total_tp + total_fp + total_fn + 1)),
            "iou": float((total_tp + 1) / (total_tp + total_fp + total_fn + 1)),
            "sensitivity": float(total_tp / (total_tp + total_fn + 1e-6)),
            "specificity": float(total_tn / (total_tn + total_fp + 1e-6)),
            "pixel_accuracy": float((total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + 1e-6)),
        },
        "per_image": aggregate,
    }
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    summary_path = output.with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved {output} and {summary_path}")


if __name__ == "__main__":
    main()
