"""Main evaluation script."""
import argparse
import csv
import sys
import yaml
import torch
import numpy as np
import time
from pathlib import Path
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_segmentation import SegmentationDataset
from src.data.transforms import get_val_transforms
from src.models.faster_rcnn import LesionDetector
from src.models.attention_unet import AttentionUNet
from src.models.unet import UNet
from src.evaluation.metrics_detection import compute_map
from src.evaluation.metrics_detection import compute_iou
from src.evaluation.metrics_segmentation import compute_segmentation_metrics


def _read_split(split_dir, split_name):
    split_path = Path(split_dir) / f"{split_name}.csv"
    with open(split_path, newline="") as handle:
        return [row["filename"] for row in csv.DictReader(handle)]


def _mask_to_box(mask_path: Path):
    import cv2

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    ys, xs = np.where(mask > 127)
    if len(xs) == 0 or len(ys) == 0:
        return []
    return [[float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]]


def _find_mask(mask_dir: Path, image_name: str):
    stem = Path(image_name).stem
    for suffix in [f"{stem}_segmentation.png", f"{stem}.png", f"{stem}_mask.png"]:
        path = mask_dir / suffix
        if path.exists():
            return path
    raise FileNotFoundError(f"No mask found for image {image_name}")


def _detection_counts(pred_boxes, gt_boxes, iou_threshold=0.5):
    matched = [False] * len(gt_boxes)
    tp = 0
    fp = 0
    best_ious = []
    for pred_box in pred_boxes:
        best_iou = 0.0
        best_index = -1
        for index, gt_box in enumerate(gt_boxes):
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_index = index
        best_ious.append(best_iou)
        if best_iou >= iou_threshold and best_index >= 0 and not matched[best_index]:
            tp += 1
            matched[best_index] = True
        else:
            fp += 1
    fn = matched.count(False)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "best_iou": max(best_ious) if best_ious else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["detection", "segmentation"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", help="Training config used for the model")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-csv", help="Save per-image metrics to this CSV path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating {args.model} on {device}...")

    if args.model == "detection":
        import cv2
        from torchvision import transforms as T

        if args.config:
            with open(args.config, "r") as handle:
                config = yaml.safe_load(handle)
        else:
            config = {
                "data": {
                    "image_dir": "data/processed/isic2018/images",
                    "mask_dir": "data/processed/isic2018/masks",
                    "split_dir": "data/splits/isic2018",
                },
                "evaluation": {"confidence_threshold": 0.5, "iou_threshold": 0.5},
            }

        data_config = config.get("data", {})
        eval_config = config.get("evaluation", {})
        image_dir = Path(data_config.get("image_dir", "data/processed/isic2018/images"))
        mask_dir = Path(data_config.get("mask_dir", "data/processed/isic2018/masks"))
        split_dir = Path(data_config.get("split_dir", "data/splits/isic2018"))
        confidence_threshold = eval_config.get("confidence_threshold", 0.5)
        iou_threshold = eval_config.get("iou_threshold", 0.5)

        model = LesionDetector(num_classes=2, pretrained=False)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model.eval().to(device)

        files = _read_split(split_dir, args.split)
        predictions = []
        ground_truths = []
        per_image_rows = []
        to_tensor = T.ToTensor()

        with torch.no_grad():
            for filename in files:
                image = cv2.imread(str(image_dir / filename))
                if image is None:
                    raise FileNotFoundError(f"Image not found: {image_dir / filename}")
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                tensor = to_tensor(image_rgb).to(device)

                t0 = time.time()
                pred = model([tensor])[0]
                inference_time_ms = (time.time() - t0) * 1000

                scores = pred["scores"].detach().cpu().numpy()
                keep = scores >= confidence_threshold
                pred_boxes = pred["boxes"].detach().cpu().numpy()[keep].tolist()
                pred_scores = scores[keep].tolist()
                gt_boxes = _mask_to_box(_find_mask(mask_dir, filename))

                predictions.append({"boxes": pred_boxes, "scores": pred_scores})
                ground_truths.append({"boxes": gt_boxes})

                counts = _detection_counts(pred_boxes, gt_boxes, iou_threshold)
                per_image_rows.append(
                    {
                        "filename": filename,
                        "split": args.split,
                        "inference_time_ms": inference_time_ms,
                        "gt_boxes": len(gt_boxes),
                        "pred_boxes": len(pred_boxes),
                        **counts,
                    }
                )

        map_50 = compute_map(predictions, ground_truths, iou_threshold=iou_threshold)
        tp = sum(row["tp"] for row in per_image_rows)
        fp = sum(row["fp"] for row in per_image_rows)
        fn = sum(row["fn"] for row in per_image_rows)
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        print(f"Evaluated {len(files)} images from the {args.split} split:")
        print(f"  mAP@{iou_threshold}: {map_50:.4f}")
        print(f"  precision@{confidence_threshold}: {precision:.4f}")
        print(f"  recall@{confidence_threshold}: {recall:.4f}")

        if args.output_csv:
            output_path = Path(args.output_csv)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fieldnames = [
                "filename",
                "split",
                "inference_time_ms",
                "gt_boxes",
                "pred_boxes",
                "tp",
                "fp",
                "fn",
                "precision",
                "recall",
                "best_iou",
            ]
            with open(output_path, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(per_image_rows)
            print(f"Saved per-image metrics to {output_path}")
    else:
        if not args.config:
            parser.error("--config is required for segmentation evaluation")
        with open(args.config, "r") as handle:
            config = yaml.safe_load(handle)

        model_config = config["model"]
        model_args = {
            "in_ch": model_config.get("in_channels", 3),
            "out_ch": model_config.get("out_channels", 1),
            "features": tuple(model_config.get("features", [64, 128, 256, 512])),
        }
        model = (
            UNet(**model_args)
            if model_config["name"] == "unet"
            else AttentionUNet(**model_args)
        )
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model.eval().to(device)

        data_config = config["data"]
        image_size = data_config.get("image_size", [256, 256])[0]
        files = _read_split(data_config["split_dir"], args.split)
        dataset = SegmentationDataset(
            data_config["image_dir"],
            data_config["mask_dir"],
            file_list=files,
            transforms=get_val_transforms(image_size),
        )
        loader = DataLoader(
            dataset,
            batch_size=data_config.get("batch_size", 4),
            num_workers=data_config.get("num_workers", 0),
        )

        predictions, targets = [], []
        per_image_rows = []
        sample_index = 0
        with torch.no_grad():
            for images, masks in loader:
                t0 = time.time()
                logits = model(images.to(device))
                batch_time_ms = (time.time() - t0) * 1000
                batch_predictions = torch.sigmoid(logits).cpu().numpy()
                batch_targets = masks.numpy()
                predictions.append(batch_predictions)
                targets.append(batch_targets)

                for i in range(batch_predictions.shape[0]):
                    filename = files[sample_index]
                    image_metrics = compute_segmentation_metrics(
                        batch_predictions[i : i + 1],
                        batch_targets[i : i + 1],
                        threshold=config.get("evaluation", {}).get("threshold", 0.5),
                    )
                    per_image_rows.append(
                        {
                            "filename": filename,
                            "split": args.split,
                            "inference_time_ms": batch_time_ms / batch_predictions.shape[0],
                            **image_metrics,
                        }
                    )
                    sample_index += 1

        metrics = compute_segmentation_metrics(
            np.concatenate(predictions),
            np.concatenate(targets),
            threshold=config.get("evaluation", {}).get("threshold", 0.5),
        )
        print(f"Evaluated {len(dataset)} images from the {args.split} split:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")

        if args.output_csv:
            output_path = Path(args.output_csv)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fieldnames = [
                "filename",
                "split",
                "inference_time_ms",
                "dice",
                "iou",
                "sensitivity",
                "specificity",
                "pixel_accuracy",
            ]
            with open(output_path, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(per_image_rows)
            print(f"Saved per-image metrics to {output_path}")


if __name__ == "__main__":
    main()
