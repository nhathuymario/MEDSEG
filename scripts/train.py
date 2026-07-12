"""Main training script."""
import argparse
import csv
import sys
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_detection import DetectionDataset
from src.data.dataset_segmentation import SegmentationDataset
from src.data.transforms import (
    get_chest_xray_train_transforms,
    get_detection_transforms,
    get_train_transforms,
    get_val_transforms,
)
from src.models.faster_rcnn import LesionDetector
from src.models.unet import UNet
from src.models.attention_unet import AttentionUNet
from src.training.train_detection import train_detection
from src.training.train_segmentation import train_segmentation


def _read_split(split_dir, split_name):
    split_path = Path(split_dir) / f"{split_name}.csv"
    if not split_path.exists():
        raise FileNotFoundError(
            f"Missing split file: {split_path}. Create dataset splits first."
        )
    with open(split_path, newline="") as handle:
        return [row["filename"] for row in csv.DictReader(handle)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config yaml")
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override the number of training epochs",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs

    model_name = config["model"]["name"]
    data_config = config.get("data", {})
    dataset_name = data_config.get("dataset", "isic2018")
    image_size = data_config.get("image_size", [256, 256])[0]
    print(f"Training {model_name} on {dataset_name}...")

    # Data
    if model_name == "faster_rcnn":
        if dataset_name != "isic2018":
            raise ValueError("Faster R-CNN training currently supports ISIC 2018 only.")
        split_dir = data_config.get("split_dir", f"data/splits/{dataset_name}")
        train_files = _read_split(split_dir, "train")
        val_files = _read_split(split_dir, "val")
        train_ds = DetectionDataset(
            image_dir=data_config.get(
                "image_dir", "data/processed/isic2018/images"
            ),
            annotation_file=data_config.get(
                "annotation_file",
                "data/processed/isic2018/annotations/train.json",
            ),
            file_list=train_files,
            transforms=get_detection_transforms(size=image_size, train=True),
        )
        val_ds = DetectionDataset(
            image_dir=data_config.get(
                "image_dir", "data/processed/isic2018/images"
            ),
            annotation_file=data_config.get(
                "annotation_file",
                "data/processed/isic2018/annotations/train.json",
            ),
            file_list=val_files,
            transforms=get_detection_transforms(size=image_size, train=False),
        )
        model = LesionDetector(num_classes=config["model"]["num_classes"])
        training_config = {
            **data_config,
            **config.get("training", {}),
            **config.get("evaluation", {}),
        }
        train_detection(model, train_ds, val_dataset=val_ds, config=training_config)

    elif model_name in ["unet", "attention_unet"]:
        image_dir = data_config.get(
            "image_dir", f"data/processed/{dataset_name}/images"
        )
        mask_dir = data_config.get(
            "mask_dir", f"data/processed/{dataset_name}/masks"
        )
        split_dir = data_config.get(
            "split_dir", f"data/splits/{dataset_name}"
        )
        train_files = _read_split(split_dir, "train")
        val_files = _read_split(split_dir, "val")
        train_transforms = (
            get_chest_xray_train_transforms(image_size)
            if dataset_name == "chest_xray"
            else get_train_transforms(image_size)
        )

        train_ds = SegmentationDataset(
            image_dir=image_dir,
            mask_dir=mask_dir,
            file_list=train_files,
            transforms=train_transforms,
        )
        val_ds = SegmentationDataset(
            image_dir=image_dir,
            mask_dir=mask_dir,
            file_list=val_files,
            transforms=get_val_transforms(image_size),
        )
        model_config = config["model"]
        model_args = {
            "in_ch": model_config.get("in_channels", 3),
            "out_ch": model_config.get("out_channels", 1),
            "features": tuple(model_config.get("features", [64, 128, 256, 512])),
        }
        model = (
            UNet(**model_args)
            if model_name == "unet"
            else AttentionUNet(**model_args)
        )
        training_config = {**data_config, **config.get("training", {})}
        train_segmentation(
            model,
            train_ds,
            val_dataset=val_ds,
            config=training_config,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")


if __name__ == "__main__":
    # Ensure output dir exists
    Path("outputs/checkpoints").mkdir(parents=True, exist_ok=True)
    Path("outputs/logs").mkdir(parents=True, exist_ok=True)
    main()
