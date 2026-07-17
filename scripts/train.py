"""Main training script."""
import argparse
import atexit
import ctypes
import csv
import os
import sys
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def _pid_exists(pid):
    if os.name == "nt":
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _acquire_training_lock(config_path):
    """Prevent two expensive training jobs for the same config from running."""
    lock_dir = Path("outputs/system")
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{Path(config_path).stem}.train.lock"

    if lock_path.exists():
        try:
            existing_pid = int(lock_path.read_text(encoding="utf-8").strip())
        except ValueError:
            existing_pid = None
        if existing_pid and _pid_exists(existing_pid):
            raise RuntimeError(
                f"Training is already running for {config_path} (PID {existing_pid}). "
                "Do not start a duplicate job."
            )
        lock_path.unlink(missing_ok=True)

    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(
            f"Training is already starting for {config_path}."
        ) from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))

    def release_lock():
        try:
            if lock_path.read_text(encoding="utf-8").strip() == str(os.getpid()):
                lock_path.unlink(missing_ok=True)
        except FileNotFoundError:
            pass

    atexit.register(release_lock)
    return lock_path


def _read_split(split_dir, split_name):
    split_path = Path(split_dir) / f"{split_name}.csv"
    if not split_path.exists():
        raise FileNotFoundError(
            f"Missing split file: {split_path}. Create dataset splits first."
        )
    with open(split_path, newline="") as handle:
        return [row["filename"] for row in csv.DictReader(handle)]


def _detection_dataset(source, split_name, image_size, train, augmentation="default"):
    from src.data.dataset_detection import DetectionDataset
    from src.data.transforms import get_detection_transforms

    split_dir = source["split_dir"]
    files = _read_split(split_dir, split_name)
    return DetectionDataset(
        image_dir=source["image_dir"],
        annotation_file=source["annotation_file"],
        file_list=files,
        transforms=get_detection_transforms(size=image_size, train=train, augmentation=augmentation),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config yaml")
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override the number of training epochs",
    )
    args = parser.parse_args()
    _acquire_training_lock(args.config)

    # Import ML dependencies only after acquiring the lock. This prevents a
    # duplicate command from loading PyTorch/models and exhausting RAM before
    # it is rejected.
    from torch.utils.data import ConcatDataset
    from src.data.dataset_segmentation import SegmentationDataset
    from src.data.transforms import (
        get_chest_xray_train_transforms,
        get_train_transforms,
        get_val_transforms,
    )
    from src.models.faster_rcnn import LesionDetector
    from src.models.unet import UNet
    from src.models.attention_unet import AttentionUNet
    from src.training.train_detection import train_detection
    from src.training.train_segmentation import train_segmentation

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
        sources = data_config.get("sources") or [{
            "image_dir": data_config.get(
                "image_dir", "data/processed/isic2018/images"
            ),
            "annotation_file": data_config.get(
                "annotation_file",
                "data/processed/isic2018/annotations/train.json",
            ),
            "split_dir": data_config.get(
                "split_dir", f"data/splits/{dataset_name}"
            ),
            "train_split": "train",
            "val_split": "val",
            "use_for_validation": True,
        }]
        augmentation = config.get("training", {}).get("augmentation", "default")
        train_sets = [
            _detection_dataset(
                source,
                source.get("train_split", "train"),
                image_size,
                train=True,
                augmentation=augmentation,
            )
            for source in sources
        ]
        val_sets = [
            _detection_dataset(
                source,
                source.get("val_split", "val"),
                image_size,
                train=False,
            )
            for source in sources
            if source.get("use_for_validation", True)
            and source.get("val_split")
        ]
        train_ds = train_sets[0] if len(train_sets) == 1 else ConcatDataset(train_sets)
        val_ds = (
            val_sets[0]
            if len(val_sets) == 1
            else ConcatDataset(val_sets)
            if val_sets
            else None
        )
        model_config = config["model"]
        initial_checkpoint = config.get("training", {}).get("initial_checkpoint")
        model = (
            LesionDetector.from_checkpoint(
                initial_checkpoint,
                num_classes=model_config["num_classes"],
            )
            if initial_checkpoint
            else LesionDetector(
                num_classes=model_config["num_classes"],
                pretrained=model_config.get("pretrained", True),
            )
        )
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
    Path("outputs/detection/checkpoints").mkdir(parents=True, exist_ok=True)
    Path("outputs/detection/logs").mkdir(parents=True, exist_ok=True)
    Path("outputs/segmentation/skin/checkpoints").mkdir(parents=True, exist_ok=True)
    Path("outputs/segmentation/skin/logs").mkdir(parents=True, exist_ok=True)
    main()
