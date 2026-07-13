"""Prepare official iToBoS train/test filename splits for detection."""
import argparse
import csv
import json
from pathlib import Path


def prepare_itobos(
    source_split: Path,
    annotation_file: Path,
    output_dir: Path,
):
    with open(annotation_file, encoding="utf-8") as handle:
        coco = json.load(handle)
    available = {Path(item["file_name"]).stem: item["file_name"] for item in coco["images"]}

    grouped = {"train": [], "test": []}
    with open(source_split, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            split = row["split"].strip().lower()
            image_id = row["isic_id"].strip()
            if split not in grouped:
                continue
            if image_id not in available:
                raise ValueError(f"Missing iToBoS annotation for {image_id}")
            grouped[split].append(available[image_id])

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, filenames in grouped.items():
        destination = output_dir / f"{split}.csv"
        with open(destination, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["filename"])
            writer.writeheader()
            writer.writerows({"filename": name} for name in filenames)
        print(f"{split}: {len(filenames)} images -> {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-split",
        type=Path,
        default=Path("data/raw/clinical_skin/itobos/supplements/split.csv"),
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/raw/clinical_skin/itobos/supplements/labels.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/splits/itobos"),
    )
    args = parser.parse_args()
    prepare_itobos(args.source_split, args.annotations, args.output_dir)
