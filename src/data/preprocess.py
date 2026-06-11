"""Preprocessing: resize, normalize, generate bboxes from masks."""
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def resize_and_save(src_dir: Path, dst_dir: Path, size=(256, 256), ext="*.jpg"):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for img_path in tqdm(sorted(src_dir.glob(ext))):
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, size)
        cv2.imwrite(str(dst_dir / img_path.name), img)


def mask_to_binary(mask_dir: Path, out_dir: Path, size=(256, 256)):
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in tqdm(sorted(mask_dir.glob("*.png"))):
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        m = cv2.resize(m, size)
        m = (m > 127).astype(np.uint8) * 255
        cv2.imwrite(str(out_dir / p.name), m)


def mask_to_bbox(mask: np.ndarray) -> list:
    """Extract bounding box [x1, y1, x2, y2] from binary mask."""
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return [0, 0, 1, 1]
    return [
        int(coords[1].min()),
        int(coords[0].min()),
        int(coords[1].max()),
        int(coords[0].max()),
    ]


def generate_coco_annotations(image_dir: Path, mask_dir: Path, output_path: Path):
    """Generate COCO-format JSON annotations from masks."""
    import json

    images, annotations = [], []
    ann_id = 1
    for idx, img_path in enumerate(sorted(image_dir.glob("*.*"))):
        mask_name = img_path.stem + "_segmentation.png"
        mask_path = mask_dir / mask_name
        if not mask_path.exists():
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        bbox = mask_to_bbox(mask)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

        images.append({"id": idx, "file_name": img_path.name, "width": mask.shape[1], "height": mask.shape[0]})
        annotations.append({"id": ann_id, "image_id": idx, "category_id": 1, "bbox": [bbox[0], bbox[1], w, h], "area": w * h, "iscrowd": 0})
        ann_id += 1

    coco = {"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "lesion"}]}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)
    print(f"Saved {len(annotations)} annotations to {output_path}")


def prepare_chest_xray(
    raw_dir: Path = Path("data/raw/chest_xray/montgomery"),
    output_dir: Path = Path("data/processed/chest_xray"),
    size=(256, 256),
):
    """Prepare Montgomery X-rays and combine left/right lung masks."""
    image_dir = raw_dir / "CXR_png"
    left_mask_dir = raw_dir / "ManualMask" / "leftMask"
    right_mask_dir = raw_dir / "ManualMask" / "rightMask"
    output_image_dir = output_dir / "images"
    output_mask_dir = output_dir / "masks"

    required_dirs = (image_dir, left_mask_dir, right_mask_dir)
    missing_dirs = [str(path) for path in required_dirs if not path.is_dir()]
    if missing_dirs:
        raise FileNotFoundError(
            "Missing Montgomery dataset directories: " + ", ".join(missing_dirs)
        )

    image_paths = sorted(image_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No X-ray images found in {image_dir}")

    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    prepared = 0
    for image_path in tqdm(image_paths, desc="Preprocessing Chest X-ray"):
        left_path = left_mask_dir / image_path.name
        right_path = right_mask_dir / image_path.name
        if not left_path.exists() or not right_path.exists():
            raise FileNotFoundError(f"Missing lung mask for {image_path.name}")

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        left_mask = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
        right_mask = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)
        if image is None or left_mask is None or right_mask is None:
            raise ValueError(f"Could not read image or masks for {image_path.name}")

        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        image = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        left_mask = cv2.resize(left_mask, size, interpolation=cv2.INTER_NEAREST)
        right_mask = cv2.resize(right_mask, size, interpolation=cv2.INTER_NEAREST)
        combined_mask = ((left_mask > 127) | (right_mask > 127)).astype(
            np.uint8
        ) * 255

        cv2.imwrite(str(output_image_dir / image_path.name), image)
        cv2.imwrite(str(output_mask_dir / image_path.name), combined_mask)
        prepared += 1

    print(f"Prepared {prepared} Chest X-ray image/mask pairs in {output_dir}")
    return prepared


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess MedSeg datasets")
    parser.add_argument(
        "--dataset",
        choices=("chest_xray",),
        required=True,
    )
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()

    if args.dataset == "chest_xray":
        prepare_chest_xray(size=(args.size, args.size))
