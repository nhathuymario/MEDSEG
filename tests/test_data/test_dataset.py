import pytest
import os
import torch
import cv2
import numpy as np
import json
from src.data.dataset_detection import DetectionDataset
from src.data.dataset_segmentation import SegmentationDataset
from src.data.transforms import get_detection_transforms

def test_segmentation_dataset_creation(tmp_path):
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)

    cv2.imwrite(str(img_dir / "1.jpg"), img)
    cv2.imwrite(str(mask_dir / "1.png"), mask)

    ds = SegmentationDataset(str(img_dir), str(mask_dir))
    assert len(ds) == 1
    
    img_t, mask_t = ds[0]
    assert img_t.shape == (3, 256, 256)
    assert mask_t.shape == (1, 256, 256)


def test_detection_dataset_clips_boxes_to_image_bounds(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    cv2.imwrite(str(image_dir / "clinical.jpg"), np.zeros((100, 200, 3), dtype=np.uint8))
    annotations = {
        "images": [{"id": 1, "file_name": "clinical.jpg", "width": 200, "height": 100}],
        "annotations": [
            {"id": 1, "image_id": 1, "bbox": [180.0, 90.0, 25.0, 15.0]},
        ],
        "categories": [{"id": 0, "name": "lesion"}],
    }
    annotation_file = tmp_path / "labels.json"
    annotation_file.write_text(json.dumps(annotations), encoding="utf-8")

    dataset = DetectionDataset(
        image_dir,
        annotation_file,
        transforms=get_detection_transforms(size=256, train=False),
    )
    _, target = dataset[0]

    assert target["boxes"].shape == (1, 4)
    assert torch.all(target["boxes"] >= 0)
    assert torch.all(target["boxes"] <= 256)
