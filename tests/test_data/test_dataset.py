import pytest
import os
import torch
import cv2
import numpy as np
from src.data.dataset_segmentation import SegmentationDataset

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
