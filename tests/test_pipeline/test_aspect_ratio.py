import numpy as np
import torch

from src.pipeline.inference import letterbox_image, run_segmentation
from src.pipeline.full_pipeline import MedSegPipeline


class _CenterMask(torch.nn.Module):
    def forward(self, images):
        batch, _, height, width = images.shape
        logits = torch.full((batch, 1, height, width), -20.0, device=images.device)
        logits[:, :, height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = 20.0
        return logits


def test_letterbox_preserves_landscape_aspect_ratio():
    image = np.zeros((100, 400, 3), dtype=np.uint8)
    padded, geometry = letterbox_image(image, 256)

    assert padded.shape == (256, 256, 3)
    assert geometry["resized_width"] == 256
    assert geometry["resized_height"] == 64
    assert geometry["pad_top"] == 96


def test_segmentation_restores_original_portrait_shape():
    image = np.zeros((480, 160, 3), dtype=np.uint8)
    mask = run_segmentation(_CenterMask(), image, size=256)

    assert mask.shape == image.shape[:2]
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 1})


def test_roi_expansion_is_clipped_to_image_boundaries():
    box = MedSegPipeline.expand_and_clip_box(
        [-5, 8, 95, 110], (100, 80, 3), margin=0.1
    )

    assert box == [0, 0, 80, 100]
