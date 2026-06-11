"""Tests for full pipeline."""
import torch
import numpy as np
import pytest
from src.models.faster_rcnn import LesionDetector
from src.models.attention_unet import AttentionUNet
from src.pipeline.full_pipeline import MedSegPipeline


@pytest.fixture
def pipeline():
    det = LesionDetector(num_classes=2, pretrained=False)
    seg = AttentionUNet(in_ch=3, out_ch=1, features=(32, 64, 128, 256))
    return MedSegPipeline(det, seg, device="cpu")


def test_pipeline_creation(pipeline):
    assert pipeline is not None


def test_pipeline_run(pipeline):
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    results = pipeline.run(img)
    assert "detection" in results
    assert "full_mask" in results
    assert "time_ms" in results


def test_visualization(pipeline):
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    results = pipeline.run(img)
    vis = pipeline.visualize(img, results)
    assert vis.shape == img.shape
