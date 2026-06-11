"""Tests for Faster R-CNN LesionDetector."""
import torch
import pytest
from src.models.faster_rcnn import LesionDetector


@pytest.fixture
def model():
    return LesionDetector(num_classes=2, pretrained=False)


def test_model_creation(model):
    assert model is not None


def test_forward_training(model):
    model.train()
    images = [torch.rand(3, 256, 256)]
    targets = [{"boxes": torch.tensor([[50, 50, 150, 150]], dtype=torch.float32), "labels": torch.tensor([1])}]
    losses = model(images, targets)
    assert "loss_classifier" in losses
    assert "loss_box_reg" in losses


def test_forward_inference(model):
    model.eval()
    images = [torch.rand(3, 256, 256)]
    with torch.no_grad():
        preds = model(images)
    assert isinstance(preds, list)
    assert "boxes" in preds[0]
    assert "scores" in preds[0]


def test_output_shape(model):
    model.eval()
    with torch.no_grad():
        preds = model([torch.rand(3, 128, 128)])[0]
    assert preds["boxes"].dim() == 2
    assert preds["boxes"].shape[1] == 4
