"""Tests for U-Net."""
import torch
import pytest
from src.models.unet import UNet


@pytest.fixture
def model():
    return UNet(in_ch=3, out_ch=1, features=(32, 64, 128, 256))


def test_creation(model):
    assert model is not None


def test_forward(model):
    x = torch.rand(2, 3, 256, 256)
    out = model(x)
    assert out.shape == (2, 1, 256, 256)


def test_single_channel():
    m = UNet(in_ch=1, out_ch=1, features=(32, 64))
    out = m(torch.rand(1, 1, 64, 64))
    assert out.shape == (1, 1, 64, 64)


def test_gradient_flow(model):
    x = torch.rand(1, 3, 128, 128, requires_grad=True)
    out = model(x)
    out.sum().backward()
    assert x.grad is not None
