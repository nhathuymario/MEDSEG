"""Tests for Attention U-Net."""
import torch
import pytest
from src.models.attention_unet import AttentionUNet
from src.models.components.attention_gate import AttentionGate


@pytest.fixture
def model():
    return AttentionUNet(in_ch=3, out_ch=1, features=(32, 64, 128, 256))


def test_creation(model):
    assert model is not None


def test_forward(model):
    x = torch.rand(2, 3, 256, 256)
    out = model(x)
    assert out.shape == (2, 1, 256, 256)


def test_attention_gate():
    ag = AttentionGate(F_g=64, F_l=64, F_int=32)
    g = torch.rand(1, 64, 16, 16)
    x = torch.rand(1, 64, 16, 16)
    out = ag(g, x)
    assert out.shape == x.shape
    assert out.min() >= 0  # sigmoid output


def test_same_interface_as_unet(model):
    """Attention U-Net should have same I/O as U-Net."""
    from src.models.unet import UNet
    unet = UNet(in_ch=3, out_ch=1, features=(32, 64, 128, 256))
    x = torch.rand(1, 3, 128, 128)
    assert model(x).shape == unet(x).shape
