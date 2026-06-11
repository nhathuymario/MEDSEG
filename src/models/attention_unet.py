"""Attention U-Net: U-Net + Attention Gates on skip connections."""
import torch
import torch.nn as nn
from .components import DoubleConv, AttentionGate


class AttentionUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, features=(64, 128, 256, 512)):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        # Encoder
        self.enc = nn.ModuleList()
        ch = in_ch
        for f in features:
            self.enc.append(DoubleConv(ch, f))
            ch = f

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder (with attention gates)
        self.ups = nn.ModuleList()
        self.attn = nn.ModuleList()
        self.dec = nn.ModuleList()
        rev = list(reversed(features))
        prev = features[-1] * 2
        for f in rev:
            self.ups.append(nn.ConvTranspose2d(prev, f, kernel_size=2, stride=2))
            self.attn.append(AttentionGate(F_g=f, F_l=f, F_int=f // 2))
            self.dec.append(DoubleConv(f * 2, f))
            prev = f

        self.outc = nn.Conv2d(features[0], out_ch, 1)

    def forward(self, x):
        # Encoder
        skips = []
        for enc in self.enc:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        # Decoder
        for up, attn, dec, skip in zip(self.ups, self.attn, self.dec, reversed(skips)):
            x = up(x)
            skip = attn(g=x, x=skip)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return self.outc(x)
