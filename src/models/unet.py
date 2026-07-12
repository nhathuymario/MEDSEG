"""U-Net chuẩn cho phân đoạn ảnh y khoa."""
import torch
import torch.nn as nn
from .components import DoubleConv, DownBlock, UpBlock


class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, features=(64, 128, 256, 512)):
        super().__init__()
        # Encoder giảm dần kích thước không gian và tăng số kênh đặc trưng.
        self.inc = DoubleConv(in_ch, features[0])
        self.downs = nn.ModuleList([DownBlock(features[i], features[i + 1]) for i in range(len(features) - 1)])
        self.bottleneck = DownBlock(features[-1], features[-1] * 2)

        # Decoder khôi phục kích thước ảnh và nối skip connection từ encoder.
        self.ups = nn.ModuleList()
        rev = list(reversed(features))
        self.ups.append(UpBlock(features[-1] * 2, rev[0]))  # bottleneck -> last feature
        for i in range(1, len(rev)):
            self.ups.append(UpBlock(rev[i - 1], rev[i]))

        self.outc = nn.Conv2d(features[0], out_ch, 1)

    def forward(self, x):
        # Lưu feature map ở từng tầng encoder để decoder dùng làm skip connection.
        skips = [self.inc(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))
        x = self.bottleneck(skips[-1])
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)
        return self.outc(x)
