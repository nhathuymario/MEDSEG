"""Các block convolution tái sử dụng cho U-Net."""
import torch.nn as nn


class DoubleConv(nn.Module):
    """Hai lớp Conv2d liên tiếp, mỗi lớp kèm BatchNorm và ReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DownBlock(nn.Module):
    """Giảm kích thước H/W bằng MaxPool rồi trích đặc trưng bằng DoubleConv."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """Upsample bằng ConvTranspose2d, nối skip connection, rồi DoubleConv."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        import torch
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)
